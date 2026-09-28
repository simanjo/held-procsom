import itertools
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from joblib import Memory

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_validate

from training.train_utils import DataSettings, ModelSettings, get_data, model_name, model_path
from training.grids import get_full_grid


EMBEDDINGS = ['maccs', 'rdkit', 'pubchem', 'mol2vec', 'abrahams']
R2_CUTOFF = 0.85
SAVE_PATH = Path(os.getcwd()) / 'models' / 'leaveXout' / "test"
DATA_PATH = Path(os.getcwd()) / 'data'

# We need to patch the fit and predict methods of the merf package.
# As the package was not initially designed to act as sklearn estimator
# stuff breaks inside a nested cv.
# This is caused by an index mismatch on the cluster array, which MUST be a
# pandas Series in the merf package, but nesting group leave-outs destroys indizes
# and we need the cluster given as numpy array instead.
# To avoid rewriting the whole function, we perform a dirty live patch
# of the predict function when using it, by applying a patch to the function
# source code interpreted as temporary file (this approach is due to gemini).
# The patch diffs are in merf_fit.diff resp. merf_predict.diff.

# To avaoid hiding this, we do this whenever we use the merf package and the
# associated SklearnMERF estimator, instead of hiding the patching close to
# the class.

import inspect
import tempfile
import textwrap
import sys

import patch_ng
import merf

def patch_merf_fit():
    fit_source = textwrap.dedent(inspect.getsource(getattr(merf.MERF, "fit")))

    patch_set = patch_ng.fromfile("training/merf_fit.diff")

    # use a tempfile to apply patch to
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', encoding='utf-8', delete=False) as fh:
        fh.write(fit_source)
        tmp_path = fh.name

    for single_patch in patch_set.items:
        # change patch file name in place
        single_patch.source = tmp_path.encode('utf-8')
        single_patch.target = tmp_path.encode('utf-8')

    try:
        if not patch_set.apply(root=os.path.dirname(tmp_path)):
            raise RuntimeError(f"Failed to cleanly apply the diff to the fit function!")

        with open(tmp_path, 'r') as fh:
            fit_patched = fh.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    merf_globals = sys.modules[merf.MERF.__module__].__dict__
    exec(fit_patched, merf_globals)
    setattr(merf.MERF, 'fit', merf_globals['fit'])


if __name__ == '__main__':
    patch_merf_fit()
    from training.SklearnMERF import SklearnMERF 

    sklearn.set_config(enable_metadata_routing=True)
    data_setting = DataSettings(
        r2_cutoff=R2_CUTOFF,
        conc_range=False,
        salinity='all_water',
        temperature=True,
        sorbent='name'
    )
    model_settings = (
            ModelSettings('rf', emb, 'kf') for emb in EMBEDDINGS # type: ignore
        )
    data_path = DATA_PATH
    target_cache = {}
    current_data = None
    for model_setting in model_settings:
        model_dir = model_path(
            data_setting,
            path_prefix=SAVE_PATH / 'publication_merf'
            )

        print(f' Starting {model_setting}')

        # skip calculation of data if a similarly named binary exists
        # instead of preparing data and skipping later on
        if (model_dir / model_name(model_setting)).is_file():
            print('  Skipped preparing dataset for ' + model_name(model_setting) +' as a similarly named model binary already exists.')
            continue
        X, y, groups = target_cache.setdefault(
            (model_setting.embedding, model_setting.target, data_setting, 'publication'),
            get_data(data_setting, model_setting, data_path, group_split='publication')
        )
        # add merf additional columns
        X['Z_intercept'] = 1.0
        X['cluster_ids'] = groups

        # check embedding columns for duplicates and drop them if necessary
        # HACK: need to provide cols as index and not name
        if model_setting.embedding == 'abrahams':
            emb_cols = None
        else:
            all_emb_cols = [col for col in X.columns if col.startswith(model_setting.embedding)]
            single_emb_cols = X.loc[:, all_emb_cols].T.drop_duplicates().T.columns
            X = X.loc[:, [col for col in X.columns if not col.startswith(model_setting.embedding) or col in single_emb_cols]]
            emb_cols =  [i for i, col in enumerate(X.columns) if col.startswith(model_setting.embedding)]
        # get col_ids for normal cols and merf additional cols (should be len(X.columns)+1/2)
        scaling_cols = []
        Z_col, cluster_col = None, None
        for i, col in enumerate(X.columns):
            if col == 'Z_intercept':
                Z_col = i
            elif col == 'cluster_ids':
                cluster_col = i
            else:
                scaling_cols.append(i)
        # correct for dim reduction in case of non-Abrahams embeddings
        if model_setting.embedding != 'abrahams':
            Z_col = Z_col - len(emb_cols) + 10
            cluster_col = cluster_col - len(emb_cols) + 10

        cachedir = tempfile.mkdtemp()
        mem = Memory(location=cachedir, verbose=0)
        pipe = Pipeline(
            steps=[
                ('scaling', ColumnTransformer(
                    transformers=[('scaling', StandardScaler(), scaling_cols)],
                    remainder = 'passthrough'
                    ),
                ),
                ('dim_reduction', 'passthrough'),
                ('classifier', SklearnMERF(z_col=Z_col, cluster_col=cluster_col))
            ],
            memory=mem)

        os.makedirs(model_dir, exist_ok=True)
        if (model_dir / model_name(model_setting)).is_file():
            print('  Skipped fitting ' + model_name(model_setting) +' as a similarly named model binary already exists.')
            break
        print('  Start CV for ' + model_name(model_setting))
        inner_cv = LeaveOneGroupOut()
        outer_cv = LeaveOneGroupOut()
        param_grid = get_full_grid(
                model_setting.model,
                is_multimodel=False,
                embedding_cols=emb_cols,
                dim_red_components= [10]
                )
        clf = GridSearchCV(
            estimator=pipe, param_grid=param_grid,
            cv=inner_cv, n_jobs=-1, error_score='raise',
            scoring='neg_root_mean_squared_error',
            return_train_score=True
        )
        # add Z intercept and clusters to X
        score = cross_validate(
            clf, X=X, y=y, cv=outer_cv,
            scoring='neg_root_mean_squared_error',
            return_train_score=True,
            return_estimator=True,
            return_indices=True, # type: ignore,
            params={'groups': groups},
        )
        with open(model_dir / model_name(model_setting), 'wb') as fh:
            pickle.dump(((X, y), clf, param_grid, (inner_cv, outer_cv), score), fh)
        print('  Successfully pickled model binary for ' + model_name(model_setting))
        print(f'  With Mean: {score['test_score'].mean()} (Median: {np.median(score['test_score'])}) +- {2*score['test_score'].std()}')

        shutil.rmtree(cachedir, ignore_errors=True)
