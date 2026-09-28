import itertools
import os
import shutil
import tempfile
from pathlib import Path
from joblib import Memory

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from training.training import run_and_save
from training.train_utils import get_data, get_estimator_from_model, model_name, model_path
from training.train_utils import DataSettings, ModelSettings
from training.grids import get_full_grid

EMBEDDINGS = ['abrahams', 'maccs', 'rdkit', 'pubchem', 'mol2vec']
MODEL_CLASSES = ['krr', 'rf', 'gp', 'mlp']
R2_CUTOFF = 0.85
SAVE_PATH = Path(os.getcwd()) / 'models' / 'leavexout'
DATA_PATH = Path(os.getcwd()) / 'data'


if __name__=='__main__':
    data_setting = DataSettings(
        r2_cutoff=R2_CUTOFF,
        conc_range=False,
        salinity='all_water',
        temperature=True,
        sorbent='name'
    )
    model_settings = (
        ModelSettings(model, emb, tar) # type: ignore
        for model, emb, tar in itertools.product(
            MODEL_CLASSES, EMBEDDINGS, ['kf', 'n']
        )
    )
    # pd.options.mode.copy_on_write = True
    data_path = DATA_PATH
    target_cache = {}
    current_data = None
    for model_setting, group_split in itertools.product(model_settings, ('publication', 'polymer', 'sorbate')):
        model_dir = model_path(
            data_setting,
            path_prefix=SAVE_PATH / group_split
            )

        print(f' Starting {model_setting} with {group_split}')

        # skip calculation of data if a similarly named binary exists
        # instead of preparing data and skipping later on
        if (model_dir / model_name(model_setting)).is_file():
            print('  Skipped preparing dataset for ' + model_name(model_setting) +' as a similarly named model binary already exists.')
            continue
        X, y, groups = target_cache.setdefault(
            (model_setting.embedding, model_setting.target, data_setting, group_split),
            get_data(data_setting, model_setting, data_path, group_split=group_split)
        )

        # check embedding columns for duplicates and drop them if necessary
        # HACK: need to provide cols as index and not name
        if model_setting.embedding == 'abrahams':
            emb_cols = None
        else:
            all_emb_cols = [col for col in X.columns if col.startswith(model_setting.embedding)]
            single_emb_cols = X.loc[:, all_emb_cols].T.drop_duplicates().T.columns
            X = X.loc[:, [col for col in X.columns if not col.startswith(model_setting.embedding) or col in single_emb_cols]]
            emb_cols =  [i for i, col in enumerate(X.columns) if col.startswith(model_setting.embedding)]

        cachedir = tempfile.mkdtemp()
        mem = Memory(location=cachedir, verbose=0)
        pipe = Pipeline(steps=[
            ('scaling', StandardScaler()),
            ('dim_reduction', 'passthrough'),
            ('classifier', get_estimator_from_model(model_setting))
            ],
            memory=mem)

        run_and_save(
            data_setting=data_setting,
            model_setting=model_setting,
            pipe=pipe,
            grid=get_full_grid(
                model_setting.model,
                is_multimodel=False,
                embedding_cols=emb_cols,
                dim_red_components= [5, 10]
                ),
            X=X, y=np.asarray(y),
            groups=groups,
            save_path=SAVE_PATH / group_split,
            # save_suffix=save_suffix
        )
        shutil.rmtree(cachedir, ignore_errors=True)