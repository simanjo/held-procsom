#%%
import itertools
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from joblib import Memory

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, LeaveOneGroupOut
from sklearn.pipeline import Pipeline

from training.train_utils import get_data, get_estimator_from_model, model_name, model_path
from training.train_utils import DataSettings, ModelSettings
from training.grids import get_full_grid
from training.isotherm_filters import has_enough_ooms, is_linear_fit, is_non_linear_fit

def run_and_save(
        data_setting, model_setting, pipe, grid,
        save_path, X, y, save_suffix=None,
        n_splits=7, random_state=1357911,
        groups=None
    ):

    save_path = model_path(data_setting, save_path)
    if save_suffix is not None:
        save_path /= save_suffix
    os.makedirs(save_path, exist_ok=True)
    name = model_name(model_setting)
    if (save_path / name).is_file():
        print('  Skipped fitting ' + name +' as a similarly named model binary already exists.')
        return
    print('  Start CV for ' + name)
    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if groups is None:
        outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state+1)
    else:
        outer_cv = LeaveOneGroupOut()
    clf = GridSearchCV(
        estimator=pipe, param_grid=grid,
        cv=inner_cv, n_jobs=-1, error_score='raise',
        scoring='neg_root_mean_squared_error',
        return_train_score=True
    )
    score = cross_validate(
        clf, X=X, y=y, cv=outer_cv,
        groups=groups,
        scoring='neg_root_mean_squared_error',
        return_train_score=True,
        return_estimator=True,
        return_indices=True # type: ignore
    )
    with open(save_path / name, 'wb') as fh:
        pickle.dump(((X, y), clf, grid, (inner_cv, outer_cv), score), fh)
    print('  Successfully pickled model binary for ' + name)
    print(f'  With Mean: {score['test_score'].mean()} (Median: {np.median(score['test_score'])}) +- {2*score['test_score'].std()}')

EMBEDDINGS = ['maccs', 'rdkit', 'pubchem', 'mol2vec', 'abrahams']
MODEL_CLASSES = ['rf', 'mlp', 'gp', 'pls', 'krr']
R2_CUTOFF = 0.85
SAVE_PATH = Path(os.getcwd()) / 'models' / 'emb_reduction'
DATA_PATH = Path(os.getcwd()) / 'data'


if __name__=='__main__':
    data_settings = (
        DataSettings(
            r2_cutoff=R2_CUTOFF,
            conc_range=crange,
            salinity=sal,
            temperature=temp,
            sorbent=sorb
        ) for sal, crange, temp, sorb in itertools.product(
            ['freshwater', 'all_water', 'seawater', 'all_vals'], [True, False], [True, False], ['name', 'comp']
        ))
    model_settings = (
        ModelSettings(model, emb, tar) # type: ignore
        for model, emb, tar in itertools.product(
            MODEL_CLASSES, EMBEDDINGS, ['kf', 'n']
        )
    )
    filter_settings = ['iso_fit_as_bool', has_enough_ooms, is_linear_fit, is_non_linear_fit]
    data_path = DATA_PATH
    target_cache = {}
    current_data = None
    # dim_red_settings = ([5, 10], 20, 50)#, 200)
    # for data_setting, model_setting, dim_red in itertools.product(data_settings, model_settings, dim_red_settings):
    for data_setting, model_setting, filt in itertools.product(data_settings, model_settings, filter_settings):
        filt_name = filt if isinstance(filt, str) else filt.__name__
        model_dir = model_path(
            data_setting,
            path_prefix=SAVE_PATH / filt_name
            )

        if data_setting == current_data:
            # print(f' Starting {model_setting}')
            print(f' Starting {model_setting} with {filt_name}')
            # print(f' Starting {model_setting} with {dim_red} maximal dimension reduction')
        else:
            # print(f'Starting hpt tuning for {data_setting} and {model_setting}.')
            print(f'Starting hpt tuning for {data_setting} and {model_setting} with {filt_name}.')
            # print(f'Starting hpt tuning for {data_setting} and {model_setting} with {dim_red} maximal dimension reduction')
            print(f'Storing pickles in directory {model_dir}')
            current_data = data_setting

        # if not isinstance(dim_red, list):
        #     model_dir /= f"emb_dim_{dim_red}"
        #     if model_setting.embedding == 'abrahams':
        #         # model is identical to the [5, 10, 20] case, as no dim-red happens
        #         assert (model_dir.parent / model_name(model_setting)).is_file()
        #         shutil.copy(model_dir.parent / model_name(model_setting), model_dir)
        #         print(f' Copied identical model from {model_dir.parent}, as no dimension reduction happens here.')
        #         continue

        # skip calculation of data if a similarly named binary exists
        # instead of preparing data and skipping later on
        if (model_dir / model_name(model_setting)).is_file():
            print('  Skipped preparing dataset for ' + model_name(model_setting) +' as a similarly named model binary already exists.')
            continue
        #X, y = target_cache.setdefault(
        #    (model_setting.embedding, model_setting.target, data_setting),
        #    get_data(data_setting, model_setting, data_path)
        #)
        X, y, groups = target_cache.setdefault(
            (model_setting.embedding, model_setting.target, data_setting, filt_name),
            get_data(data_setting, model_setting, data_path, filt)
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
        # if isinstance(dim_red, list):
        #     dim_red_components = dim_red
        #     save_suffix=None
        # else:
        #     # nested 7-fold cv has min(n_features) as follows
        #     dim_red_components = min(dim_red, int(int(X.shape[0]/7*6)/7*6))
        #     print(f"Reduced dim_red to {dim_red_components}")
        #     save_suffix=f"emb_dim_{dim_red}"
        
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
            save_path=SAVE_PATH / filt_name,
            # save_suffix=save_suffix
        )
        shutil.rmtree(cachedir, ignore_errors=True)

# %%
