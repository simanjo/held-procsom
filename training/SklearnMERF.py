import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from merf import MERF

class SklearnMERF(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            z_col,
            cluster_col,
            max_iterations=30,
            n_estimators=100,
            max_depth=None,
            criterion= 'squared_error',
            min_samples_split = 2,
            min_samples_leaf = 1,
            max_leaf_nodes = 50,
            bootstrap = True,
            **rf_params
        ):
        self.z_col = z_col
        self.cluster_col = cluster_col
        self.max_iterations = max_iterations
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.bootstrap = bootstrap
        self.rf_params = rf_params
            
    def fit(self, X, y):
        df = pd.DataFrame(X).copy()
        
        self.clusters = np.asarray(df.iloc[:, self.cluster_col])
        self.Z = np.asarray(df.iloc[:, [self.z_col]])
        
        # self.X = df.drop(columns=['Z_intercept', 'cluster_ids'])
        self.X = df.drop(columns=[df.columns[self.z_col], df.columns[self.cluster_col]])
        
        rf_base = RandomForestRegressor(
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth,
            criterion = self.criterion, # type: ignore
            min_samples_split = self.min_samples_split,
            min_samples_leaf = self.min_samples_leaf,
            max_leaf_nodes = self.max_leaf_nodes,
            bootstrap = self.bootstrap,
            **self.rf_params
        )
        
        self.merf = MERF(fixed_effects_model=rf_base, max_iterations=self.max_iterations)
        self.merf.fit(self.X, self.Z, self.clusters, y)
        self._is_fitted = True
        return self
        
    def predict(self, X):
        check_is_fitted(self)
        df = pd.DataFrame(X).copy()
        
        return self.merf.predict(
            X=df.drop(columns=[df.columns[self.z_col], df.columns[self.cluster_col]]),
            Z=np.asarray(df.iloc[:, [self.z_col]]),
            clusters=np.asarray(df.iloc[:, self.cluster_col]),
        )

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
