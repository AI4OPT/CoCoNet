"""Feature and target preprocessing, fitted on D_pred only.

Features: near-constant columns dropped; continuous columns median-imputed and standardized;
categorical columns most-frequent-imputed and one-hot encoded. Targets: standardized per column.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONSTANT_STD = 1e-8


def constant_columns(X):
    dropped = []
    for col in X.columns:
        s = X[col]
        if s.nunique(dropna=True) <= 1:
            dropped.append(col)
        elif pd.api.types.is_numeric_dtype(s) and float(np.nanstd(s.to_numpy(dtype=float))) < CONSTANT_STD:
            dropped.append(col)
    return dropped


class Preprocessor:
    def __init__(self, categorical_columns):
        self.categorical_input = list(categorical_columns)

    def fit(self, X, y):
        self.dropped = constant_columns(X)
        X = X.drop(columns=self.dropped)
        categorical = [c for c in self.categorical_input if c in X.columns]
        continuous = [c for c in X.columns if c not in categorical]
        self.transformer = ColumnTransformer([
            ("continuous", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), continuous),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ], remainder="drop").fit(X)
        y = y.to_numpy(dtype=float)
        self.target_mean = np.mean(y, axis=0)
        self.target_std = np.std(y, axis=0)
        if np.any(self.target_std < CONSTANT_STD):
            raise ValueError("a target column is constant on D_pred")
        return self

    def transform_features(self, X):
        arr = self.transformer.transform(X.drop(columns=[c for c in self.dropped if c in X.columns]))
        if not np.all(np.isfinite(arr)):
            raise ValueError("non-finite feature after preprocessing")
        return arr.astype(np.float32)

    def transform_target(self, y):
        return ((y.to_numpy(dtype=float) - self.target_mean) / self.target_std).astype(np.float32)
