"""Feature and target preprocessing, fit on D_pred only and applied to every split.

Features: near-constant columns dropped, continuous columns median-imputed and standardized,
categorical columns most-frequent-imputed and one-hot encoded. Targets: standardized per
coordinate (optionally after log1p).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from coconet.utils import get_logger

logger = get_logger(__name__)

_NEAR_CONSTANT_STD_TOL = 1e-8


class PreprocessingError(RuntimeError):
    pass


class Preprocessor:
    """Fits on D_pred only; transforms D_pred/D_score/D_cal/D_test identically."""

    def __init__(self, categorical_columns: list[str], log1p_target: bool = False) -> None:
        self.categorical_columns_input = list(categorical_columns)
        self.log1p_target = log1p_target
        self._column_transformer: ColumnTransformer | None = None
        self._dropped_constant_columns: list[str] = []
        self._continuous_columns: list[str] = []
        self._categorical_columns: list[str] = []
        self._target_mean: np.ndarray = np.zeros(1)
        self._target_std: np.ndarray = np.ones(1)
        self._target_is_multivariate: bool = False
        self._fitted = False
        self._fitted_on_n_rows = 0

    def fit(self, X_pred: pd.DataFrame, y_pred: pd.Series) -> "Preprocessor":
        if len(X_pred) == 0:
            raise PreprocessingError("Cannot fit preprocessor on an empty D_pred")

        self._dropped_constant_columns = _find_constant_columns(X_pred)
        if self._dropped_constant_columns:
            logger.info("Dropping near-constant columns (fit on D_pred): %s", self._dropped_constant_columns)

        X_kept = X_pred.drop(columns=self._dropped_constant_columns)
        self._categorical_columns = [c for c in self.categorical_columns_input if c in X_kept.columns]
        self._continuous_columns = [c for c in X_kept.columns if c not in self._categorical_columns]

        continuous_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("continuous", continuous_pipeline, self._continuous_columns),
                ("categorical", categorical_pipeline, self._categorical_columns),
            ],
            remainder="drop",
        )
        self._column_transformer.fit(X_kept)

        y_arr = y_pred.to_numpy(dtype=float)
        self._target_is_multivariate = y_arr.ndim == 2
        y_2d = y_arr if self._target_is_multivariate else y_arr.reshape(-1, 1)
        transformed_target = np.log1p(y_2d) if self.log1p_target else y_2d
        self._target_mean = np.mean(transformed_target, axis=0)
        self._target_std = np.std(transformed_target, axis=0)
        if np.any(self._target_std < _NEAR_CONSTANT_STD_TOL):
            raise PreprocessingError("Target has (near-)zero variance on D_pred; cannot standardize")

        self._fitted = True
        self._fitted_on_n_rows = len(X_pred)
        return self

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_kept = X.drop(columns=[c for c in self._dropped_constant_columns if c in X.columns])
        arr = self._column_transformer.transform(X_kept)
        if not np.all(np.isfinite(arr)):
            raise PreprocessingError("Non-finite values produced by feature transform")
        return arr.astype(np.float32)

    def transform_target(self, y: pd.Series | pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        y_arr = y.to_numpy(dtype=float)
        is_2d = y_arr.ndim == 2
        y_2d = y_arr if is_2d else y_arr.reshape(-1, 1)
        if self.log1p_target:
            if np.any(y_2d < 0):
                raise PreprocessingError("log1p_target requested but negative target values are present")
            y_2d = np.log1p(y_2d)
        y_std_2d = (y_2d - self._target_mean) / self._target_std
        y_std = y_std_2d if is_2d else y_std_2d.reshape(-1)
        return y_std.astype(np.float32)

    def inverse_transform_target(self, y_std: np.ndarray) -> np.ndarray:
        self._check_fitted()
        y_std_arr = np.asarray(y_std, dtype=float)
        is_2d = y_std_arr.ndim == 2
        y_std_2d = y_std_arr if is_2d else y_std_arr.reshape(-1, 1)
        y_2d = y_std_2d * self._target_std + self._target_mean
        if self.log1p_target:
            y_2d = np.expm1(y_2d)
        return y_2d if is_2d else y_2d.reshape(-1)

    def n_features_out(self) -> int:
        self._check_fitted()
        return len(self.feature_names_out())

    def feature_names_out(self) -> list[str]:
        self._check_fitted()
        return list(self._column_transformer.get_feature_names_out())

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise PreprocessingError("Preprocessor must be fit() on D_pred before use")


def _find_constant_columns(X: pd.DataFrame, std_tol: float = _NEAR_CONSTANT_STD_TOL) -> list[str]:
    """Columns with at most one distinct non-null value, or numeric std below `std_tol`."""
    dropped = []
    for col in X.columns:
        series = X[col]
        nunique = series.nunique(dropna=True)
        if nunique <= 1:
            dropped.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            std = float(np.nanstd(series.to_numpy(dtype=float)))
            if std < std_tol:
                dropped.append(col)
    return dropped
