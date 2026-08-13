"""Build an sklearn Pipeline that refits every transform inside each CV fold.

Why this exists: FORGE used to fit the imputer, scaler, one-hot encoder and clip
bounds on ALL training rows and only then cut cross-validation folds. Every
fold's validation rows had therefore helped shape the transform applied to them,
so the reported CV score was optimistic — measurably so (on a near-random target
with high-cardinality categoricals the leaky protocol reported 0.67 where the
unbiased one reported 0.44).

Passing ``build_cv_pipeline(recipe, estimator)`` raw training data to
``cross_val_score`` puts feature ops, clipping and preprocessing INSIDE the fold,
which is the unbiased protocol.

Not included: feature selection. Its SHAP and RFECV stages cost seconds per fit,
so refitting them per fold per HPO trial is computationally prohibitive. That
residual optimism is disclosed rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PipelineRecipe:
    """Everything needed to rebuild the preprocessing chain UNFITTED."""

    feature_specs: list = field(default_factory=list)
    numerical_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    clip_cols: list[str] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)


class ClipTransformer(BaseEstimator, TransformerMixin):
    """Clip numeric columns to the 1st/99th percentile learned at fit time.

    Fitting the bounds inside the fold is the point: bounds learned on the full
    training set would leak the validation rows' spread into their own clipping.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y: Any = None) -> ClipTransformer:
        self.bounds_: dict[str, tuple[float, float]] = {}
        for col in self.columns:
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                lo, hi = X[col].quantile([0.01, 0.99])
                if np.isfinite(lo) and np.isfinite(hi):
                    self.bounds_[col] = (float(lo), float(hi))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in getattr(self, "bounds_", {}).items():
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)
        return X


class CategoricalAsString(BaseEstimator, TransformerMixin):
    """Match the training-time convention: categoricals as strings, NaN->MISSING."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y: Any = None) -> CategoricalAsString:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].astype(str).replace("nan", "MISSING")
        return X


class SelectByName(BaseEstimator, TransformerMixin):
    """Keep a fixed set of post-preprocessing columns, by name.

    Feature SELECTION itself is not refitted per fold — its SHAP and RFECV stages
    are far too slow to run inside every fold of every HPO trial — so the chosen
    names are applied as a fixed subset. Names are sanitised the same way the
    fitted pipeline sanitises them, and missing names are tolerated: a fold whose
    training portion lacks a rare category simply won't produce that one-hot
    column, and dropping it is correct rather than an error.
    """

    def __init__(self, names: list[str] | None = None):
        self.names = names or []

    @staticmethod
    def _sanitise(cols) -> list[str]:
        import re

        return [re.sub(r"[\[\]<>]", "_", str(c)).strip() for c in cols]

    def fit(self, X: pd.DataFrame, y: Any = None) -> SelectByName:
        wanted = set(self._sanitise(self.names))
        self.keep_ = [c for c, s in zip(X.columns, self._sanitise(X.columns)) if s in wanted]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in getattr(self, "keep_", []) if c in X.columns]
        return X[keep] if keep else X


def build_preprocessor(numerical_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Construct the same ColumnTransformer the fitted pipeline uses, unfitted."""
    transformers = []
    if numerical_cols:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            numerical_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ))
    ct = ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )
    # pandas output keeps column NAMES through the chain, which is what lets the
    # selection step below address columns by name inside a fold.
    return ct.set_output(transform="pandas")


def build_cv_pipeline(recipe: PipelineRecipe, estimator: Any) -> Pipeline:
    """Unfitted pipeline: feature ops -> clip -> categorical cast -> preprocess -> model.

    Every step is unfitted, so ``cross_val_score`` refits the whole chain on each
    fold's training portion only.
    """
    from forge.feature_engineering.feature_ops import FeatureSpecTransformer

    steps: list[tuple[str, Any]] = []
    if recipe.feature_specs:
        steps.append(("ops", FeatureSpecTransformer(list(recipe.feature_specs))))
    if recipe.clip_cols:
        steps.append(("clip", ClipTransformer(list(recipe.clip_cols))))
    if recipe.categorical_cols:
        steps.append(("cast", CategoricalAsString(list(recipe.categorical_cols))))
    steps.append(("prep", build_preprocessor(recipe.numerical_cols, recipe.categorical_cols)))
    if recipe.selected_features:
        steps.append(("select", SelectByName(list(recipe.selected_features))))
    steps.append(("model", clone(estimator)))
    return Pipeline(steps)
