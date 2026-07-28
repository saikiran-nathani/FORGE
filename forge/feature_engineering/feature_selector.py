"""Multi-stage feature selection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    RFECV,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.model_selection import StratifiedKFold

from forge.training.task_router import TaskType


@dataclass
class FeatureSelectionResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    selected_features: list[str]
    removed_features: list[dict[str, str]] = field(default_factory=list)
    importance_scores: dict[str, float] = field(default_factory=dict)


class FeatureSelector:
    """Correlation filter → MI → SHAP → RFECV selection."""

    def __init__(
        self,
        task_type: TaskType,
        target_column: str = "",
        random_state: int = 42,
        mi_threshold: float = 0.01,
        corr_threshold: float = 0.95,
    ):
        self.task_type = task_type
        self.target_column = target_column
        self.random_state = random_state
        self.mi_threshold = mi_threshold
        self.corr_threshold = corr_threshold

    def select(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        skip_rfe: bool = False,
    ) -> FeatureSelectionResult:
        removed: list[dict[str, str]] = []
        features = list(X_train.columns)

        features, removed_corr = self._correlation_filter(features, X_train, y_train)
        removed.extend(removed_corr)

        features, removed_mi = self._mutual_info_filter(features, X_train, y_train)
        removed.extend(removed_mi)

        importance = self._shap_importance(features, X_train, y_train)
        if importance:
            threshold = max(importance.values()) * 0.01
            kept = []
            for f in features:
                if importance.get(f, 0) >= threshold:
                    kept.append(f)
                else:
                    removed.append({"feature": f, "reason": "low_shap_importance"})
            features = kept

        if not skip_rfe and len(features) > 10:
            features, removed_rfe = self._rfe_filter(features, X_train, y_train)
            removed.extend(removed_rfe)

        return FeatureSelectionResult(
            X_train=X_train[features],
            X_test=X_test[features],
            selected_features=features,
            removed_features=removed,
            importance_scores=importance,
        )

    def _correlation_filter(
        self,
        features: list[str],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[list[str], list[dict[str, str]]]:
        removed = []
        corr = X[features].corr().abs()
        target_corr = X[features].corrwith(y).abs()
        drop: set[str] = set()

        for i, f1 in enumerate(features):
            if f1 in drop:
                continue
            for f2 in features[i + 1:]:
                if f2 in drop:
                    continue
                if corr.loc[f1, f2] > self.corr_threshold:
                    worse = f2 if target_corr.get(f1, 0) >= target_corr.get(f2, 0) else f1
                    drop.add(worse)
                    removed.append({"feature": worse, "reason": f"correlated_with_{f1 if worse == f2 else f2}"})

        return [f for f in features if f not in drop], removed

    def _mutual_info_filter(
        self,
        features: list[str],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[list[str], list[dict[str, str]]]:
        removed = []
        if self.task_type == TaskType.REGRESSION:
            mi = mutual_info_regression(X[features], y, random_state=self.random_state)
        else:
            mi = mutual_info_classif(X[features], y, random_state=self.random_state)

        kept = [f for f, score in zip(features, mi) if score >= self.mi_threshold]
        if not kept:
            # All features scored below the threshold — keep the top-k by MI rather
            # than returning zero features (which crashes training on an empty X).
            k = min(10, len(features))
            top_idx = set(np.argsort(mi)[::-1][:k])
            kept = [f for i, f in enumerate(features) if i in top_idx]
        kept_set = set(kept)
        removed = [
            {"feature": f, "reason": "low_mutual_information"}
            for f in features if f not in kept_set
        ]
        return kept, removed

    def _shap_importance(
        self,
        features: list[str],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, float]:
        try:
            import shap
            from lightgbm import LGBMClassifier, LGBMRegressor

            if self.task_type == TaskType.REGRESSION:
                model = LGBMRegressor(n_estimators=50, verbose=-1, random_state=self.random_state)
            else:
                model = LGBMClassifier(n_estimators=50, verbose=-1, random_state=self.random_state)
            model.fit(X[features], y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[features])
            # Aggregate |SHAP| over samples AND classes. The old code took only
            # class-1 (list) or produced a 2-D array (3-D input) that broke the
            # zip → bare except → silent {} for multiclass.
            if isinstance(shap_values, list):
                abs_arr = np.stack([np.abs(np.asarray(s)) for s in shap_values], axis=-1)
            else:
                abs_arr = np.abs(np.asarray(shap_values))
            if abs_arr.ndim == 3:
                mean_abs = abs_arr.mean(axis=(0, 2))
            else:
                mean_abs = abs_arr.mean(axis=0)
            mean_abs = np.asarray(mean_abs).flatten()
            return {f: float(v) for f, v in zip(features, mean_abs)}
        except Exception:
            return {}

    def _rfe_filter(
        self,
        features: list[str],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[list[str], list[dict[str, str]]]:
        from lightgbm import LGBMClassifier, LGBMRegressor

        removed = []
        try:
            if self.task_type == TaskType.REGRESSION:
                estimator = LGBMRegressor(n_estimators=50, verbose=-1, random_state=self.random_state)
                cv = 3
            else:
                estimator = LGBMClassifier(n_estimators=50, verbose=-1, random_state=self.random_state)
                cv = StratifiedKFold(3, shuffle=True, random_state=self.random_state)

            selector = RFECV(
                estimator,
                step=max(1, len(features) // 10),
                cv=cv,
                # "f1" is binary-only and RAISES on multiclass — which the broad
                # except below then swallowed, silently skipping RFE for every
                # multiclass run. f1_macro works for multiclass.
                scoring=(
                    "neg_root_mean_squared_error"
                    if self.task_type == TaskType.REGRESSION
                    else "f1"
                    if self.task_type == TaskType.BINARY_CLASSIFICATION
                    else "f1_macro"
                ),
                n_jobs=1,
            )
            selector.fit(X[features], y)
            selected = [f for f, s in zip(features, selector.support_) if s]
            for f, s in zip(features, selector.support_):
                if not s:
                    removed.append({"feature": f, "reason": "rfe_eliminated"})
            return selected, removed
        except Exception:
            return features, removed
