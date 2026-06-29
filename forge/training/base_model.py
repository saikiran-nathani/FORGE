"""Base interface for all FORGE model implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import optuna
import pandas as pd

from forge.training.task_router import TaskType


class BaseModel(ABC):
    name: str = "base"
    family: str = "classical"

    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        self.model: Any = None

    @abstractmethod
    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        ...

    @abstractmethod
    def build_model(self, params: dict[str, Any]) -> Any:
        ...

    def train(self, X: pd.DataFrame, y: pd.Series, params: dict[str, Any]) -> Any:
        self.model = self.build_model(params)
        self.model.fit(X, y)
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def get_feature_importance(self) -> dict[str, float] | None:
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return {f"feature_{i}": float(v) for i, v in enumerate(importances)}
        if hasattr(self.model, "coef_"):
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            return {f"feature_{i}": float(v) for i, v in enumerate(coef)}
        return None
