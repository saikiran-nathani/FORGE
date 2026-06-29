from __future__ import annotations

from typing import Any

import optuna
from lightgbm import LGBMClassifier, LGBMRegressor

from forge.training.base_model import BaseModel
from forge.training.task_router import TaskType


class LightGBMModel(BaseModel):
    name = "lightgbm"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

    def build_model(self, params: dict[str, Any]) -> Any:
        if self.task_type == TaskType.REGRESSION:
            return LGBMRegressor(**params)
        return LGBMClassifier(**params)
