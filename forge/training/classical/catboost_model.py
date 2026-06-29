from __future__ import annotations

from typing import Any

import optuna
from catboost import CatBoostClassifier, CatBoostRegressor

from forge.training.base_model import BaseModel
from forge.training.task_router import TaskType


class CatBoostModel(BaseModel):
    name = "catboost"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_state": 42,
            "verbose": 0,
        }
        if self.task_type != TaskType.REGRESSION:
            params["auto_class_weights"] = trial.suggest_categorical(
                "auto_class_weights", [None, "Balanced"]
            )
        return params

    def build_model(self, params: dict[str, Any]) -> Any:
        if self.task_type == TaskType.REGRESSION:
            params = {k: v for k, v in params.items() if k != "auto_class_weights"}
            return CatBoostRegressor(**params)
        return CatBoostClassifier(**params)
