from __future__ import annotations

from typing import Any

import optuna
from xgboost import XGBClassifier, XGBRegressor

from forge.training.base_model import BaseModel
from forge.training.task_router import TaskType


class XGBoostModel(BaseModel):
    name = "xgboost"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 0,
        }

    def build_model(self, params: dict[str, Any]) -> Any:
        if self.task_type == TaskType.REGRESSION:
            return XGBRegressor(**params)
        params["eval_metric"] = "logloss" if self.task_type == TaskType.BINARY_CLASSIFICATION else "mlogloss"
        return XGBClassifier(**params)
