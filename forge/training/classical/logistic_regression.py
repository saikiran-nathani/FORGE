from __future__ import annotations

from typing import Any

import optuna
from sklearn.linear_model import LogisticRegression, Ridge

from forge.training.base_model import BaseModel
from forge.training.task_router import TaskType


class LogisticRegressionModel(BaseModel):
    name = "logistic_regression"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        solver = "saga" if penalty in ("l1", "elasticnet") else "lbfgs"
        params: dict[str, Any] = {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "penalty": penalty,
            "solver": solver,
            "max_iter": 1000,
        }
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        if self.task_type != TaskType.REGRESSION:
            params["class_weight"] = trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            )
        return params

    def build_model(self, params: dict[str, Any]) -> Any:
        if self.task_type == TaskType.REGRESSION:
            return Ridge(alpha=1.0 / params["C"], max_iter=params["max_iter"])
        return LogisticRegression(random_state=42, **params)
