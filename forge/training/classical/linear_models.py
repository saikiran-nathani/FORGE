from __future__ import annotations

from typing import Any

import optuna
from sklearn.linear_model import ElasticNet, Lasso, Ridge, SGDClassifier, SGDRegressor

from forge.training.base_model import BaseModel
from forge.training.task_router import TaskType


class RidgeModel(BaseModel):
    name = "ridge"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }

    def build_model(self, params: dict[str, Any]) -> Any:
        return Ridge(random_state=42, **params)


class LassoModel(BaseModel):
    name = "lasso"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True)}

    def build_model(self, params: dict[str, Any]) -> Any:
        return Lasso(random_state=42, max_iter=2000, **params)


class ElasticNetModel(BaseModel):
    name = "elastic_net"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }

    def build_model(self, params: dict[str, Any]) -> Any:
        return ElasticNet(random_state=42, max_iter=2000, **params)


class SGDModel(BaseModel):
    name = "sgd"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = {
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
            "max_iter": 1000,
            "random_state": 42,
        }
        if self.task_type == TaskType.REGRESSION:
            params["loss"] = trial.suggest_categorical("loss", ["squared_error", "huber"])
        else:
            params["loss"] = trial.suggest_categorical(
                "loss", ["hinge", "log_loss", "modified_huber"]
            )
        return params

    def build_model(self, params: dict[str, Any]) -> Any:
        if self.task_type == TaskType.REGRESSION:
            return SGDRegressor(**params)
        return SGDClassifier(**params)
