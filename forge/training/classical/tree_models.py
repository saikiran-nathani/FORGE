from __future__ import annotations

from typing import Any

import optuna
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from forge.training.base_model import BaseModel
from forge.training.task_router import TaskType


class DecisionTreeModel(BaseModel):
    name = "decision_tree"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": 42,
        }
        if self.task_type == TaskType.REGRESSION:
            params["criterion"] = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"])
        else:
            params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
        return params

    def build_model(self, params: dict[str, Any]) -> Any:
        if self.task_type == TaskType.REGRESSION:
            return DecisionTreeRegressor(**params)
        return DecisionTreeClassifier(**params)


class ExtraTreesModel(BaseModel):
    name = "extra_trees"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "n_jobs": 1,
            "random_state": 42,
        }
        if self.task_type != TaskType.REGRESSION:
            params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
        return params

    def build_model(self, params: dict[str, Any]) -> Any:
        if self.task_type == TaskType.REGRESSION:
            params = {k: v for k, v in params.items() if k != "class_weight"}
            return ExtraTreesRegressor(**params)
        return ExtraTreesClassifier(**params)
