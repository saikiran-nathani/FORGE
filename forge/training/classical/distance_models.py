from __future__ import annotations

from typing import Any

import optuna
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from forge.training.base_model import BaseModel
from forge.training.task_router import TaskType


class KNNModel(BaseModel):
    name = "knn"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
        params: dict[str, Any] = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": metric,
            "n_jobs": 1,
        }
        if metric == "minkowski":
            params["p"] = trial.suggest_int("p", 1, 5)
        return params

    def build_model(self, params: dict[str, Any]) -> Any:
        if self.task_type == TaskType.REGRESSION:
            return KNeighborsRegressor(**params)
        return KNeighborsClassifier(**params)


class SVMModel(BaseModel):
    name = "svm"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
        params: dict[str, Any] = {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "kernel": kernel,
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
        if self.task_type != TaskType.REGRESSION:
            params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
            params["probability"] = True
        return params

    def build_model(self, params: dict[str, Any]) -> Any:
        if self.task_type == TaskType.REGRESSION:
            params = {k: v for k, v in params.items() if k not in ("class_weight", "probability")}
            return SVR(**params)
        return SVC(random_state=42, **params)


class NaiveBayesModel(BaseModel):
    name = "naive_bayes"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-3, log=True)}

    def build_model(self, params: dict[str, Any]) -> Any:
        return GaussianNB(**params)
