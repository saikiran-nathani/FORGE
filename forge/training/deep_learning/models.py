from __future__ import annotations

from typing import Any

import optuna

from forge.training.base_model import BaseModel
from forge.training.deep_learning.nn_wrappers import TabularMLP, TabularTransformer
from forge.training.task_router import TaskType


class MLPModel(BaseModel):
    name = "mlp"
    family = "deep_learning"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "n_layers": trial.suggest_int("n_layers", 2, 4),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "epochs": 40,
            "patience": 6,
            "use_batch_norm": trial.suggest_categorical("use_batch_norm", [True, False]),
            "random_state": 42,
        }

    def build_model(self, params: dict[str, Any]) -> Any:
        task = "regression" if self.task_type == TaskType.REGRESSION else "classification"
        return TabularMLP(task_type=task, **params)


class TabTransformerModel(BaseModel):
    name = "tab_transformer"
    family = "deep_learning"

    def get_search_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "epochs": 40,
            "patience": 6,
            "random_state": 42,
        }

    def build_model(self, params: dict[str, Any]) -> Any:
        task = "regression" if self.task_type == TaskType.REGRESSION else "classification"
        return TabularTransformer(task_type=task, **params)
