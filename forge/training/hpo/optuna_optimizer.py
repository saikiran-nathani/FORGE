"""Optuna hyperparameter optimization."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from forge.training.base_model import BaseModel
from forge.training.task_router import TaskRouter, TaskType

_SCORING_MAP = {
    "accuracy": "accuracy",
    "f1": "f1",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
    "rmse": "neg_root_mean_squared_error",
}


def sklearn_scoring(metric_name: str) -> str:
    """Map a FORGE metric name to an sklearn scoring string.

    Shared by OptunaOptimizer and EnsembleBuilder so base-model and ensemble
    CV scores are always the SAME metric (otherwise the leaderboard compares,
    e.g., an ensemble's f1_macro against base models' roc_auc).
    """
    return _SCORING_MAP.get(metric_name, "accuracy")


@dataclass
class ModelResult:
    model_name: str
    best_params: dict[str, Any]
    cv_score: float
    cv_score_std: float
    training_time: float
    inference_latency_ms: float
    fitted_model: Any
    metric_name: str


class OptunaOptimizer:
    """Bayesian HPO with cross-validation for each model."""

    def __init__(
        self,
        task_type: TaskType,
        metric_name: str,
        n_trials: int = 30,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.task_type = task_type
        self.metric_name = metric_name
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.router = TaskRouter()

    def optimize_model(
        self,
        model_cls: type[BaseModel],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> ModelResult:
        model_instance = model_cls(self.task_type)
        scoring = self._sklearn_scoring()

        def objective(trial: optuna.Trial) -> float:
            params = model_instance.get_search_space(trial)
            model = model_instance.build_model(params)
            cv = self._get_cv(y)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            return float(scores.mean())

        study = optuna.create_study(
            # Every sklearn scorer used here follows "higher is better" — including
            # neg_root_mean_squared_error, where less-negative means lower RMSE.
            # So we always maximize. (Previously this minimized neg-RMSE, which
            # selected the WORST hyperparameters on every regression run.)
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = model_instance.get_search_space(
            _FixedTrial(study.best_params)
        )
        start = time.perf_counter()
        fitted = model_instance.train(X, y, best_params)
        training_time = time.perf_counter() - start

        sample = X.head(min(100, len(X)))
        lat_start = time.perf_counter()
        for _ in range(10):
            fitted.predict(sample)
        inference_latency_ms = (time.perf_counter() - lat_start) / 10 * 1000

        cv = self._get_cv(y)
        scores = cross_val_score(fitted, X, y, cv=cv, scoring=scoring, n_jobs=1)
        cv_mean = float(scores.mean())
        if scoring == "neg_root_mean_squared_error":
            cv_mean = -cv_mean

        return ModelResult(
            model_name=model_instance.name,
            best_params=study.best_params,
            cv_score=cv_mean,
            cv_score_std=float(scores.std()),
            training_time=training_time,
            inference_latency_ms=inference_latency_ms,
            fitted_model=fitted,
            metric_name=self.metric_name,
        )

    def _get_cv(self, y: pd.Series):
        if self.router.is_classification(self.task_type):
            return StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

    def _sklearn_scoring(self) -> str:
        return sklearn_scoring(self.metric_name)


class _FixedTrial:
    """Wrap fixed params so get_search_space can rebuild the model."""

    def __init__(self, params: dict[str, Any]):
        self._params = params

    def suggest_float(self, name: str, *args, **kwargs) -> float:
        return self._params[name]

    def suggest_int(self, name: str, *args, **kwargs) -> int:
        return self._params[name]

    def suggest_categorical(self, name: str, choices: list) -> Any:
        return self._params.get(name, choices[0])
