"""Ensemble models built from individually tuned base models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from forge.training.hpo.optuna_optimizer import ModelResult, sklearn_scoring
from forge.training.task_router import TaskRouter, TaskType


class EnsembleBuilder:
    """Creates voting and stacking ensembles from top HPO results."""

    def __init__(
        self,
        task_type: TaskType,
        metric_name: str,
        random_state: int = 42,
        cv_folds: int = 5,
    ):
        self.task_type = task_type
        self.metric_name = metric_name
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.router = TaskRouter()

    def build_voting(
        self,
        results: list[ModelResult],
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 3,
    ) -> tuple[Any, str, float]:
        top = sorted(results, key=lambda r: -r.cv_score)[:top_n]
        estimators = [(r.model_name, r.fitted_model) for r in top]

        if self.task_type == TaskType.REGRESSION:
            model = VotingRegressor(estimators=estimators, n_jobs=1)
        else:
            model = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
        model.fit(X, y)
        cv_score = self._cv_score(model, X, y)
        return model, "voting_ensemble", cv_score

    def build_stacking(
        self,
        results: list[ModelResult],
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 5,
    ) -> tuple[Any, str, float]:
        top = sorted(results, key=lambda r: -r.cv_score)[:min(top_n, len(results))]
        estimators = [(r.model_name, r.fitted_model) for r in top]

        if self.task_type == TaskType.REGRESSION:
            model = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=3,
                n_jobs=1,
            )
        else:
            model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=3,
                n_jobs=1,
            )
        model.fit(X, y)
        cv_score = self._cv_score(model, X, y)
        return model, "stacking_ensemble", cv_score

    def _cv_score(self, model: Any, X: pd.DataFrame, y: pd.Series) -> float:
        # Same metric AND fold count as the base models (OptunaOptimizer), so the
        # leaderboard/Pareto/_pick_best compare like with like instead of stamping
        # an f1_macro number with the run's roc_auc label.
        scoring = sklearn_scoring(self.metric_name)
        if self.router.is_classification(self.task_type):
            cv = StratifiedKFold(self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        mean = float(scores.mean())
        return -mean if scoring == "neg_root_mean_squared_error" else mean
