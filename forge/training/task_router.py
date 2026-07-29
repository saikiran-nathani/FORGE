"""Task type detection for ML pipeline routing."""

from __future__ import annotations

from enum import Enum

import pandas as pd

from forge.profiling.models import ProfileReport


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class TaskRouter:
    """Detects ML task type from target column and profile."""

    def detect(
        self,
        profile: ProfileReport,
        task_description: str = "",
        override: TaskType | None = None,
    ) -> TaskType:
        if override:
            return override

        target_analysis = profile.target_analysis
        if target_analysis["task_type"] == "regression":
            return TaskType.REGRESSION

        n_unique = target_analysis["n_unique"]
        if n_unique == 2:
            return TaskType.BINARY_CLASSIFICATION
        # Everything else non-numeric is multiclass. (Time-series was removed:
        # there is no forecasting model, and a numeric "forecast" target already
        # routes to REGRESSION above — honest rather than a dead capability.)
        return TaskType.MULTICLASS_CLASSIFICATION

    def is_classification(self, task_type: TaskType) -> bool:
        return task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        )
