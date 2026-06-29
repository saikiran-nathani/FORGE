"""Task type detection for ML pipeline routing."""

from __future__ import annotations

from enum import Enum

import pandas as pd

from forge.profiling.models import ProfileReport


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"


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
        if 3 <= n_unique <= 50:
            return TaskType.MULTICLASS_CLASSIFICATION

        forecast_keywords = ("forecast", "time series", "timeseries", "predict future")
        if any(kw in task_description.lower() for kw in forecast_keywords):
            return TaskType.TIME_SERIES

        return TaskType.MULTICLASS_CLASSIFICATION

    def is_classification(self, task_type: TaskType) -> bool:
        return task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        )
