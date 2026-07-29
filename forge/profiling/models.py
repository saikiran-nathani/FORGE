"""Data structures for profiling results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ColumnType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BINARY = "binary"
    ID = "id"


@dataclass
class ColumnProfile:
    name: str
    detected_type: ColumnType
    statistics: dict[str, Any] = field(default_factory=dict)
    missing_pct: float = 0.0
    n_unique: int = 0


@dataclass
class ProfileReport:
    n_rows: int
    n_cols: int
    column_types: dict[str, ColumnType]
    column_profiles: dict[str, ColumnProfile]
    target_analysis: dict[str, Any]
    correlations: dict[str, Any]
    missing_analysis: dict[str, Any]
    outlier_analysis: dict[str, Any]
    quality_score: float
    recommended_metric: str
    memory_usage_mb: float
    # Per-component sub-scores behind quality_score, so the composite isn't a
    # black box (e.g. {"completeness": 98.2, "uniqueness": 100.0, ...}).
    quality_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "quality_score": self.quality_score,
            "quality_breakdown": self.quality_breakdown,
            "recommended_metric": self.recommended_metric,
            "memory_usage_mb": self.memory_usage_mb,
            "column_types": {k: v.value for k, v in self.column_types.items()},
            "target_analysis": self.target_analysis,
            "missing_analysis": self.missing_analysis,
        }
