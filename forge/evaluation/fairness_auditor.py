"""Fairness auditing across sensitive attributes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forge.profiling.semantic_profiler import SemanticProfile


class FairnessAuditor:
    """Computes fairness metrics for sensitive columns."""

    DISPARATE_IMPACT_RANGE = (0.8, 1.25)

    def audit(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
        semantic: SemanticProfile,
        output_dir: Path,
        task_type: str = "binary_classification",
    ) -> dict[str, Any]:
        sensitive_cols = [
            col for col, info in semantic.columns.items()
            if info.get("sensitive") and col in df.columns
        ]
        if not sensitive_cols:
            return {"sensitive_columns": [], "metrics": {}, "flags": []}

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, Any] = {}
        flags: list[str] = []

        for col in sensitive_cols:
            col_metrics = self._compute_group_metrics(df[col], y_true, y_pred, y_proba, task_type)
            metrics[col] = col_metrics
            flags.extend(self._check_fairness(col, col_metrics, task_type))

        result = {"sensitive_columns": sensitive_cols, "metrics": metrics, "flags": flags}
        with open(output_dir / "fairness_report.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        return result

    def _compute_group_metrics(
        self,
        groups: pd.Series,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
        task_type: str,
    ) -> dict[str, Any]:
        # Binary 0/1 accuracy and positive-rate/disparate-impact only make sense
        # for binary classification. Regression uses per-group error (MAE);
        # multiclass uses per-group accuracy only (no "positive" class to rate).
        is_regression = task_type == "regression"
        is_binary = task_type == "binary_classification"
        group_stats: dict[str, Any] = {}
        unique_groups = groups.dropna().unique()

        for g in unique_groups:
            mask = (groups == g).values
            if mask.sum() < 5:
                continue
            stat: dict[str, Any] = {"count": int(mask.sum())}
            if is_regression:
                stat["mae"] = float(np.abs(y_true[mask] - y_pred[mask]).mean())
                stat["mean_prediction"] = float(np.mean(y_pred[mask]))
            else:
                stat["accuracy"] = float((y_true[mask] == y_pred[mask]).mean())
                if is_binary:
                    # positive class after label encoding is 1
                    stat["positive_rate"] = float((y_pred[mask] == 1).mean())
                    if y_proba is not None:
                        proba = y_proba[mask][:, 1] if y_proba.ndim > 1 else y_proba[mask]
                        stat["mean_predicted_proba"] = float(np.mean(proba))
            group_stats[str(g)] = stat

        # Disparate impact requires a binary positive rate.
        if is_binary:
            rates = [v["positive_rate"] for v in group_stats.values() if "positive_rate" in v]
            if len(rates) >= 2:
                group_stats["disparate_impact_ratio"] = (
                    float(min(rates) / max(rates)) if max(rates) > 0 else 1.0
                )
        return group_stats

    def _check_fairness(self, col: str, metrics: dict[str, Any], task_type: str) -> list[str]:
        flags = []
        di = metrics.get("disparate_impact_ratio")
        if di is not None and not (self.DISPARATE_IMPACT_RANGE[0] <= di <= self.DISPARATE_IMPACT_RANGE[1]):
            flags.append(f"Disparate impact out of range for '{col}': {di:.3f}")

        group_vals = [v for v in metrics.values() if isinstance(v, dict)]
        if task_type == "regression":
            maes = [v["mae"] for v in group_vals if "mae" in v]
            if len(maes) >= 2 and min(maes) > 0 and max(maes) / min(maes) > 1.5:
                flags.append(f"Error (MAE) gap > 1.5x across groups in '{col}'")
        else:
            accs = [v["accuracy"] for v in group_vals if "accuracy" in v]
            if accs and max(accs) - min(accs) > 0.1:
                flags.append(f"Accuracy gap > 10% across groups in '{col}'")
        return flags
