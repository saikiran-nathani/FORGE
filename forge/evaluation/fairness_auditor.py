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
            col_metrics = self._compute_group_metrics(df[col], y_true, y_pred, y_proba)
            metrics[col] = col_metrics
            flags.extend(self._check_fairness(col, col_metrics))

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
    ) -> dict[str, Any]:
        group_stats: dict[str, Any] = {}
        unique_groups = groups.dropna().unique()

        for g in unique_groups:
            mask = (groups == g).values
            if mask.sum() < 5:
                continue
            acc = float((y_true[mask] == y_pred[mask]).mean())
            pos_rate = float(y_pred[mask].mean())
            stat: dict[str, Any] = {"count": int(mask.sum()), "accuracy": acc, "positive_rate": pos_rate}
            if y_proba is not None:
                proba = y_proba[mask][:, 1] if y_proba.ndim > 1 else y_proba[mask]
                stat["mean_predicted_proba"] = float(np.mean(proba))
            group_stats[str(g)] = stat

        if len(group_stats) >= 2:
            rates = [v["positive_rate"] for v in group_stats.values()]
            group_stats["disparate_impact_ratio"] = float(min(rates) / max(rates)) if max(rates) > 0 else 1.0
        return group_stats

    def _check_fairness(self, col: str, metrics: dict[str, Any]) -> list[str]:
        flags = []
        di = metrics.get("disparate_impact_ratio")
        if di is not None and not (self.DISPARATE_IMPACT_RANGE[0] <= di <= self.DISPARATE_IMPACT_RANGE[1]):
            flags.append(f"Disparate impact out of range for '{col}': {di:.3f}")

        accs = [v["accuracy"] for k, v in metrics.items() if isinstance(v, dict) and "accuracy" in v]
        if accs and max(accs) - min(accs) > 0.1:
            flags.append(f"Accuracy gap > 10% across groups in '{col}'")
        return flags
