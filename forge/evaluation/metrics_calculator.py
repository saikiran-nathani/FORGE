"""Extended evaluation metrics including calibration and fairness support."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


class MetricsCalculator:
    """Computes comprehensive evaluation metrics."""

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
        task_type: str,
        is_binary: bool = True,
    ) -> dict[str, Any]:
        if task_type == "regression":
            return self._regression_metrics(y_true, y_pred)
        return self._classification_metrics(y_true, y_pred, y_proba, is_binary)

    def _classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
        is_binary: bool,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        if is_binary and y_proba is not None:
            proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
                metrics["pr_auc"] = float(average_precision_score(y_true, proba))
                metrics["brier_score"] = float(brier_score_loss(y_true, proba))
                metrics["log_loss"] = float(log_loss(y_true, np.column_stack([1 - proba, proba])))
                metrics["ece"] = float(self._expected_calibration_error(y_true, proba))
            except ValueError:
                metrics["roc_auc"] = None
            metrics["f1"] = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
        return metrics

    def _regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        n = len(y_true)
        p = 1
        adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "adjusted_r2": float(adj_r2),
            "mape": mape,
            "max_error": float(np.max(np.abs(y_true - y_pred))),
        }

    @staticmethod
    def _expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (proba >= bins[i]) & (proba < bins[i + 1])
            if mask.sum() == 0:
                continue
            acc = y_true[mask].mean()
            conf = proba[mask].mean()
            ece += mask.mean() * abs(acc - conf)
        return ece
