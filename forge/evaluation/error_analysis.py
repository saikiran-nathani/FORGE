"""Error analysis: worst predictions, slice performance, confusion deep dive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ErrorAnalyzer:
    """Analyzes model errors and underperforming slices."""

    def analyze(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
        task_type: str,
        feature_names: list[str],
        output_dir: Path,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {}

        if task_type == "regression":
            result["residual_analysis"] = self._residual_analysis(y_test.values, y_pred)
        else:
            result["worst_predictions"] = self._worst_predictions(
                X_test, y_test.values, y_pred, y_proba, feature_names
            )
            result["confusion_analysis"] = self._confusion_analysis(y_test.values, y_pred)
            result["slice_analysis"] = self._slice_analysis(X_test, y_test.values, y_pred)

        with open(output_dir / "error_analysis.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        return result

    def _worst_predictions(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
        feature_names: list[str],
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
        # Confidence (max class prob) for display; correct per-sample cross-entropy
        # for RANKING. The old code used a binary p = proba if y==1 else 1-proba,
        # which is meaningless for multiclass. -log P(true class) is correct for both.
        confidence = None
        if y_proba is not None:
            confidence = y_proba.max(axis=1) if y_proba.ndim > 1 else y_proba

        losses = []
        for i in range(len(y_true)):
            if y_proba is not None and y_proba.ndim > 1:
                c = int(y_true[i])
                p = y_proba[i, c] if 0 <= c < y_proba.shape[1] else 1e-8
                loss = -np.log(max(float(p), 1e-8))
            elif y_proba is not None:
                p = y_proba[i] if y_true[i] == 1 else 1 - y_proba[i]
                loss = -np.log(max(float(p), 1e-8))
            else:
                loss = float(y_true[i] != y_pred[i])
            losses.append(loss)

        worst_idx = np.argsort(losses)[-top_n:][::-1]
        records = []
        for idx in worst_idx:
            row = {feature_names[j]: float(X.iloc[idx, j]) for j in range(min(len(feature_names), X.shape[1]))}
            record: dict[str, Any] = {
                "index": int(idx),
                "true_label": int(y_true[idx]) if np.issubdtype(type(y_true[idx]), np.integer) else float(y_true[idx]),
                "predicted_label": int(y_pred[idx]) if np.issubdtype(type(y_pred[idx]), np.integer) else float(y_pred[idx]),
                "loss": float(losses[idx]),
                # These are the row's first columns by position, not ranked by
                # importance — name them honestly (was mislabeled "top_features").
                "feature_values": dict(list(row.items())[:8]),
            }
            if confidence is not None:
                record["probability"] = float(confidence[idx])
            records.append(record)
        return records

    def _confusion_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    pairs.append({
                        "true_class": int(i),
                        "predicted_class": int(j),
                        "count": int(cm[i, j]),
                    })
        pairs.sort(key=lambda x: -x["count"])
        return {"confused_pairs": pairs[:10]}

    def _slice_analysis(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.1,
    ) -> dict[str, Any]:
        overall_acc = float((y_true == y_pred).mean())
        slices: list[dict[str, Any]] = []

        for col in X.columns[:10]:
            series = X[col]
            if series.nunique() <= 10:
                for val in series.unique():
                    mask = series == val
                    if mask.sum() < 5:
                        continue
                    acc = float((y_true[mask] == y_pred[mask]).mean())
                    slices.append({
                        "feature": col,
                        "value": str(val),
                        "accuracy": acc,
                        "count": int(mask.sum()),
                        "underperforming": acc < overall_acc - threshold,
                    })
            else:
                try:
                    quartiles = pd.qcut(series, 4, duplicates="drop")
                    for q in quartiles.unique():
                        mask = quartiles == q
                        acc = float((y_true[mask] == y_pred[mask]).mean())
                        slices.append({
                            "feature": col,
                            "value": str(q),
                            "accuracy": acc,
                            "count": int(mask.sum()),
                            "underperforming": acc < overall_acc - threshold,
                        })
                except ValueError:
                    continue

        underperforming = [s for s in slices if s["underperforming"]]
        return {
            "overall_accuracy": overall_acc,
            "slices": slices[:30],
            "underperforming_slices": underperforming[:10],
        }

    def _residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        residuals = y_true - y_pred
        return {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "max_abs_residual": float(np.max(np.abs(residuals))),
            "residual_skewness": float(pd.Series(residuals).skew()),
        }
