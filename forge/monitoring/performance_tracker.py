"""Production performance and system monitoring."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PredictionRecord:
    timestamp: str
    latency_ms: float
    prediction: Any
    confidence: float | None = None
    error: str = ""


class PerformanceTracker:
    """Tracks inference latency, volume, and prediction distribution."""

    def __init__(self, monitoring_dir: Path, max_records: int = 10000):
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self._records: deque[PredictionRecord] = deque(maxlen=max_records)
        self._training_metrics: dict[str, float] = {}

    def set_training_metrics(self, metrics: dict[str, Any]) -> None:
        self._training_metrics = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and k != "confusion_matrix"
        }
        (self.monitoring_dir / "training_metrics.json").write_text(
            json.dumps(self._training_metrics, indent=2)
        )

    def record(
        self,
        latency_ms: float,
        prediction: Any,
        confidence: float | None = None,
        error: str = "",
    ) -> None:
        self._records.append(PredictionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency_ms,
            prediction=prediction,
            confidence=confidence,
            error=error,
        ))

    def summary(self) -> dict[str, Any]:
        if not self._records:
            return {"total_requests": 0}

        latencies = [r.latency_ms for r in self._records if not r.error]
        errors = [r for r in self._records if r.error]
        confidences = [r.confidence for r in self._records if r.confidence is not None]

        preds = [r.prediction for r in self._records if not r.error]
        pred_distribution: dict[str, int] = {}
        for p in preds:
            key = str(p)
            pred_distribution[key] = pred_distribution.get(key, 0) + 1

        result = {
            "total_requests": len(self._records),
            "error_rate": len(errors) / len(self._records),
            "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0,
            "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0,
            "latency_p99_ms": float(np.percentile(latencies, 99)) if latencies else 0,
            "prediction_distribution": pred_distribution,
            "avg_confidence": float(np.mean(confidences)) if confidences else None,
            "training_metrics": self._training_metrics,
        }
        (self.monitoring_dir / "performance_summary.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        return result

    class Timer:
        def __init__(self, tracker: PerformanceTracker):
            self.tracker = tracker
            self.start = 0.0
            self.prediction = None
            self.confidence = None
            self.error = ""

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            latency = (time.perf_counter() - self.start) * 1000
            self.tracker.record(latency, self.prediction, self.confidence, self.error)
