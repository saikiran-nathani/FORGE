"""Data drift monitoring with Evidently or statistical fallback."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DriftMonitor:
    """Tracks feature drift against a training baseline."""

    PSI_THRESHOLD = 0.25

    def __init__(self, monitoring_dir: Path):
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_path = self.monitoring_dir / "baseline.parquet"
        self.reports_dir = self.monitoring_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

    def set_baseline(self, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        features = df.drop(columns=[target_column], errors="ignore")
        features.to_parquet(self.baseline_path)
        meta = {
            "n_rows": len(features),
            "columns": list(features.columns),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.monitoring_dir / "baseline_meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    def check_drift(self, current_df: pd.DataFrame, target_column: str = "") -> dict[str, Any]:
        if not self.baseline_path.exists():
            return {"error": "No baseline set"}

        baseline = pd.read_parquet(self.baseline_path)
        current = current_df.drop(columns=[target_column], errors="ignore")
        common_cols = [c for c in baseline.columns if c in current.columns]

        if self._evidently_available():
            return self._evidently_report(baseline[common_cols], current[common_cols])

        return self._statistical_drift(baseline[common_cols], current[common_cols])

    def _statistical_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> dict[str, Any]:
        feature_scores = {}
        alerts = []

        for col in reference.columns:
            psi = self._compute_psi(reference[col], current[col])
            feature_scores[col] = {"psi": round(psi, 4), "drifted": psi > self.PSI_THRESHOLD}
            if psi > self.PSI_THRESHOLD:
                alerts.append(f"Feature '{col}' PSI={psi:.3f} exceeds threshold")

        overall = float(np.mean([v["psi"] for v in feature_scores.values()])) if feature_scores else 0.0
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_drift_score": round(overall, 4),
            "feature_scores": feature_scores,
            "alerts": alerts,
            "method": "psi",
        }
        report_path = self.reports_dir / f"drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        report_path.write_text(json.dumps(report, indent=2))
        return report

    def _compute_psi(self, expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        expected = expected.dropna()
        actual = actual.dropna()
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        if pd.api.types.is_numeric_dtype(expected):
            breakpoints = np.linspace(expected.min(), expected.max(), buckets + 1)
            breakpoints[0] -= 1e-6
            breakpoints[-1] += 1e-6
            expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
            actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        else:
            categories = expected.value_counts(normalize=True).index.tolist()
            expected_pct = np.array([((expected == c).mean()) for c in categories])
            actual_pct = np.array([((actual == c).mean()) for c in categories])

        expected_pct = np.clip(expected_pct, 0.001, None)
        actual_pct = np.clip(actual_pct, 0.001, None)
        expected_pct /= expected_pct.sum()
        actual_pct /= actual_pct.sum()
        return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

    def _evidently_available(self) -> bool:
        try:
            import evidently  # noqa: F401
            return True
        except ImportError:
            return False

    def _evidently_report(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference, current_data=current)
            result = report.as_dict()
            summary = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": "evidently",
                "drift_detected": result.get("metrics", [{}])[0].get("result", {}).get("dataset_drift", False),
                "raw": result,
            }
            return summary
        except Exception as exc:
            return self._statistical_drift(reference, current) | {"evidently_error": str(exc)}
