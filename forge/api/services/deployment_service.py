"""Deployment and prediction service."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from forge.deployment.deploy_manager import DeployManager, DeploymentResult
from forge.deployment.inference import ModelServer
from forge.monitoring.drift_monitor import DriftMonitor
from forge.monitoring.performance_tracker import PerformanceTracker


@dataclass
class DeploymentState:
    experiment_id: str
    deployment: DeploymentResult
    server: ModelServer
    drift_monitor: DriftMonitor
    performance: PerformanceTracker


class DeploymentService:
    """Manages deployed models for experiments."""

    def __init__(self):
        self.manager = DeployManager()
        self._active: dict[str, DeploymentState] = {}

    def deploy(
        self,
        experiment_id: str,
        artifact_dir: Path,
        task_description: str,
        target_column: str,
    ) -> dict[str, Any]:
        ref_path = artifact_dir / "reference_data.parquet"
        reference = pd.read_parquet(ref_path) if ref_path.exists() else None

        result = self.manager.deploy(
            experiment_id, artifact_dir, task_description, target_column, reference
        )

        server = ModelServer(result.deployment_dir / "artifacts")
        monitor = DriftMonitor(result.deployment_dir / "monitoring")
        perf = PerformanceTracker(result.deployment_dir / "monitoring")
        metrics_path = artifact_dir / "test_metrics.json"
        if metrics_path.exists():
            import json
            perf.set_training_metrics(json.loads(metrics_path.read_text()))

        self._active[experiment_id] = DeploymentState(
            experiment_id=experiment_id,
            deployment=result,
            server=server,
            drift_monitor=monitor,
            performance=perf,
        )

        return {
            "deployment_id": result.deployment_id,
            # The REAL live endpoint (served by this backend, routed via VITE_API_URL),
            # not the fabricated http://localhost:8080 the deploy manager returned.
            "api_url": f"/api/v1/experiments/{experiment_id}/predict",
            "local_bundle_port": 8080,  # port the downloadable serve.py/Docker bundle uses
            "model_card": str(result.model_card_path),
            "files": result.files,
        }

    def predict(self, experiment_id: str, features: dict[str, Any]) -> dict[str, Any]:
        state = self._require(experiment_id)
        start = time.perf_counter()
        try:
            result = state.server.predict(features)
            latency = (time.perf_counter() - start) * 1000
            state.performance.record(
                latency, result.get("prediction"), result.get("confidence")
            )
            return result
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            state.performance.record(latency, None, None, str(exc))
            raise

    def predict_batch(self, experiment_id: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        state = self._require(experiment_id)
        start = time.perf_counter()
        results = state.server.predict(records)
        latency = (time.perf_counter() - start) / max(len(records), 1) * 1000
        for r in results:
            state.performance.record(latency, r.get("prediction"), r.get("confidence"))
        return results

    def check_drift(self, experiment_id: str, df: pd.DataFrame, target_column: str = "") -> dict[str, Any]:
        state = self._require(experiment_id)
        return state.drift_monitor.check_drift(df, target_column)

    def get_monitoring(self, experiment_id: str) -> dict[str, Any]:
        state = self._active.get(experiment_id)
        if not state:
            return {"deployed": False}
        return {
            "deployed": True,
            "deployment_id": state.deployment.deployment_id,
            "performance": state.performance.summary(),
        }

    def get_model_info(self, experiment_id: str) -> dict[str, Any]:
        state = self._require(experiment_id)
        return state.server.model_info()

    def is_deployed(self, experiment_id: str) -> bool:
        return experiment_id in self._active

    def _require(self, experiment_id: str) -> DeploymentState:
        state = self._active.get(experiment_id)
        if not state:
            raise ValueError(f"Experiment {experiment_id} is not deployed")
        return state


deployment_service = DeploymentService()
