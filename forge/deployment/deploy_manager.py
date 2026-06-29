"""Orchestrates model deployment."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from forge.deployment.api_generator import APIGenerator
from forge.deployment.batch_pipeline import BatchPipelineGenerator
from forge.deployment.model_card_generator import ModelCardGenerator
from forge.deployment.model_exporter import ModelExporter
from forge.monitoring.drift_monitor import DriftMonitor


@dataclass
class DeploymentResult:
    deployment_id: str
    deployment_dir: Path
    api_url: str
    model_card_path: Path
    files: list[str] = field(default_factory=list)
    monitoring_baseline: dict[str, Any] = field(default_factory=dict)


class DeployManager:
    """One-click deploy: export → API → model card → monitoring baseline."""

    def __init__(self, deployments_root: Path = Path("deployments")):
        self.deployments_root = deployments_root
        self.deployments_root.mkdir(parents=True, exist_ok=True)

    def deploy(
        self,
        experiment_id: str,
        artifact_dir: Path,
        task_description: str,
        target_column: str,
        reference_data: pd.DataFrame | None = None,
    ) -> DeploymentResult:
        deployment_id = f"{experiment_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        deployment_dir = self.deployments_root / deployment_id

        export = ModelExporter().export(artifact_dir, deployment_dir / "artifacts")
        shutil.copytree(artifact_dir, deployment_dir / "artifacts", dirs_exist_ok=True)

        api_files = APIGenerator().generate(deployment_dir)
        dag_path = BatchPipelineGenerator().generate(deployment_dir)
        card_path = deployment_dir / "model_card.md"
        ModelCardGenerator().generate(artifact_dir, task_description, target_column, card_path)

        monitoring_baseline = {}
        if reference_data is not None:
            monitor = DriftMonitor(deployment_dir / "monitoring")
            monitoring_baseline = monitor.set_baseline(reference_data, target_column)

        meta = {
            "deployment_id": deployment_id,
            "experiment_id": experiment_id,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "artifact_dir": str(artifact_dir),
            "api_port": 8080,
            "files": export.files + api_files + [str(dag_path.name), "model_card.md"],
        }
        (deployment_dir / "deployment.json").write_text(json.dumps(meta, indent=2))

        return DeploymentResult(
            deployment_id=deployment_id,
            deployment_dir=deployment_dir,
            api_url=f"http://localhost:8080",
            model_card_path=card_path,
            files=meta["files"],
            monitoring_baseline=monitoring_baseline,
        )

    def get_deployment(self, deployment_id: str) -> dict[str, Any] | None:
        path = self.deployments_root / deployment_id / "deployment.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def list_deployments(self) -> list[dict[str, Any]]:
        results = []
        for d in self.deployments_root.iterdir():
            meta_path = d / "deployment.json"
            if meta_path.exists():
                results.append(json.loads(meta_path.read_text()))
        return results
