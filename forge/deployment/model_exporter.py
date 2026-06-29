"""Export model artifacts for production deployment."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExportResult:
    deployment_dir: Path
    files: list[str]


class ModelExporter:
    """Packages model artifacts into a deployable directory."""

    def export(self, artifact_dir: Path, deployment_dir: Path) -> ExportResult:
        deployment_dir.mkdir(parents=True, exist_ok=True)
        files_copied = []

        for name in [
            "best_model.joblib", "profile.json", "test_metrics.json",
            "feature_selection.json", "semantic_profile.json",
        ]:
            src = artifact_dir / name
            if src.exists():
                shutil.copy2(src, deployment_dir / name)
                files_copied.append(name)

        manifest = {
            "artifact_source": str(artifact_dir),
            "files": files_copied,
        }
        (deployment_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        files_copied.append("manifest.json")
        return ExportResult(deployment_dir=deployment_dir, files=files_copied)
