"""Experiment state and pipeline execution service."""

from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from forge.config import ForgeConfig
from forge.pipeline import ForgePipeline


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Experiment:
    id: str
    name: str
    task_description: str
    target_column: str
    status: ExperimentStatus
    created_at: str
    dataset_path: Path
    output_dir: Path
    error: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    progress: str = ""
    stage: str = ""
    progress_log: list[str] = field(default_factory=list)


class ExperimentStore:
    """In-memory experiment registry with filesystem artifacts."""

    def __init__(self, base_dir: Path = Path("experiments")):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._experiments: dict[str, Experiment] = {}
        self._lock = threading.Lock()

    def create(
        self,
        name: str,
        target_column: str,
        task_description: str,
        dataset_path: Path,
    ) -> Experiment:
        exp_id = str(uuid.uuid4())[:8]
        output_dir = self.base_dir / exp_id
        output_dir.mkdir(parents=True, exist_ok=True)
        exp = Experiment(
            id=exp_id,
            name=name,
            task_description=task_description,
            target_column=target_column,
            status=ExperimentStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset_path=dataset_path,
            output_dir=output_dir,
        )
        with self._lock:
            self._experiments[exp_id] = exp
        return exp

    def add(self, exp: Experiment) -> None:
        with self._lock:
            self._experiments[exp.id] = exp

    def get(self, exp_id: str) -> Experiment | None:
        return self._experiments.get(exp_id)

    def list_all(self) -> list[Experiment]:
        return list(self._experiments.values())

    def run_async(self, exp: Experiment, trials: int = 10, fast_mode: bool = True) -> None:
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(exp, trials, fast_mode),
            daemon=True,
        )
        thread.start()

    def _run_pipeline(self, exp: Experiment, trials: int, fast_mode: bool) -> None:
        exp.status = ExperimentStatus.RUNNING
        exp.stage = "profiling"
        exp.progress = "Starting pipeline…"

        def on_progress(stage: str, message: str) -> None:
            exp.stage = stage
            exp.progress = message
            exp.progress_log.append(message)
            if len(exp.progress_log) > 16:
                del exp.progress_log[:-16]

        try:
            config = ForgeConfig(
                target_column=exp.target_column,
                task_description=exp.task_description,
                output_dir=exp.output_dir / "outputs",
                hpo_trials_per_model=trials,
                fast_mode=fast_mode,
                enable_feature_selection=True,
                enable_shap=True,
                enable_ensembles=True,
                enable_llm=True,
            )
            pipeline = ForgePipeline(config, progress_cb=on_progress)
            result = pipeline.run(exp.dataset_path)
            exp.status = ExperimentStatus.COMPLETED
            exp.stage = "done"
            exp.result = {
                "best_model_name": result.best_model_name,
                "best_metrics": result.best_metrics,
                "task_type": result.task_type,
                "model_results": result.model_results,
                "quality_score": result.profile.get("quality_score"),
                "generated_features": result.generated_features,
                "semantic_profile": result.semantic_profile,
                "shap_summary": result.shap_summary,
                "error_analysis": result.error_analysis,
                "fairness_report": result.fairness_report,
                "pareto_frontier": result.pareto_frontier,
                "llm_report": result.llm_report_path,
                "artifact_dir": str(result.output_dir),
                "eda_report": str(result.output_dir / "eda_report.html"),
            }
            exp.progress = "Complete"
        except Exception as exc:
            exp.status = ExperimentStatus.FAILED
            exp.stage = "failed"
            exp.error = str(exc)
            exp.progress = "Failed"


store = ExperimentStore()
