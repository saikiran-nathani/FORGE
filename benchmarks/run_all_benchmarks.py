"""Run FORGE on benchmark datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from forge.config import ForgeConfig
from forge.pipeline import ForgePipeline

BENCHMARKS = [
    {
        "name": "titanic",
        "path": "tests/fixtures/titanic_sample.csv",
        "target": "Survived",
        "task": "Predict passenger survival",
        "metric": "roc_auc",
    },
]


def run_benchmarks(output_dir: Path = Path("benchmarks/results"), trials: int = 5) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for bench in BENCHMARKS:
        path = Path(bench["path"])
        if not path.exists():
            results[bench["name"]] = {"status": "skipped", "reason": "dataset not found"}
            continue

        config = ForgeConfig(
            target_column=bench["target"],
            task_description=bench["task"],
            output_dir=output_dir,
            hpo_trials_per_model=trials,
            fast_mode=True,
        )
        pipeline = ForgePipeline(config)
        result = pipeline.run(path)
        key_metric = bench["metric"]
        score = result.best_metrics.get(key_metric) or result.best_metrics.get("f1") or result.best_metrics.get("accuracy")

        results[bench["name"]] = {
            "status": "completed",
            "best_model": result.best_model_name,
            "metrics": result.best_metrics,
            "primary_score": score,
        }

    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_benchmarks()
