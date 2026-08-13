#!/usr/bin/env python
"""Build the committed demo seed the FORGE API loads at startup.

Runs the full pipeline on a dataset, then copies the resulting artifacts and a
serialized result dict into ``forge/api/demo/`` so the deployed app shows a real,
pre-trained experiment (leaderboard, SHAP, reports) instantly — with a working
prediction Playground — and never has to train anything on first load.

Usage:
    python scripts/build_demo_seed.py <dataset.csv> --target <col> \
        --task "Predict ..." --name "Churn demo"

Swap in your own dataset any time and re-run; commit the regenerated
forge/api/demo/ directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from forge.config import ForgeConfig
from forge.pipeline import ForgePipeline

DEMO_DIR = Path(__file__).resolve().parent.parent / "forge" / "api" / "demo"
DEMO_ID = "demo"


def build_result_dict(result) -> dict:
    """Mirror the result mapping in ExperimentStore._run_pipeline."""
    return {
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
        "baseline_metrics": result.baseline_metrics,
        "eval_context": result.eval_context,
        "warnings": result.warnings,
        "significance": result.significance,
        "task_plan": result.task_plan,
        # path fields below are recomputed at load time from the committed location
        "llm_report": result.llm_report_path,
        "artifact_dir": str(result.output_dir),
        "eda_report": str(result.output_dir / "eda_report.html"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FORGE demo seed.")
    parser.add_argument("dataset", type=Path, help="CSV/Parquet/JSON dataset path")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--task", default="", help="Natural-language task description")
    parser.add_argument("--name", default="Demo experiment", help="Display name")
    parser.add_argument("--trials", type=int, default=12, help="HPO trials per model")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        config = ForgeConfig(
            target_column=args.target,
            task_description=args.task,
            output_dir=Path(tmp),
            hpo_trials_per_model=args.trials,
            fast_mode=True,        # matches the deployed API; fast + lean
            enable_llm=False,      # no API key needed to build the seed
            enable_shap=True,
            enable_ensembles=True,
            enable_deep_learning=False,
        )
        print(f"Training demo seed on {args.dataset} (target={args.target})...")
        result = ForgePipeline(config).run(args.dataset)

        artifacts_src = result.output_dir
        artifacts_dst = DEMO_DIR / "artifacts"
        if artifacts_dst.exists():
            shutil.rmtree(artifacts_dst)
        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(artifacts_src, artifacts_dst)

        (DEMO_DIR / "result.json").write_text(json.dumps(build_result_dict(result), indent=2))
        (DEMO_DIR / "meta.json").write_text(json.dumps({
            "experiment_id": DEMO_ID,
            "name": args.name,
            "task_description": args.task,
            "target_column": args.target,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

    print(f"\nDemo seed written to {DEMO_DIR}")
    print(f"  best model: {result.best_model_name}")
    print(f"  metrics:    {json.dumps({k: round(v, 4) for k, v in result.best_metrics.items() if isinstance(v, (int, float))})}")
    print("\nCommit forge/api/demo/ and the API will serve this experiment on startup.")


if __name__ == "__main__":
    main()
