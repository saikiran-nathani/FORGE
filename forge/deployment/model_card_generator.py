"""Auto-generated model card documentation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelCardGenerator:
    """Generates a model card markdown document."""

    def generate(
        self,
        artifact_dir: Path,
        task_description: str,
        target_column: str,
        output_path: Path | None = None,
    ) -> str:
        artifact_dir = Path(artifact_dir)
        metrics = self._load_json(artifact_dir / "test_metrics.json")
        profile = self._load_json(artifact_dir / "profile.json")
        semantic = self._load_json(artifact_dir / "semantic_profile.json")
        fairness = self._load_json(artifact_dir / "fairness" / "fairness_report.json")
        errors = self._load_json(artifact_dir / "errors" / "error_analysis.json")
        honest = self._load_json(artifact_dir / "honest_context.json")

        import joblib
        bundle = joblib.load(artifact_dir / "best_model.joblib")
        model_name = bundle.get("model_name", "unknown")

        # Format row count defensively: "{x:,}" on the "N/A" string default raises
        # ValueError, turning /deploy into a cryptic 500.
        n_rows = profile.get("n_rows")
        rows_str = f"{n_rows:,}" if isinstance(n_rows, int) else "N/A"

        limitations = []
        if errors.get("underperforming_slices") or errors.get("slice_analysis", {}).get("underperforming_slices"):
            limitations.append("Model underperforms on certain data slices — see error analysis")
        if fairness.get("flags"):
            limitations.extend(fairness["flags"])
        # Fold in the pipeline's honest-numbers warnings (small data, imbalance,
        # below-baseline, CV->test gap) so the card can't claim "None identified"
        # while the run itself flagged concerns.
        limitations.extend(honest.get("warnings", []))

        card = f"""# Model Card: {model_name}

**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## Model Details
- **Model:** {model_name}
- **Target:** `{target_column}`
- **Task:** {task_description or "Not specified"}
- **Type:** {profile.get("target_analysis", {}).get("task_type", "unknown")}

## Training Data
- **Rows:** {rows_str}
- **Columns:** {profile.get("n_cols", "N/A")}
- **Quality Score:** {profile.get("quality_score", "N/A")}/100

## Performance Metrics
{self._format_metrics(metrics)}

## Performance vs Baseline
{self._format_baseline(metrics, honest.get("baseline_metrics", {}))}

## Features
- **Input columns:** {len(bundle.get("metadata", {}).get("input_columns", []))}
- **Engineered features:** {len(bundle.get("feature_names", []))}

## Fairness Assessment
{self._format_fairness(fairness)}

## Known Limitations
{chr(10).join(f"- {l}" for l in limitations) if limitations else "- None identified"}

## Intended Use
- Batch and real-time prediction for the task described above
- Decision support — not autonomous decision-making without human review

## Out of Scope
- Data distributions significantly different from training data
- Features not present during training
- Adversarial or manipulated inputs

## Ethical Considerations
{semantic.get("data_quality_summary", "Review sensitive attributes flagged in semantic profile.")}
"""
        output_path = output_path or artifact_dir / "model_card.md"
        output_path.write_text(card, encoding="utf-8")
        return card

    def _load_json(self, path: Path) -> dict[str, Any]:
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _format_metrics(self, metrics: dict[str, Any]) -> str:
        if not metrics:
            return "No metrics available."
        lines = []
        for k, v in metrics.items():
            if k == "confusion_matrix":
                continue
            if isinstance(v, float):
                lines.append(f"- **{k}:** {v:.4f}")
            else:
                lines.append(f"- **{k}:** {v}")
        return "\n".join(lines)

    def _format_baseline(self, metrics: dict[str, Any], baseline: dict[str, Any]) -> str:
        if not baseline:
            return "Baseline metrics not available."
        keys = [k for k in ("roc_auc", "f1", "balanced_accuracy", "accuracy", "rmse", "mae")
                if k in metrics and k in baseline]
        if not keys:
            return "Baseline metrics not available."
        lines = ["| Metric | Model | Baseline (majority-class / mean) |", "|---|---|---|"]
        for k in keys:
            m, b = metrics.get(k), baseline.get(k)
            if isinstance(m, (int, float)) and isinstance(b, (int, float)):
                lines.append(f"| {k} | {m:.4f} | {b:.4f} |")
        return "\n".join(lines)

    def _format_fairness(self, fairness: dict[str, Any]) -> str:
        if not fairness.get("sensitive_columns"):
            return "No sensitive attributes detected."
        flags = fairness.get("flags", [])
        if flags:
            return "⚠️ Issues flagged:\n" + "\n".join(f"- {f}" for f in flags)
        return "No fairness issues flagged."
