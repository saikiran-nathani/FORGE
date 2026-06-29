"""LLM-generated natural language analysis report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forge.llm.client import LLMClient


class LLMReportGenerator:
    """Generates markdown analysis reports from pipeline results."""

    SYSTEM = "You are a senior data scientist writing a concise ML analysis report in markdown."

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def generate(
        self,
        task_description: str,
        profile: dict[str, Any],
        semantic: dict[str, Any],
        best_model: str,
        metrics: dict[str, Any],
        shap_summary: dict[str, Any],
        error_analysis: dict[str, Any],
        fairness: dict[str, Any],
        n_original_features: int,
        n_engineered_features: int,
        output_path: Path,
    ) -> str:
        prompt = self._build_prompt(
            task_description, profile, semantic, best_model, metrics,
            shap_summary, error_analysis, fairness, n_original_features, n_engineered_features,
        )

        if self.llm.available:
            try:
                report = self.llm.complete(self.SYSTEM, prompt)
            except Exception:
                report = self._heuristic_report(
                    task_description, best_model, metrics, shap_summary, error_analysis, fairness
                )
        else:
            report = self._heuristic_report(
                task_description, best_model, metrics, shap_summary, error_analysis, fairness
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        return report

    def _build_prompt(self, *args) -> str:
        task_description, profile, semantic, best_model, metrics, shap, errors, fairness, n_orig, n_eng = args
        top_features = shap.get("top_features", [])[:10]
        return f"""Write a comprehensive ML analysis report (500-800 words) in markdown covering:

Task: {task_description}
Quality score: {profile.get('quality_score')}/100
Features: {n_orig} original → {n_eng} after engineering
Best model: {best_model}
Metrics: {json.dumps({k: v for k, v in metrics.items() if k != 'confusion_matrix'}, indent=2)}
Top SHAP features: {json.dumps(top_features)}
Error analysis summary: {json.dumps(errors.get('underperforming_slices', errors.get('worst_predictions', []))[:5], default=str)}
Fairness flags: {fairness.get('flags', [])}
Semantic insights: {semantic.get('data_quality_summary', '')}

Sections: Executive Summary, Feature Engineering Insights, Model Selection Rationale,
Key Feature Insights, Error Patterns, Fairness Assessment (if applicable),
Recommendations (3-5 actions), Deployment Considerations."""

    def _heuristic_report(
        self,
        task: str,
        model: str,
        metrics: dict[str, Any],
        shap: dict[str, Any],
        errors: dict[str, Any],
        fairness: dict[str, Any],
    ) -> str:
        top = shap.get("top_features", [])[:5]
        top_lines = "\n".join(f"- **{f['feature']}** (SHAP: {f['mean_abs_shap']:.4f})" for f in top) or "- No SHAP data"
        metric_lines = "\n".join(
            f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}"
            for k, v in metrics.items() if k != "confusion_matrix"
        )
        under = errors.get("slice_analysis", {}).get("underperforming_slices", [])
        error_lines = "\n".join(
            f"- {s['feature']}={s['value']}: accuracy {s['accuracy']:.2f}" for s in under[:5]
        ) or "- No significant underperforming slices detected"

        fairness_section = ""
        if fairness.get("flags"):
            fairness_section = "## Fairness Assessment\n" + "\n".join(f"- ⚠️ {f}" for f in fairness["flags"])

        return f"""# FORGE Analysis Report

## Executive Summary
The pipeline trained **{model}** for task: _{task or 'unspecified'}_.
Primary metrics indicate {'strong' if metrics.get('accuracy', metrics.get('r2', 0)) and (metrics.get('accuracy', 0) > 0.7 or metrics.get('r2', 0) > 0.5) else 'moderate'} performance.

## Model Metrics
{metric_lines}

## Key Feature Insights
{top_lines}

## Error Patterns
{error_lines}

{fairness_section}

## Recommendations
1. Review underperforming slices and consider targeted feature engineering
2. Monitor top SHAP features for drift in production
3. Validate fairness metrics if sensitive attributes are present
4. Retrain when data quality score drops below 80
5. Consider ensemble methods if single-model variance is high

## Deployment Considerations
- Serialize preprocessing pipeline alongside the model
- Set up drift monitoring on top 5 SHAP features
- Track latency for real-time inference requirements
"""
