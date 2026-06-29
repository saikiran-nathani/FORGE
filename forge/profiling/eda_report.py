"""Auto-generated interactive EDA HTML report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.io as pio

from forge.profiling.models import ColumnType, ProfileReport
from forge.profiling.semantic_profiler import SemanticProfile


class EDAReportGenerator:
    """Builds a self-contained HTML EDA report with Plotly charts."""

    def generate(
        self,
        df: pd.DataFrame,
        profile: ProfileReport,
        semantic: SemanticProfile | None,
        target_column: str,
        output_path: Path,
    ) -> Path:
        sections: list[str] = []
        sections.append(self._overview_section(profile))
        sections.append(self._target_section(df, profile, target_column))
        sections.append(self._distributions_section(df, profile, target_column))
        sections.append(self._correlation_section(df, profile, target_column))
        sections.append(self._missing_section(profile))
        if semantic:
            sections.append(self._llm_section(semantic))

        html = self._wrap_html("\n".join(sections), profile.quality_score)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def _overview_section(self, profile: ProfileReport) -> str:
        type_counts: dict[str, int] = {}
        for t in profile.column_types.values():
            type_counts[t.value] = type_counts.get(t.value, 0) + 1
        fig = px.bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            title="Column Types",
            labels={"x": "Type", "y": "Count"},
        )
        return f"""
        <section><h2>Dataset Overview</h2>
        <p>Shape: {profile.n_rows:,} rows × {profile.n_cols} columns</p>
        <p>Quality Score: <strong>{profile.quality_score}/100</strong></p>
        <p>Memory: {profile.memory_usage_mb} MB</p>
        {pio.to_html(fig, full_html=False, include_plotlyjs=False)}
        </section>"""

    def _target_section(self, df: pd.DataFrame, profile: ProfileReport, target: str) -> str:
        series = df[target].dropna()
        if profile.target_analysis["task_type"] == "regression":
            fig = px.histogram(series, title=f"Target: {target}", nbins=30)
        else:
            counts = series.value_counts()
            fig = px.bar(x=counts.index.astype(str), y=counts.values, title=f"Target: {target}")
        return f"""
        <section><h2>Target Variable</h2>
        <p>Recommended metric: {profile.recommended_metric}</p>
        {pio.to_html(fig, full_html=False, include_plotlyjs=False)}
        </section>"""

    def _distributions_section(self, df: pd.DataFrame, profile: ProfileReport, target: str) -> str:
        charts = []
        for col, col_type in profile.column_types.items():
            if col == target:
                continue
            if col_type == ColumnType.NUMERICAL:
                fig = px.histogram(df, x=col, title=col, nbins=25)
                charts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))
            elif col_type in (ColumnType.CATEGORICAL, ColumnType.BINARY):
                counts = df[col].value_counts().head(15)
                fig = px.bar(x=counts.index.astype(str), y=counts.values, title=col)
                charts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))
        return f"<section><h2>Feature Distributions</h2>{''.join(charts[:8])}</section>"

    def _correlation_section(self, df: pd.DataFrame, profile: ProfileReport, target: str) -> str:
        numerical = [
            c for c, t in profile.column_types.items()
            if t == ColumnType.NUMERICAL and c in df.columns
        ]
        if len(numerical) < 2:
            return "<section><h2>Correlation Analysis</h2><p>Not enough numerical columns.</p></section>"
        corr = df[numerical].corr()
        fig = px.imshow(corr, title="Pearson Correlation Heatmap", color_continuous_scale="RdBu_r")
        pairs = profile.correlations.get("high_correlation_pairs", [])
        pairs_html = "<ul>" + "".join(
            f"<li>{p['col1']} ↔ {p['col2']} (r={p['pearson']:.2f})</li>" for p in pairs[:10]
        ) + "</ul>"
        return f"""
        <section><h2>Correlation Analysis</h2>
        {pio.to_html(fig, full_html=False, include_plotlyjs=False)}
        <h3>Highly Correlated Pairs</h3>{pairs_html}
        </section>"""

    def _missing_section(self, profile: ProfileReport) -> str:
        missing = profile.missing_analysis.get("missing_pct_by_column", {})
        if not missing:
            return "<section><h2>Missing Data</h2><p>No missing values detected.</p></section>"
        fig = px.bar(
            x=list(missing.keys()),
            y=list(missing.values()),
            title="Missing % by Column",
            labels={"x": "Column", "y": "Missing %"},
        )
        return f"""
        <section><h2>Missing Data</h2>
        {pio.to_html(fig, full_html=False, include_plotlyjs=False)}
        </section>"""

    def _llm_section(self, semantic: SemanticProfile) -> str:
        cols_html = ""
        for name, info in semantic.columns.items():
            transforms = info.get("suggested_transforms", [])
            cols_html += f"""
            <div class="insight-card">
              <h4>{name}</h4>
              <p>{info.get('meaning', '')}</p>
              <p>Importance: {info.get('importance', 'MEDIUM')} | Sensitive: {info.get('sensitive', False)}</p>
              <ul>{''.join(f'<li>{t}</li>' for t in transforms[:3])}</ul>
            </div>"""
        leakage = "".join(f"<li>{r}</li>" for r in semantic.leakage_risks)
        interactions = ", ".join(semantic.key_interactions)
        return f"""
        <section><h2>LLM Insights</h2>
        <p>{semantic.data_quality_summary}</p>
        <p><strong>Key interactions:</strong> {interactions or 'None identified'}</p>
        <h3>Leakage Risks</h3><ul>{leakage or '<li>None detected</li>'}</ul>
        <h3>Column Semantics</h3>{cols_html}
        </section>"""

    def _wrap_html(self, body: str, quality_score: float) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FORGE EDA Report</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 2rem; background: #0f172a; color: #e2e8f0; }}
    h1 {{ color: #38bdf8; }}
    h2 {{ color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 0.5rem; }}
    section {{ margin-bottom: 2rem; }}
    .insight-card {{ background: #1e293b; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }}
    .score {{ font-size: 2rem; color: #4ade80; }}
  </style>
</head>
<body>
  <h1>FORGE EDA Report</h1>
  <p class="score">Quality Score: {quality_score}/100</p>
  {body}
</body>
</html>"""
