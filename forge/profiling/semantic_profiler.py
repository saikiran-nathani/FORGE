"""LLM-powered semantic column analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from forge.llm.client import LLMClient
from forge.profiling.models import ColumnType, ProfileReport

SENSITIVE_KEYWORDS = {"age", "gender", "sex", "race", "ethnicity", "religion", "disability"}
# "age" is also a DURATION in many names (account_age, session_age, record_age) —
# not a person's age. When "age" co-occurs with one of these it isn't protected.
AGE_DURATION_CONTEXT = {
    "account", "page", "session", "subscription", "record", "file",
    "data", "system", "cache", "vintage", "device", "product", "tenure",
}
IMPORTANCE_HINTS = {
    "high": {"charge", "price", "amount", "tenure", "salary", "income", "score", "rate"},
    "low": {"id", "name", "note", "comment", "description"},
}


@dataclass
class SemanticProfile:
    columns: dict[str, dict[str, Any]]
    data_quality_summary: str
    key_interactions: list[str]
    leakage_risks: list[str]
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "data_quality_summary": self.data_quality_summary,
            "key_interactions": self.key_interactions,
            "leakage_risks": self.leakage_risks,
            "source": self.source,
        }


class SemanticProfiler:
    """Infers semantic meaning of columns using LLM or heuristics."""

    SYSTEM_PROMPT = (
        "You are a data scientist analyzing a dataset. Return valid JSON only."
    )

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def profile(
        self,
        report: ProfileReport,
        task_description: str,
        target_column: str,
    ) -> SemanticProfile:
        if self.llm.available:
            try:
                return self._llm_profile(report, task_description, target_column)
            except Exception:
                pass
        return self._heuristic_profile(report, task_description, target_column)

    def _llm_profile(
        self,
        report: ProfileReport,
        task_description: str,
        target_column: str,
    ) -> SemanticProfile:
        column_summaries = []
        for name, col_profile in report.column_profiles.items():
            column_summaries.append({
                "name": name,
                "type": col_profile.detected_type.value,
                "statistics": col_profile.statistics,
                "missing_pct": col_profile.missing_pct,
            })

        user_prompt = f"""Task description: {task_description}
Target column: {target_column}
Column profiles: {column_summaries}

For each column provide:
1. meaning (semantic description)
2. importance (HIGH/MEDIUM/LOW/IRRELEVANT)
3. sensitive (boolean)
4. suggested_transforms (list of strings)
5. quality_issues (string)

Also provide: data_quality_summary, key_interactions (list), leakage_risks (list).

Return JSON: {{"columns": {{"col_name": {{...}}}}, "data_quality_summary": "...", "key_interactions": [], "leakage_risks": []}}"""

        data = self.llm.complete_json(self.SYSTEM_PROMPT, user_prompt)
        return SemanticProfile(
            columns=data.get("columns", {}),
            data_quality_summary=data.get("data_quality_summary", ""),
            key_interactions=data.get("key_interactions", []),
            leakage_risks=data.get("leakage_risks", []),
            source="llm",
        )

    def _heuristic_profile(
        self,
        report: ProfileReport,
        task_description: str,
        target_column: str,
    ) -> SemanticProfile:
        columns: dict[str, dict[str, Any]] = {}
        high_cols: list[str] = []

        for name, col_profile in report.column_profiles.items():
            lower = name.lower()
            # Token match (split on separators + camelCase), not substring:
            # otherwise "age" flags usage/mileage/average/page_views as sensitive
            # and the fairness auditor fabricates concerns about non-protected cols.
            tokens = set(re.findall(r"[a-z0-9]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).lower()))
            matched = tokens & SENSITIVE_KEYWORDS
            # "age" alongside a duration-context token (account_age_days) is not a
            # protected attribute; a bare/person "age" (customer_age) still is.
            if matched == {"age"} and (tokens & AGE_DURATION_CONTEXT):
                matched = set()
            sensitive = bool(matched)
            importance = "MEDIUM"
            if any(kw in lower for kw in IMPORTANCE_HINTS["high"]):
                importance = "HIGH"
                high_cols.append(name)
            elif any(kw in lower for kw in IMPORTANCE_HINTS["low"]):
                importance = "LOW"
            elif col_profile.detected_type == ColumnType.ID:
                importance = "IRRELEVANT"

            transforms = self._default_transforms(name, col_profile.detected_type)
            quality = "None detected"
            if col_profile.missing_pct > 5:
                quality = f"{col_profile.missing_pct:.1f}% missing values"

            columns[name] = {
                "meaning": self._guess_meaning(name),
                "importance": importance,
                "sensitive": sensitive,
                "suggested_transforms": transforms,
                "quality_issues": quality,
            }

        interactions = []
        if len(high_cols) >= 2:
            interactions.append(f"{high_cols[0]} × {high_cols[1]}")

        leakage = []
        for name in report.column_profiles:
            if target_column.lower() in name.lower() and name != target_column:
                leakage.append(f"'{name}' may leak target information")

        avg_missing = report.missing_analysis.get("avg_missing_rate", 0)
        summary = (
            f"Dataset quality score {report.quality_score}/100. "
            f"Average missing rate {avg_missing * 100:.1f}%. "
            f"Task: {task_description or 'not specified'}."
        )

        return SemanticProfile(
            columns=columns,
            data_quality_summary=summary,
            key_interactions=interactions,
            leakage_risks=leakage,
            source="heuristic",
        )

    def _guess_meaning(self, name: str) -> str:
        readable = re.sub(r"[_-]+", " ", name).strip()
        return f"Column representing {readable}"

    def _default_transforms(self, name: str, col_type: ColumnType) -> list[str]:
        if col_type == ColumnType.DATETIME:
            return [
                f"Extract day of week from {name}",
                f"Extract month from {name}",
                f"Create is_weekend flag from {name}",
            ]
        if col_type == ColumnType.NUMERICAL:
            return [
                f"Log transform {name} if skewed",
                f"Z-score normalize {name}",
                f"Percentile rank for {name}",
            ]
        if col_type in (ColumnType.CATEGORICAL, ColumnType.BINARY):
            return [f"Frequency encode {name}", f"One-hot encode {name}"]
        return []
