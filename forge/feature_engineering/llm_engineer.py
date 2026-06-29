"""LLM-powered feature engineering agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forge.feature_engineering.code_sandbox import CodeSandbox, SandboxResult
from forge.llm.client import LLMClient
from forge.profiling.models import ColumnType, ProfileReport
from forge.profiling.semantic_profiler import SemanticProfile


@dataclass
class FeatureEngineeringResult:
    df: pd.DataFrame
    generated_features: list[dict[str, Any]] = field(default_factory=list)
    failed_transformations: list[dict[str, str]] = field(default_factory=list)


class LLMFeatureEngineer:
    """Generates and executes pandas feature transformations via LLM."""

    CODE_SYSTEM = "You are an expert feature engineer. Output ONLY executable Python code."

    def __init__(
        self,
        target_column: str,
        task_description: str = "",
        llm: LLMClient | None = None,
        max_retries: int = 3,
    ):
        self.target_column = target_column
        self.task_description = task_description
        self.llm = llm or LLMClient()
        self.sandbox = CodeSandbox()
        self.max_retries = max_retries

    def engineer(
        self,
        df: pd.DataFrame,
        profile: ProfileReport,
        semantic: SemanticProfile | None = None,
    ) -> FeatureEngineeringResult:
        result_df = df.copy()
        generated: list[dict[str, Any]] = []
        failed: list[dict[str, str]] = []

        feature_cols = [
            c for c in df.columns
            if c != self.target_column
            and profile.column_types.get(c) != ColumnType.ID
        ]

        for col in feature_cols:
            col_type = profile.column_types.get(col, ColumnType.CATEGORICAL)
            col_profile = profile.column_profiles.get(col)
            stats = col_profile.statistics if col_profile else {}
            semantic_info = (semantic.columns.get(col, {}) if semantic else {})

            code = self._generate_code(col, col_type, stats, semantic_info)
            sandbox_result = self._execute_with_retry(code, result_df, col)

            if sandbox_result.success and sandbox_result.df is not None:
                result_df = sandbox_result.df
                generated.append({
                    "source_column": col,
                    "new_columns": sandbox_result.new_columns,
                    "code": sandbox_result.code,
                    "execution_time_ms": sandbox_result.execution_time_ms,
                })
            else:
                failed.append({"column": col, "error": sandbox_result.error})

        result_df = self._apply_heuristic_features(result_df, profile, feature_cols)
        return FeatureEngineeringResult(
            df=result_df,
            generated_features=generated,
            failed_transformations=failed,
        )

    def _generate_code(
        self,
        column: str,
        col_type: ColumnType,
        stats: dict[str, Any],
        semantic_info: dict[str, Any],
    ) -> str:
        if self.llm.available:
            try:
                return self._llm_generate_code(column, col_type, stats, semantic_info)
            except Exception:
                pass
        return self._heuristic_code(column, col_type)

    def _llm_generate_code(
        self,
        column: str,
        col_type: ColumnType,
        stats: dict[str, Any],
        semantic_info: dict[str, Any],
    ) -> str:
        samples = stats.get("sample_values", [])[:10]
        prompt = f"""Column: {column}
Type: {col_type.value}
Sample values: {samples}
Statistics: {stats}
Task: {self.task_description}
Target: {self.target_column}
Suggested transforms: {semantic_info.get('suggested_transforms', [])}

Generate Python Pandas code that creates new features from this column.
Requirements:
- Use only pandas, numpy, sklearn.preprocessing (available as pd, np, preprocessing)
- Operate on variable `df`
- Column names in snake_case
- Do NOT modify original columns
- Do NOT use target column

Output ONLY Python code."""

        return self.llm.complete(self.CODE_SYSTEM, prompt)

    def _heuristic_code(self, column: str, col_type: ColumnType) -> str:
        safe = column.replace("'", "\\'")
        if col_type == ColumnType.DATETIME:
            return f"""
df['{safe}_day_of_week'] = pd.to_datetime(df['{safe}']).dt.dayofweek
df['{safe}_month'] = pd.to_datetime(df['{safe}']).dt.month
df['{safe}_is_weekend'] = pd.to_datetime(df['{safe}']).dt.dayofweek.isin([5, 6]).astype(int)
""".strip()
        if col_type == ColumnType.NUMERICAL:
            return f"""
df['{safe}_log'] = np.log1p(df['{safe}'].clip(lower=0))
df['{safe}_zscore'] = (df['{safe}'] - df['{safe}'].mean()) / (df['{safe}'].std() + 1e-8)
df['{safe}_rank'] = df['{safe}'].rank(pct=True)
""".strip()
        if col_type in (ColumnType.CATEGORICAL, ColumnType.BINARY):
            return f"""
_freq = df['{safe}'].value_counts(normalize=True).to_dict()
df['{safe}_frequency'] = df['{safe}'].map(_freq)
""".strip()
        return f"# no transform for {safe}"

    def _execute_with_retry(
        self, code: str, df: pd.DataFrame, column: str
    ) -> SandboxResult:
        result = self.sandbox.execute(code, df, self.target_column)
        if result.success or not self.llm.available:
            return result

        for _ in range(self.max_retries - 1):
            fix_prompt = f"""This code failed:
```python
{code}
```
Error: {result.error}

Fix the code. Column: {column}. Output ONLY corrected Python code."""
            try:
                fixed_code = self.llm.complete(self.CODE_SYSTEM, fix_prompt)
                result = self.sandbox.execute(fixed_code, df, self.target_column)
                if result.success:
                    return result
            except Exception:
                break
        return result

    def _apply_heuristic_features(
        self,
        df: pd.DataFrame,
        profile: ProfileReport,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        numerical = [
            c for c in feature_cols
            if profile.column_types.get(c) == ColumnType.NUMERICAL
        ]
        if len(numerical) >= 2:
            c1, c2 = numerical[0], numerical[1]
            df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
            df[f"{c1}_ratio_{c2}"] = df[c1] / (df[c2].replace(0, np.nan) + 1e-8)
        return df
