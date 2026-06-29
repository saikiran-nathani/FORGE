"""Safe execution environment for LLM-generated pandas code."""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

FORBIDDEN_IMPORTS = {"os", "sys", "subprocess", "socket", "requests", "shutil", "pathlib"}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "getattr", "globals", "locals"}
ALLOWED_MODULES = {"pandas", "numpy", "sklearn", "math", "datetime", "re", "pd", "np"}


@dataclass
class SandboxResult:
    success: bool
    df: pd.DataFrame | None = None
    new_columns: list[str] = field(default_factory=list)
    error: str = ""
    execution_time_ms: float = 0.0
    code: str = ""


class CodeSandbox:
    """Executes and validates LLM-generated feature engineering code."""

    def __init__(self, timeout_seconds: int = 30, max_nan_pct: float = 0.5):
        self.timeout_seconds = timeout_seconds
        self.max_nan_pct = max_nan_pct

    def execute(self, code: str, df: pd.DataFrame, target_column: str) -> SandboxResult:
        code = self._extract_code(code)
        validation_error = self._validate_code(code, target_column)
        if validation_error:
            return SandboxResult(success=False, error=validation_error, code=code)

        df_copy = df.copy()
        original_cols = set(df_copy.columns)
        namespace = self._build_namespace(df_copy)

        start = time.perf_counter()
        try:
            exec(compile(code, "<llm_feature>", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            return SandboxResult(
                success=False,
                error=str(exc),
                code=code,
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        result_df = namespace.get("df", df_copy)
        if not isinstance(result_df, pd.DataFrame):
            return SandboxResult(success=False, error="Code must operate on `df`", code=code)

        elapsed = (time.perf_counter() - start) * 1000
        if elapsed > self.timeout_seconds * 1000:
            return SandboxResult(success=False, error="Execution timeout", code=code)

        output_error = self._validate_output(result_df, len(df), original_cols, target_column)
        if output_error:
            return SandboxResult(success=False, error=output_error, code=code, df=result_df)

        new_cols = [c for c in result_df.columns if c not in original_cols]
        return SandboxResult(
            success=True,
            df=result_df,
            new_columns=new_cols,
            execution_time_ms=elapsed,
            code=code,
        )

    def _extract_code(self, text: str) -> str:
        text = text.strip()
        match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def _validate_code(self, code: str, target_column: str) -> str:
        if target_column in code:
            return f"Code must not reference target column '{target_column}'"
        for pattern in FORBIDDEN_IMPORTS:
            if re.search(rf"\bimport\s+{pattern}\b|\bfrom\s+{pattern}\b", code):
                return f"Forbidden import: {pattern}"
        for call in FORBIDDEN_CALLS:
            if re.search(rf"\b{call}\s*\(", code):
                return f"Forbidden function: {call}"
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"Syntax error: {exc}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in ALLOWED_MODULES:
                        return f"Forbidden import: {alias.name}"
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] not in ALLOWED_MODULES:
                    return f"Forbidden import: {node.module}"
        return ""

    def _build_namespace(self, df: pd.DataFrame) -> dict[str, Any]:
        import math
        import datetime

        from sklearn import preprocessing

        return {
            "df": df,
            "pd": pd,
            "np": np,
            "math": math,
            "datetime": datetime,
            "preprocessing": preprocessing,
        }

    def _validate_output(
        self,
        df: pd.DataFrame,
        expected_rows: int,
        original_cols: set[str],
        target_column: str,
    ) -> str:
        if len(df) != expected_rows:
            return f"Row count changed: {expected_rows} → {len(df)}"
        new_cols = [c for c in df.columns if c not in original_cols]
        if not new_cols:
            return "No new columns created"
        for col in new_cols:
            if col == target_column:
                return f"Cannot create column named '{target_column}'"
            nan_pct = df[col].isna().mean()
            if nan_pct > self.max_nan_pct:
                return f"Column '{col}' has {nan_pct:.0%} NaN (max {self.max_nan_pct:.0%})"
            if df[col].nunique(dropna=True) <= 1:
                return f"Column '{col}' is constant"
        return ""
