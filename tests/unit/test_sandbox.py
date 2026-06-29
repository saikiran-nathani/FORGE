"""Unit tests for code sandbox."""

import pandas as pd

from forge.feature_engineering.code_sandbox import CodeSandbox


def test_sandbox_executes_valid_code():
    df = pd.DataFrame({"age": [25, 30, 35], "income": [50000, 60000, 70000]})
    code = "df['age_doubled'] = df['age'] * 2"
    result = CodeSandbox().execute(code, df, "target")
    assert result.success
    assert "age_doubled" in result.new_columns


def test_sandbox_blocks_target_reference():
    df = pd.DataFrame({"age": [25, 30], "target": [0, 1]})
    code = "df['leak'] = df['target']"
    result = CodeSandbox().execute(code, df, "target")
    assert not result.success


def test_sandbox_blocks_forbidden_import():
    df = pd.DataFrame({"age": [25, 30]})
    code = "import os\ndf['bad'] = 1"
    result = CodeSandbox().execute(code, df, "target")
    assert not result.success
