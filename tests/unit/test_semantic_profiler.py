"""Unit tests for semantic profiler."""

from forge.profiling.semantic_profiler import SemanticProfiler
from forge.profiling.statistical_profiler import StatisticalProfiler


def test_heuristic_semantic_profile(sample_classification_df):
    stat = StatisticalProfiler("churn").profile(sample_classification_df)
    semantic = SemanticProfiler().profile(stat, "Predict churn", "churn")
    assert semantic.source == "heuristic"
    assert "income" in semantic.columns
    assert semantic.columns["income"]["importance"] == "HIGH"
    assert isinstance(semantic.key_interactions, list)
