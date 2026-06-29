"""Unit tests for statistical profiler."""

from forge.profiling.models import ColumnType
from forge.profiling.statistical_profiler import StatisticalProfiler


def test_type_detection(sample_classification_df):
    profiler = StatisticalProfiler("churn")
    report = profiler.profile(sample_classification_df)

    assert report.n_rows == 10
    assert report.n_cols == 4
    assert report.column_types["income"] == ColumnType.NUMERICAL
    assert report.column_types["city"] == ColumnType.CATEGORICAL
    assert 0 <= report.quality_score <= 100


def test_target_analysis_binary(sample_classification_df):
    profiler = StatisticalProfiler("churn")
    report = profiler.profile(sample_classification_df)

    assert report.target_analysis["task_type"] == "classification"
    assert report.target_analysis["is_binary"] is True
    assert report.recommended_metric in ("roc_auc", "f1")


def test_regression_target(sample_regression_df):
    profiler = StatisticalProfiler("price")
    report = profiler.profile(sample_regression_df)

    assert report.target_analysis["task_type"] == "regression"
    assert report.recommended_metric == "rmse"
