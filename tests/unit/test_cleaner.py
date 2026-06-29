"""Unit tests for feature pipeline."""

from forge.feature_engineering.cleaner import FeaturePipeline
from forge.profiling.statistical_profiler import StatisticalProfiler


def test_feature_pipeline_classification(sample_classification_df):
    profiler = StatisticalProfiler("churn")
    profile = profiler.profile(sample_classification_df)

    pipeline = FeaturePipeline("churn", random_state=42)
    result = pipeline.fit_transform(sample_classification_df, profile, test_size=0.3)

    assert len(result.X_train) + len(result.X_test) == 10
    assert len(result.feature_names) > 0
    assert result.label_encoder is not None


def test_feature_pipeline_regression(sample_regression_df):
    profiler = StatisticalProfiler("price")
    profile = profiler.profile(sample_regression_df)

    pipeline = FeaturePipeline("price", random_state=42)
    result = pipeline.fit_transform(sample_regression_df, profile, test_size=0.3)

    assert len(result.X_train) + len(result.X_test) == 10
    assert result.label_encoder is None
