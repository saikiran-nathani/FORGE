"""Integration tests for deployment pipeline."""

import json

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forge.deployment.deploy_manager import DeployManager
from forge.deployment.inference import ModelServer
from forge.monitoring.drift_monitor import DriftMonitor


@pytest.fixture
def mock_artifact_dir(tmp_path):
    X = pd.DataFrame({"age": [25, 30, 35, 40], "income": [50000.0, 60000.0, 70000.0, 80000.0]})
    y = pd.Series([0, 0, 1, 1])

    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(X, y)

    profile = {
        "n_rows": 4, "n_cols": 3, "quality_score": 90,
        "recommended_metric": "f1", "memory_usage_mb": 0.1,
        "column_types": {"age": "numerical", "income": "numerical", "target": "binary"},
        "target_analysis": {"task_type": "classification", "n_unique": 2, "is_binary": True},
        "missing_analysis": {"missing_pct_by_column": {}, "avg_missing_rate": 0},
        "outlier_analysis": {"per_column": {}},
    }

    preprocessor = ColumnTransformer([("num", StandardScaler(), ["age", "income"])], remainder="drop")
    preprocessor.fit(X)

    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "label_encoder": None,
        "feature_names": ["age", "income"],
        "model_name": "test_model",
        "params": {},
        "metadata": {
            "input_columns": ["age", "income"],
            "numerical_cols": ["age", "income"],
            "categorical_cols": [],
            "clip_bounds": {},
        },
        "profile": profile,
        "task_type": "binary_classification",
        "target_column": "target",
        "selected_features": [],
    }

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    joblib.dump(bundle, artifact_dir / "best_model.joblib")
    (artifact_dir / "profile.json").write_text(json.dumps(profile))
    (artifact_dir / "test_metrics.json").write_text(json.dumps({"accuracy": 0.75}))
    X.to_parquet(artifact_dir / "reference_data.parquet")
    return artifact_dir


def test_deploy_manager(mock_artifact_dir):
    result = DeployManager(deployments_root=mock_artifact_dir.parent / "deploys").deploy(
        "test-exp", mock_artifact_dir, "Test task", "target",
        pd.read_parquet(mock_artifact_dir / "reference_data.parquet"),
    )
    assert result.deployment_dir.exists()
    assert (result.deployment_dir / "serve.py").exists()
    assert (result.deployment_dir / "model_card.md").exists()


def test_model_server_predict(mock_artifact_dir):
    server = ModelServer(mock_artifact_dir)
    result = server.predict({"age": 28, "income": 55000.0})
    assert "prediction" in result


def test_drift_monitor(mock_artifact_dir):
    ref = pd.read_parquet(mock_artifact_dir / "reference_data.parquet")
    ref = ref.assign(target=[0, 0, 1, 1])
    monitor = DriftMonitor(mock_artifact_dir / "monitoring")
    monitor.set_baseline(ref, "target")
    report = monitor.check_drift(ref, "target")
    assert report.get("overall_drift_score", 0) < 0.1
