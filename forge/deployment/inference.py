"""Model loading and inference for deployed models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from forge.feature_engineering.cleaner import FeaturePipeline


class ModelServer:
    """Loads a FORGE model bundle and serves predictions."""

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        self.bundle = joblib.load(self.artifact_dir / "best_model.joblib")
        profile_path = self.artifact_dir / "profile.json"
        if profile_path.exists():
            self.bundle["profile"] = json.loads(profile_path.read_text())
        if "metadata" not in self.bundle:
            self.bundle["metadata"] = {}
        selection_path = self.artifact_dir / "feature_selection.json"
        if selection_path.exists():
            sel = json.loads(selection_path.read_text())
            self.bundle["selected_features"] = sel.get("selected", [])
        self.metrics = {}
        metrics_path = self.artifact_dir / "test_metrics.json"
        if metrics_path.exists():
            self.metrics = json.loads(metrics_path.read_text())
        self._pipeline = FeaturePipeline(
            target_column=self.bundle.get("target_column", "target")
        )

    @classmethod
    def from_artifact_dir(cls, path: str | Path) -> ModelServer:
        return cls(Path(path))

    def predict(self, features: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
        single = isinstance(features, dict)
        rows = [features] if single else features
        df = pd.DataFrame(rows)
        X = self._pipeline.transform_raw(df, self.bundle)
        model = self.bundle["model"]
        preds = model.predict(X)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)

        results = []
        for i, pred in enumerate(preds):
            entry: dict[str, Any] = {"prediction": self._decode_prediction(pred)}
            if proba is not None:
                if proba.shape[1] == 2:
                    entry["probability"] = float(proba[i, 1])
                    entry["confidence"] = float(max(proba[i]))
                else:
                    entry["probabilities"] = proba[i].tolist()
                    entry["confidence"] = float(proba[i].max())
            results.append(entry)
        return results[0] if single else results

    def _decode_prediction(self, pred: Any) -> Any:
        le = self.bundle.get("label_encoder")
        if le is not None:
            try:
                return le.inverse_transform([int(pred)])[0]
            except (ValueError, TypeError):
                pass
        if isinstance(pred, (np.floating, float)):
            return float(pred)
        return int(pred) if isinstance(pred, (np.integer, int)) else pred

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.bundle.get("model_name"),
            "feature_names": self.bundle.get("feature_names", []),
            "input_columns": self.bundle.get("metadata", {}).get("input_columns", []),
            "metrics": self.metrics,
            "task_type": self.bundle.get("task_type"),
        }
