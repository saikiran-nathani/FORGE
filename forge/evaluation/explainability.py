"""SHAP, LIME, PDP, and permutation importance explainability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ExplainabilityEngine:
    """Generates multi-method model explanations."""

    def explain(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_names: list[str],
        model_name: str,
        output_dir: Path,
        y: pd.Series | None = None,
        task_type: str = "classification",
        max_samples: int = 200,
        enable_lime: bool = True,
        enable_pdp: bool = True,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        sample = X.head(min(max_samples, len(X)))

        summary = self._shap_explain(model, sample, feature_names, model_name, output_dir)
        if summary.get("error"):
            summary = {"model_name": model_name, "top_features": [], "n_samples_explained": len(sample)}

        if enable_lime and task_type == "classification":
            summary["lime"] = self._lime_explain(model, sample, feature_names, task_type, output_dir)

        if enable_pdp and y is not None:
            summary["pdp"] = self._pdp_explain(model, sample, feature_names, output_dir)
            summary["permutation_importance"] = self._permutation_importance(
                model, sample, y.head(len(sample)), feature_names
            )

        with open(output_dir / "explainability.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        return summary

    def _shap_explain(
        self,
        model: Any,
        sample: pd.DataFrame,
        feature_names: list[str],
        model_name: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        import shap

        explainer, shap_values = self._get_shap_values(model, sample, model_name)
        if shap_values is None:
            return {"error": "Could not compute SHAP values"}

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        mean_abs = np.abs(shap_values).mean(axis=0)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=0)
        mean_abs = np.asarray(mean_abs).flatten()

        importance = {
            feature_names[i]: float(mean_abs[i])
            for i in range(min(len(feature_names), len(mean_abs)))
        }
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:20]
        summary = {
            "model_name": model_name,
            "top_features": [{"feature": f, "mean_abs_shap": v} for f, v in top_features],
            "n_samples_explained": len(sample),
        }

        with open(output_dir / "shap_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            shap.summary_plot(
                shap_values, sample,
                feature_names=feature_names[: shap_values.shape[1]],
                show=False, max_display=15,
            )
            plt.tight_layout()
            plt.savefig(output_dir / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        return summary

    def _lime_explain(
        self,
        model: Any,
        sample: pd.DataFrame,
        feature_names: list[str],
        task_type: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        try:
            from lime.lime_tabular import LimeTabularExplainer

            mode = "classification" if task_type == "classification" else "regression"
            explainer = LimeTabularExplainer(
                sample.values,
                feature_names=feature_names,
                mode=mode,
                discretize_continuous=True,
            )
            predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
            instance_idx = 0
            exp = explainer.explain_instance(
                sample.iloc[instance_idx].values,
                predict_fn,
                num_features=min(10, len(feature_names)),
            )
            explanations = [{"feature": f, "weight": float(w)} for f, w in exp.as_list()]
            result = {"instance_index": instance_idx, "explanations": explanations}
            with open(output_dir / "lime_explanation.json", "w") as f:
                json.dump(result, f, indent=2)
            return result
        except Exception as exc:
            return {"error": str(exc)}

    def _pdp_explain(
        self,
        model: Any,
        sample: pd.DataFrame,
        feature_names: list[str],
        output_dir: Path,
        top_n: int = 5,
    ) -> dict[str, Any]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.inspection import PartialDependenceDisplay

            top_cols = list(sample.columns[:top_n])
            fig, ax = plt.subplots(figsize=(10, 6))
            PartialDependenceDisplay.from_estimator(
                model, sample, top_cols, ax=ax, n_cols=min(3, len(top_cols))
            )
            plt.tight_layout()
            plt.savefig(output_dir / "pdp_plots.png", dpi=120, bbox_inches="tight")
            plt.close()
            return {"features": top_cols, "plot": "pdp_plots.png"}
        except Exception as exc:
            return {"error": str(exc)}

    def _permutation_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
    ) -> dict[str, float]:
        try:
            from sklearn.inspection import permutation_importance

            scoring = "f1_macro" if len(np.unique(y)) <= 20 else "neg_mean_squared_error"
            result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=1, scoring=scoring)
            return {
                feature_names[i]: float(result.importances_mean[i])
                for i in range(min(len(feature_names), len(result.importances_mean)))
            }
        except Exception:
            return {}

    def _get_shap_values(
        self,
        model: Any,
        X: pd.DataFrame,
        model_name: str,
    ) -> tuple[Any, Any]:
        import shap

        tree_models = {"xgboost", "lightgbm", "catboost", "random_forest", "extra_trees", "decision_tree"}
        dl_models = {"mlp", "tab_transformer"}
        try:
            if any(t in model_name for t in tree_models):
                explainer = shap.TreeExplainer(model)
                return explainer, explainer.shap_values(X)
            if any(t in model_name for t in dl_models):
                import torch
                bg = shap.sample(X, min(50, len(X)))
                if hasattr(model, "model_") and model.model_ is not None:
                    model.model_.eval()
                    def predict_fn(x):
                        device = next(model.model_.parameters()).device
                        with torch.no_grad():
                            out = model.model_(torch.tensor(x, dtype=torch.float32, device=device))
                            if out.shape[-1] == 1:
                                return torch.sigmoid(out).cpu().numpy()
                            return torch.softmax(out, dim=-1).cpu().numpy()
                    explainer = shap.DeepExplainer(model.model_, torch.tensor(bg.values, dtype=torch.float32))
                    return explainer, explainer.shap_values(torch.tensor(X.values, dtype=torch.float32))
            if hasattr(model, "predict_proba"):
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, min(50, len(X))))
                return explainer, explainer.shap_values(X)
            explainer = shap.KernelExplainer(model.predict, shap.sample(X, min(50, len(X))))
            return explainer, explainer.shap_values(X)
        except Exception:
            return None, None
