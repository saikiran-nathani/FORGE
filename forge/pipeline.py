"""End-to-end FORGE pipeline orchestrator."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from forge.config import ForgeConfig
from forge.evaluation.error_analysis import ErrorAnalyzer
from forge.evaluation.explainability import ExplainabilityEngine
from forge.evaluation.fairness_auditor import FairnessAuditor
from forge.evaluation.llm_report_generator import LLMReportGenerator
from forge.evaluation.metrics_calculator import MetricsCalculator
from forge.feature_engineering.cleaner import FeaturePipeline
from forge.feature_engineering.feature_selector import FeatureSelector
from forge.feature_engineering.llm_engineer import LLMFeatureEngineer
from forge.llm.client import LLMClient
from forge.profiling.eda_report import EDAReportGenerator
from forge.profiling.semantic_profiler import SemanticProfiler
from forge.profiling.statistical_profiler import StatisticalProfiler
from forge.training.classical import BASE_MODELS, FAST_MODELS
from forge.training.classical.ensembles import EnsembleBuilder
from forge.training.hpo.optuna_optimizer import ModelResult, OptunaOptimizer
from forge.training.hpo.pareto import ParetoAnalyzer
from forge.training.task_router import TaskRouter, TaskType

console = Console()


def _load_dl_models() -> list:
    """Import deep-learning models lazily. Returns [] if torch isn't installed.

    Keeps torch out of the default/fast-mode import path so the deployed API can
    run without the (~2GB) deep-learning extra. Install with: pip install -e '.[deep-learning]'
    """
    try:
        from forge.training.deep_learning.models import MLPModel, TabTransformerModel

        return [MLPModel, TabTransformerModel]
    except ImportError:
        return []


@dataclass
class PipelineResult:
    profile: dict[str, Any]
    semantic_profile: dict[str, Any]
    task_type: str
    model_results: list[dict[str, Any]]
    pareto_frontier: list[dict[str, Any]]
    best_model_name: str
    best_metrics: dict[str, Any]
    generated_features: list[dict[str, Any]]
    feature_selection: dict[str, Any]
    shap_summary: dict[str, Any]
    error_analysis: dict[str, Any]
    fairness_report: dict[str, Any]
    llm_report_path: str
    output_dir: Path
    baseline_metrics: dict[str, Any]
    eval_context: dict[str, Any]
    warnings: list[str]


class ForgePipeline:
    """Full pipeline: profile → LLM features → train → evaluate → explain → report."""

    def __init__(self, config: ForgeConfig, progress_cb=None):
        self.config = config
        self.router = TaskRouter()
        self.metrics_calc = MetricsCalculator()
        self.llm = LLMClient()
        self._progress_cb = progress_cb

    def _progress(self, stage: str, message: str) -> None:
        """Report a stage/step update to an optional consumer (e.g. the API)."""
        if self._progress_cb is not None:
            try:
                self._progress_cb(stage, message)
            except Exception:
                pass

    def run(self, dataset_path: Path) -> PipelineResult:
        config = self.config
        output_dir = config.ensure_output_dir()
        artifact_dir = output_dir / dataset_path.stem
        artifact_dir.mkdir(parents=True, exist_ok=True)

        console.print("[bold blue]FORGE[/bold blue] — Loading dataset...")
        df = self._load_dataset(dataset_path)
        console.print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
        self._validate_target(df)
        n_original_features = len(df.columns) - 1

        console.print("\n[bold]Stage 1:[/bold] Data Profiling")
        self._progress("profiling", f"Profiling {len(df):,} rows × {len(df.columns)} columns…")
        profiler = StatisticalProfiler(config.target_column, config.task_description)
        profile = profiler.profile(df)
        console.print(f"  Quality score: {profile.quality_score}/100")
        console.print(f"  Recommended metric: {profile.recommended_metric}")
        self._progress("profiling", f"Quality {profile.quality_score}/100 · metric {profile.recommended_metric}")

        semantic = SemanticProfiler(self.llm).profile(
            profile, config.task_description, config.target_column
        )
        console.print(f"  Semantic analysis: {semantic.source}")

        task_type = self.router.detect(profile, config.task_description)
        console.print(f"  Task type: {task_type.value}")

        eda_path = artifact_dir / "eda_report.html"
        EDAReportGenerator().generate(df, profile, semantic, config.target_column, eda_path)
        console.print(f"  EDA report: {eda_path}")

        console.print("\n[bold]Stage 2:[/bold] Feature Engineering")
        self._progress("feature_engineering", "Engineering & selecting features…")
        # LLM-suggested transforms are ADVISORY: surfaced, but NOT fed to the model.
        # Arbitrary/stateful generated code can't be safely replayed on a single
        # prediction row, so the model trains on deterministic STATELESS features
        # (log / datetime-part / interaction) added inside the feature pipeline and
        # recreated identically at inference. (Previously the engineered df was
        # passed here but silently dropped by the preprocessor — bug #28.)
        advisory_features: list[dict[str, Any]] = []
        if config.enable_llm and self.llm.available:
            engineer = LLMFeatureEngineer(
                config.target_column, config.task_description, self.llm
            )
            advisory_features = engineer.engineer(df, profile, semantic).generated_features
            console.print(f"  LLM advisory suggestions: {len(advisory_features)}")

        fe = FeaturePipeline(config.target_column, config.random_state)
        features = fe.fit_transform(df, profile, config.test_size)
        engineered_features = features.metadata.get("engineered_features", [])
        console.print(f"  Engineered (stateless) features: {len(engineered_features)}")
        self._progress("feature_engineering", f"Added {len(engineered_features)} stateless features")
        console.print(f"  Preprocessed features: {len(features.feature_names)}")

        selection_report: dict[str, Any] = {}
        if config.enable_feature_selection and len(features.feature_names) > 5:
            selector = FeatureSelector(task_type, config.target_column, config.random_state)
            selection = selector.select(features.X_train, features.X_test, features.y_train)
            features.X_train = selection.X_train
            features.X_test = selection.X_test
            features.feature_names = selection.selected_features
            selection_report = {
                "selected": selection.selected_features,
                "removed": selection.removed_features,
                "importance": selection.importance_scores,
            }
            console.print(f"  After selection: {len(selection.selected_features)} features")

        console.print(f"  Train: {len(features.X_train):,} | Test: {len(features.X_test):,}")

        console.print("\n[bold]Stage 3:[/bold] Model Training + HPO")
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment_name)

        model_registry = self._model_registry(task_type, config)
        dl_trials = min(config.hpo_trials_per_model, 5)
        n_models = len(model_registry)
        model_results: list[ModelResult] = []
        with mlflow.start_run(run_name=dataset_path.stem):
            for i, model_cls in enumerate(model_registry, start=1):
                is_dl = getattr(model_cls, "family", "") == "deep_learning"
                trials = dl_trials if is_dl else config.hpo_trials_per_model
                opt = OptunaOptimizer(
                    task_type=task_type,
                    metric_name=profile.recommended_metric,
                    n_trials=trials,
                    cv_folds=config.cv_folds,
                    random_state=config.random_state,
                )
                console.print(f"  Training [cyan]{model_cls.name}[/cyan]...")
                self._progress("training", f"Training & tuning {model_cls.name} · model {i}/{n_models}")
                # One misbehaving model family must not sink the whole run.
                try:
                    result = opt.optimize_model(model_cls, features.X_train, features.y_train)
                except Exception as exc:
                    console.print(f"    [yellow]⚠ {model_cls.name} failed — skipped ({str(exc).splitlines()[0][:80]})[/yellow]")
                    self._progress("training", f"{model_cls.name} failed — skipped ({i}/{n_models})")
                    continue
                model_results.append(result)
                console.print(
                    f"    CV {result.metric_name}: {result.cv_score:.4f} "
                    f"(±{result.cv_score_std:.4f}) | {result.inference_latency_ms:.1f}ms"
                )
                self._progress("training", f"{model_cls.name}: {result.metric_name} {result.cv_score:.4f} ({i}/{n_models})")
                mlflow.log_metrics({
                    f"{result.model_name}_cv_score": result.cv_score,
                    f"{result.model_name}_latency_ms": result.inference_latency_ms,
                })

            if not model_results:
                raise RuntimeError(
                    "All models failed to train. Check the dataset for issues "
                    "(unparseable columns, a non-predictive/constant target, or too few rows)."
                )

            if config.enable_ensembles and len(model_results) >= 3:
                ensemble_builder = EnsembleBuilder(
                    task_type, profile.recommended_metric, config.random_state, config.cv_folds
                )
                for build_fn, label in [
                    (ensemble_builder.build_voting, "voting"),
                    (ensemble_builder.build_stacking, "stacking"),
                ]:
                    console.print(f"  Training [cyan]{label}[/cyan] ensemble...")
                    self._progress("training", f"Building {label} ensemble…")
                    try:
                        model, name, cv_score = build_fn(
                            model_results, features.X_train, features.y_train
                        )
                    except Exception as exc:
                        console.print(f"    [yellow]⚠ {label} ensemble failed — skipped ({str(exc).splitlines()[0][:80]})[/yellow]")
                        continue
                    # Measure REAL inference latency — an ensemble wraps N base
                    # models and is the slowest to predict, so a hardcoded 0.0 made
                    # it always Pareto-optimal. Same protocol as the base models in
                    # OptunaOptimizer: time 10 predicts on a 100-row sample.
                    sample = features.X_train.head(min(100, len(features.X_train)))
                    lat_start = time.perf_counter()
                    for _ in range(10):
                        model.predict(sample)
                    latency_ms = (time.perf_counter() - lat_start) / 10 * 1000
                    model_results.append(ModelResult(
                        model_name=name, best_params={}, cv_score=cv_score,
                        cv_score_std=0.0, training_time=0.0, inference_latency_ms=latency_ms,
                        fitted_model=model, metric_name=profile.recommended_metric,
                    ))

            best = self._pick_best(model_results, profile.recommended_metric)

        pareto = ParetoAnalyzer().to_dict(
            ParetoAnalyzer().compute(
                self._serialize_results(model_results),
                maximize_score=profile.recommended_metric != "rmse",
            )
        )
        console.print(f"  Pareto-optimal models: {sum(1 for p in pareto if p['is_pareto_optimal'])}")

        console.print("\n[bold]Stage 4:[/bold] Evaluation + Explainability")
        self._progress("evaluation", f"Evaluating best model: {best.model_name}…")
        # CatBoost returns a 2-D (n, 1) array for classification; ravel so every
        # downstream consumer (metrics, error analysis, fairness) gets 1-D preds.
        y_pred = np.asarray(best.fitted_model.predict(features.X_test)).ravel()
        y_proba = getattr(best.fitted_model, "predict_proba", lambda x: None)(features.X_test)

        is_binary = task_type == TaskType.BINARY_CLASSIFICATION
        task_str = "regression" if task_type == TaskType.REGRESSION else "classification"
        test_metrics = self.metrics_calc.compute(
            features.y_test.values, y_pred, y_proba, task_str, is_binary,
            n_features=len(features.feature_names),
        )
        self._print_results(model_results, best, test_metrics)

        # Honest reference numbers — surfaced, never used to gate or alter the model.
        baseline_metrics = self._baseline_metrics(features, task_str, is_binary)
        eval_context = self._build_eval_context(features, task_type, best, test_metrics)
        warnings = self._build_warnings(eval_context, test_metrics, baseline_metrics, task_type)
        for w in warnings:
            console.print(f"  [yellow]⚠[/yellow] {w}")

        explain_dir = artifact_dir / "explainability"
        shap_summary: dict[str, Any] = {}
        if config.enable_shap:
            self._progress("evaluation", "Computing SHAP explanations…")
            shap_summary = ExplainabilityEngine().explain(
                best.fitted_model, features.X_test, features.feature_names,
                best.model_name, explain_dir, features.y_test, task_str,
                enable_lime=config.enable_lime, enable_pdp=config.enable_pdp,
            )

        error_analysis: dict[str, Any] = {}
        if config.enable_error_analysis:
            error_analysis = ErrorAnalyzer().analyze(
                best.fitted_model, features.X_test, features.y_test,
                y_pred, y_proba, task_str, features.feature_names,
                artifact_dir / "errors",
            )
            console.print(f"  Error analysis: {len(error_analysis.get('worst_predictions', []))} worst predictions")

        fairness_report: dict[str, Any] = {}
        if config.enable_fairness:
            self._progress("evaluation", "Auditing fairness across subgroups…")
            test_indices = features.X_test.index
            fairness_report = FairnessAuditor().audit(
                df.loc[test_indices], features.y_test.values, y_pred, y_proba,
                semantic, artifact_dir / "fairness", task_type=task_type.value,
            )
            if fairness_report.get("flags"):
                console.print(f"  Fairness flags: {len(fairness_report['flags'])}")

        report_path = artifact_dir / "analysis_report.md"
        if config.enable_llm_report:
            LLMReportGenerator(self.llm).generate(
                config.task_description, profile.to_dict(), semantic.to_dict(),
                best.model_name, test_metrics, shap_summary, error_analysis,
                fairness_report, n_original_features, len(features.feature_names),
                report_path,
            )
            console.print(f"  LLM report: {report_path}")

        self._progress("finalizing", "Packaging model, artifacts & model card…")
        df.drop(columns=[config.target_column], errors="ignore").to_parquet(
            artifact_dir / "reference_data.parquet"
        )

        bundle = {
            "model": best.fitted_model,
            "preprocessor": features.preprocessor,
            "label_encoder": features.label_encoder,
            "feature_names": features.feature_names,
            "model_name": best.model_name,
            "params": best.best_params,
            "metadata": features.metadata,
            "profile": profile.to_dict(),
            "task_type": task_type.value,
            "target_column": config.target_column,
            "selected_features": selection_report.get("selected", []),
        }
        joblib.dump(bundle, artifact_dir / "best_model.joblib")
        self._save_artifacts(
            artifact_dir, profile, semantic, test_metrics, engineered_features,
            selection_report, pareto,
        )

        from forge.deployment.model_card_generator import ModelCardGenerator
        ModelCardGenerator().generate(
            artifact_dir, config.task_description, config.target_column,
            artifact_dir / "model_card.md",
        )

        return PipelineResult(
            profile=profile.to_dict(),
            semantic_profile=semantic.to_dict(),
            task_type=task_type.value,
            model_results=self._serialize_results(model_results),
            pareto_frontier=pareto,
            best_model_name=best.model_name,
            best_metrics=test_metrics,
            generated_features=engineered_features,
            feature_selection=selection_report,
            shap_summary=shap_summary,
            error_analysis=error_analysis,
            fairness_report=fairness_report,
            llm_report_path=str(report_path),
            output_dir=artifact_dir,
            baseline_metrics=baseline_metrics,
            eval_context=eval_context,
            warnings=warnings,
        )

    def _baseline_metrics(self, features, task_str: str, is_binary: bool) -> dict[str, Any]:
        """Trivial baseline (majority class / mean) on the same split — a reference point."""
        try:
            if task_str == "regression":
                from sklearn.dummy import DummyRegressor

                dummy = DummyRegressor(strategy="mean")
                dummy.fit(features.X_train, features.y_train)
                y_pred = dummy.predict(features.X_test)
                proba = None
            else:
                from sklearn.dummy import DummyClassifier

                dummy = DummyClassifier(strategy="most_frequent")
                dummy.fit(features.X_train, features.y_train)
                y_pred = dummy.predict(features.X_test)
                proba = getattr(dummy, "predict_proba", lambda x: None)(features.X_test)
            return self.metrics_calc.compute(features.y_test.values, y_pred, proba, task_str, is_binary)
        except Exception as exc:  # baseline is best-effort; never break the run
            return {"error": str(exc)}

    def _build_eval_context(self, features, task_type: TaskType, best, test_metrics: dict) -> dict[str, Any]:
        import numpy as np

        ctx: dict[str, Any] = {
            "n_train": int(len(features.X_train)),
            "n_test": int(len(features.X_test)),
            "selection_metric": best.metric_name,
            "cv_best_score": float(best.cv_score),
        }
        test_val = test_metrics.get(best.metric_name)
        if isinstance(test_val, (int, float)):
            ctx["test_metric_value"] = float(test_val)
            ctx["cv_test_gap"] = float(best.cv_score) - float(test_val)
        if task_type != TaskType.REGRESSION:
            le = features.label_encoder
            classes, counts = np.unique(features.y_test.values, return_counts=True)
            dist = {}
            for c, n in zip(classes, counts):
                label = le.inverse_transform([int(c)])[0] if le is not None else str(c)
                dist[str(label)] = int(n)
            ctx["test_class_counts"] = dist
            total = sum(dist.values())
            if total:
                ctx["majority_fraction"] = max(dist.values()) / total
                ctx["minority_fraction"] = min(dist.values()) / total
            if le is not None and task_type == TaskType.BINARY_CLASSIFICATION and len(le.classes_) == 2:
                ctx["positive_class"] = str(le.classes_[1])
        return ctx

    def _build_warnings(self, ctx: dict, test_metrics: dict, baseline_metrics: dict, task_type: TaskType) -> list[str]:
        warnings: list[str] = []
        n = ctx.get("n_train", 0) + ctx.get("n_test", 0)
        if n and n < 500:
            warnings.append(f"Small dataset ({n} rows): CV/HPO estimates are high-variance and may not generalize.")
        mino = ctx.get("minority_fraction")
        if mino is not None and mino < 0.2:
            warnings.append(f"Imbalanced target ({mino:.0%} minority): prefer balanced accuracy / MCC / minority recall over accuracy.")
        gap = ctx.get("cv_test_gap")
        if gap is not None:
            metric = ctx.get("selection_metric", "")
            cv_val = ctx.get("cv_best_score")
            # RMSE/MAE/MAPE are unbounded & scale-dependent — a fixed 0.15 absolute
            # threshold always/never fires. Use a RELATIVE gap for those; the 0.15
            # absolute threshold only makes sense for [0,1]-bounded metrics.
            if metric in ("rmse", "mae", "mape") and cv_val:
                tripped = abs(gap) / max(abs(cv_val), 1e-9) > 0.15
            else:
                tripped = abs(gap) > 0.15
            if tripped:
                warnings.append(
                    f"Large CV→test gap on {metric} "
                    f"({ctx.get('cv_best_score'):.3f}→{ctx.get('test_metric_value'):.3f}): possible overfitting or split variance."
                )
        if isinstance(baseline_metrics, dict) and "error" not in baseline_metrics:
            if task_type == TaskType.REGRESSION:
                bm, base = test_metrics.get("rmse"), baseline_metrics.get("rmse")
                if bm is not None and base is not None and bm >= base:
                    warnings.append(f"Model RMSE ({bm:.3f}) does not beat the mean-prediction baseline ({base:.3f}).")
            else:
                # roc_auc can sit at ~0.5 (no signal) while accuracy matches the
                # majority baseline — check it explicitly, not just accuracy.
                ra, base_ra = test_metrics.get("roc_auc"), baseline_metrics.get("roc_auc")
                if ra is not None and base_ra is not None and ra <= base_ra + 0.02:
                    warnings.append(f"Model ROC-AUC ({ra:.3f}) barely exceeds the {base_ra:.3f} baseline — little discriminative signal.")
                for m in ("balanced_accuracy", "accuracy"):
                    bm, base = test_metrics.get(m), baseline_metrics.get(m)
                    if bm is not None and base is not None and bm < base:
                        warnings.append(f"Model {m.replace('_', ' ')} ({bm:.3f}) is below the majority-class baseline ({base:.3f}).")
                        break
        return warnings

    def _model_registry(self, task_type: TaskType, config: ForgeConfig) -> list:
        from forge.training.classical.linear_models import ElasticNetModel, LassoModel, RidgeModel
        from forge.training.classical.distance_models import NaiveBayesModel
        from forge.training.classical.logistic_regression import LogisticRegressionModel

        registry = list(FAST_MODELS if config.fast_mode else BASE_MODELS)
        if config.enable_deep_learning and not config.fast_mode:
            dl_models = _load_dl_models()
            if dl_models:
                registry.extend(dl_models)
            else:
                console.print("  [yellow]Deep-learning models skipped (torch not installed; pip install -e '.[deep-learning]')[/yellow]")
        if task_type == TaskType.REGRESSION:
            registry = [m for m in registry if m not in (LogisticRegressionModel, NaiveBayesModel)]
        else:
            registry = [m for m in registry if m not in (RidgeModel, LassoModel, ElasticNetModel)]
        return registry

    def _save_artifacts(self, artifact_dir, profile, semantic, metrics, engineered_features, selection, pareto):
        with open(artifact_dir / "profile.json", "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        with open(artifact_dir / "semantic_profile.json", "w") as f:
            json.dump(semantic.to_dict(), f, indent=2)
        with open(artifact_dir / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(artifact_dir / "pareto_frontier.json", "w") as f:
            json.dump(pareto, f, indent=2)
        if engineered_features:
            with open(artifact_dir / "engineered_features.json", "w") as f:
                json.dump(engineered_features, f, indent=2)
        if selection:
            with open(artifact_dir / "feature_selection.json", "w") as f:
                json.dump(selection, f, indent=2)

    def _pick_best(self, results: list[ModelResult], metric: str) -> ModelResult:
        if metric == "rmse":
            return min(results, key=lambda r: r.cv_score)
        return max(results, key=lambda r: r.cv_score)

    def _serialize_results(self, results: list[ModelResult]) -> list[dict[str, Any]]:
        return [
            {
                "model_name": r.model_name,
                "cv_score": r.cv_score,
                "cv_score_std": r.cv_score_std,
                "training_time": r.training_time,
                "inference_latency_ms": r.inference_latency_ms,
                "best_params": r.best_params,
            }
            for r in results
        ]

    def _validate_target(self, df: pd.DataFrame) -> None:
        target = self.config.target_column
        if target in df.columns:
            return
        import difflib

        matches = difflib.get_close_matches(str(target), [str(c) for c in df.columns], n=1)
        hint = f" Did you mean '{matches[0]}'?" if matches else ""
        cols = ", ".join(str(c) for c in list(df.columns)[:20])
        more = "" if len(df.columns) <= 20 else f", … (+{len(df.columns) - 20} more)"
        raise ValueError(
            f"Target column '{target}' was not found in the dataset.{hint} "
            f"Available columns: {cols}{more}"
        )

    def _load_dataset(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".json":
            return pd.read_json(path)
        raise ValueError(f"Unsupported file format: {suffix}")

    def _print_results(self, results, best, test_metrics):
        table = Table(title="Model Comparison (CV)")
        table.add_column("Model")
        table.add_column("CV Score", justify="right")
        table.add_column("Latency (ms)", justify="right")
        for r in sorted(results, key=lambda x: -x.cv_score):
            marker = " ★" if r.model_name == best.model_name else ""
            table.add_row(
                r.model_name + marker,
                f"{r.cv_score:.4f}",
                f"{r.inference_latency_ms:.1f}",
            )
        console.print(table)
        console.print(f"\n[bold green]Best model:[/bold green] {best.model_name}")
        for k, v in test_metrics.items():
            if k != "confusion_matrix":
                console.print(f"  {k}: {v}")
