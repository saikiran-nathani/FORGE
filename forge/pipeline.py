"""End-to-end FORGE pipeline orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
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


class ForgePipeline:
    """Full pipeline: profile → LLM features → train → evaluate → explain → report."""

    def __init__(self, config: ForgeConfig):
        self.config = config
        self.router = TaskRouter()
        self.metrics_calc = MetricsCalculator()
        self.llm = LLMClient()

    def run(self, dataset_path: Path) -> PipelineResult:
        config = self.config
        output_dir = config.ensure_output_dir()
        artifact_dir = output_dir / dataset_path.stem
        artifact_dir.mkdir(parents=True, exist_ok=True)

        console.print("[bold blue]FORGE[/bold blue] — Loading dataset...")
        df = self._load_dataset(dataset_path)
        console.print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
        n_original_features = len(df.columns) - 1

        console.print("\n[bold]Stage 1:[/bold] Data Profiling")
        profiler = StatisticalProfiler(config.target_column, config.task_description)
        profile = profiler.profile(df)
        console.print(f"  Quality score: {profile.quality_score}/100")
        console.print(f"  Recommended metric: {profile.recommended_metric}")

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
        working_df = df
        fe_result = None
        if config.enable_llm:
            engineer = LLMFeatureEngineer(
                config.target_column, config.task_description, self.llm
            )
            fe_result = engineer.engineer(df, profile, semantic)
            working_df = fe_result.df
            console.print(f"  LLM features: {len(fe_result.generated_features)} transformations")

        fe = FeaturePipeline(config.target_column, config.random_state)
        features = fe.fit_transform(working_df, profile, config.test_size)
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
        model_results: list[ModelResult] = []
        with mlflow.start_run(run_name=dataset_path.stem):
            for model_cls in model_registry:
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
                result = opt.optimize_model(model_cls, features.X_train, features.y_train)
                model_results.append(result)
                console.print(
                    f"    CV {result.metric_name}: {result.cv_score:.4f} "
                    f"(±{result.cv_score_std:.4f}) | {result.inference_latency_ms:.1f}ms"
                )
                mlflow.log_metrics({
                    f"{result.model_name}_cv_score": result.cv_score,
                    f"{result.model_name}_latency_ms": result.inference_latency_ms,
                })

            if config.enable_ensembles and len(model_results) >= 3:
                ensemble_builder = EnsembleBuilder(task_type, config.random_state)
                for build_fn, label in [
                    (ensemble_builder.build_voting, "voting"),
                    (ensemble_builder.build_stacking, "stacking"),
                ]:
                    console.print(f"  Training [cyan]{label}[/cyan] ensemble...")
                    model, name, cv_score = build_fn(
                        model_results, features.X_train, features.y_train
                    )
                    model_results.append(ModelResult(
                        model_name=name, best_params={}, cv_score=cv_score,
                        cv_score_std=0.0, training_time=0.0, inference_latency_ms=0.0,
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
        y_pred = best.fitted_model.predict(features.X_test)
        y_proba = getattr(best.fitted_model, "predict_proba", lambda x: None)(features.X_test)

        is_binary = task_type == TaskType.BINARY_CLASSIFICATION
        task_str = "regression" if task_type == TaskType.REGRESSION else "classification"
        test_metrics = self.metrics_calc.compute(
            features.y_test.values, y_pred, y_proba, task_str, is_binary
        )
        self._print_results(model_results, best, test_metrics)

        explain_dir = artifact_dir / "explainability"
        shap_summary: dict[str, Any] = {}
        if config.enable_shap:
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
            test_indices = features.X_test.index
            fairness_report = FairnessAuditor().audit(
                df.loc[test_indices], features.y_test.values, y_pred, y_proba,
                semantic, artifact_dir / "fairness",
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
            artifact_dir, profile, semantic, test_metrics, fe_result,
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
            generated_features=fe_result.generated_features if fe_result else [],
            feature_selection=selection_report,
            shap_summary=shap_summary,
            error_analysis=error_analysis,
            fairness_report=fairness_report,
            llm_report_path=str(report_path),
            output_dir=artifact_dir,
        )

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

    def _save_artifacts(self, artifact_dir, profile, semantic, metrics, fe_result, selection, pareto):
        with open(artifact_dir / "profile.json", "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        with open(artifact_dir / "semantic_profile.json", "w") as f:
            json.dump(semantic.to_dict(), f, indent=2)
        with open(artifact_dir / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(artifact_dir / "pareto_frontier.json", "w") as f:
            json.dump(pareto, f, indent=2)
        if fe_result:
            with open(artifact_dir / "llm_features.json", "w") as f:
                json.dump(fe_result.generated_features, f, indent=2)
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
