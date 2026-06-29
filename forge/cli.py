"""FORGE CLI entry point."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from forge.config import ForgeConfig
from forge.pipeline import ForgePipeline

app = typer.Typer(
    name="forge",
    help="FORGE — LLM-Powered Automated ML Pipeline Builder",
    add_completion=False,
)
console = Console()
server_app = typer.Typer(help="Run FORGE services")
app.add_typer(server_app, name="serve")


@app.command()
def run(
    dataset: Path = typer.Argument(..., help="Path to CSV/Parquet/JSON dataset"),
    target: str = typer.Option(..., "--target", "-t", help="Target column name"),
    task: str = typer.Option("", "--task", help="Natural language task description"),
    output: Path = typer.Option(Path("outputs"), "--output", "-o", help="Output directory"),
    trials: int = typer.Option(30, "--trials", help="HPO trials per model"),
    test_size: float = typer.Option(0.2, "--test-size", help="Hold-out test fraction"),
    fast: bool = typer.Option(False, "--fast", help="Use fast model subset only"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Disable LLM feature engineering"),
    no_shap: bool = typer.Option(False, "--no-shap", help="Disable SHAP explainability"),
) -> None:
    """Run the full FORGE pipeline on a dataset."""
    if not dataset.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {dataset}")
        raise typer.Exit(1)

    config = ForgeConfig(
        target_column=target,
        task_description=task,
        output_dir=output,
        hpo_trials_per_model=trials,
        test_size=test_size,
        fast_mode=fast,
        enable_llm=not no_llm,
        enable_shap=not no_shap,
    )

    pipeline = ForgePipeline(config)
    result = pipeline.run(dataset)
    console.print(f"\n[bold green]Done![/bold green] Artifacts saved to {result.output_dir}")


@server_app.command("api")
def serve_api(
    host: str = typer.Option("0.0.0.0", help="Host"),
    port: int = typer.Option(8000, help="Port"),
    reload: bool = typer.Option(False, help="Auto-reload"),
) -> None:
    """Start the FastAPI backend."""
    import uvicorn

    uvicorn.run("forge.api.app:app", host=host, port=port, reload=reload)


@app.command()
def deploy(
    artifacts: Path = typer.Argument(..., help="Path to experiment artifact directory"),
    experiment_id: str = typer.Option("cli", "--id", help="Experiment ID label"),
    target: str = typer.Option("", "--target", "-t", help="Target column name"),
    task: str = typer.Option("", "--task", help="Task description"),
) -> None:
    """Deploy a trained model as a production API."""
    import json

    import pandas as pd

    from forge.deployment.deploy_manager import DeployManager

    if not (artifacts / "best_model.joblib").exists():
        console.print(f"[red]Error:[/red] No model found in {artifacts}")
        raise typer.Exit(1)

    bundle = __import__("joblib").load(artifacts / "best_model.joblib")
    target = target or bundle.get("target_column", "target")
    ref_path = artifacts / "reference_data.parquet"
    reference = pd.read_parquet(ref_path) if ref_path.exists() else None

    result = DeployManager().deploy(experiment_id, artifacts, task, target, reference)
    console.print(f"[bold green]Deployed![/bold green] ID: {result.deployment_id}")
    console.print(f"  Directory: {result.deployment_dir}")
    console.print(f"  Model card: {result.model_card_path}")
    console.print(f"  Run: cd {result.deployment_dir} && docker compose up")


@app.command()
def profile(
    dataset: Path = typer.Argument(..., help="Path to dataset"),
    target: str = typer.Option(..., "--target", "-t", help="Target column name"),
) -> None:
    """Profile a dataset without training models."""
    import json

    import pandas as pd

    from forge.profiling.statistical_profiler import StatisticalProfiler
    from forge.profiling.semantic_profiler import SemanticProfiler

    df = pd.read_csv(dataset) if dataset.suffix == ".csv" else pd.read_parquet(dataset)
    profiler = StatisticalProfiler(target)
    report = profiler.profile(df)
    semantic = SemanticProfiler().profile(report, "", target)
    output = {"statistical": report.to_dict(), "semantic": semantic.to_dict()}
    console.print_json(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()
