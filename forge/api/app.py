"""FORGE FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from forge.api import settings
from forge.api.demo_loader import load_demo_seed
from forge.api.schemas import ExperimentCreate, ExperimentResponse, ExperimentStatusResponse
from forge.api.services.deployment_service import deployment_service
from forge.api.services.experiment_service import ExperimentStatus, store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Seed a pre-trained demo experiment on startup (no-op if none is committed).
    load_demo_seed()
    yield


app = FastAPI(
    title="FORGE API",
    description="LLM-powered automated ML pipeline",
    version="0.2.0",
    lifespan=lifespan,
)

# Only the configured frontend may call the API from a browser (FORGE_CORS_ORIGINS).
# No cookies or auth headers are used, so credentials stay off — a wildcard origin
# with credentials is also rejected by browsers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "version": "0.2.0"}


@app.post("/api/v1/experiments", response_model=ExperimentResponse)
async def create_experiment(
    file: UploadFile = File(...),
    name: str = Form("Untitled Experiment"),
    target_column: str = Form(...),
    task_description: str = Form(""),
    trials: int = Form(10),
    fast_mode: bool = Form(True),
):
    if not file.filename:
        raise HTTPException(400, "No file uploaded")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".csv", ".parquet", ".json"):
        raise HTTPException(400, "Supported formats: CSV, Parquet, JSON")

    max_trials = settings.max_trials()
    if not 1 <= trials <= max_trials:
        raise HTTPException(422, f"trials must be between 1 and {max_trials}")

    # One training job at a time (FORGE_MAX_CONCURRENT_RUNS): the host is shared.
    exp = store.create_if_capacity(
        name, target_column, task_description, Path("pending"),
        max_active=settings.max_concurrent_runs(),
    )
    if exp is None:
        raise HTTPException(
            429, "The server is already training a model. Try again in a few minutes.",
            headers={"Retry-After": "120"},
        )

    # Stream the upload to disk in chunks and stop at the size cap, instead of
    # reading an arbitrarily large body into memory first.
    limit = settings.max_upload_bytes()
    dataset_path = store.base_dir / exp.id / f"dataset{suffix}"
    written = 0
    with dataset_path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                out.close()
                store.discard(exp.id)
                raise HTTPException(413, f"Dataset larger than {limit // (1024 * 1024)} MB")
            out.write(chunk)
    exp.dataset_path = dataset_path

    store.run_async(exp, trials=trials, fast_mode=fast_mode)
    return _to_response(exp)


@app.get("/api/v1/experiments", response_model=list[ExperimentResponse])
def list_experiments():
    return [_to_response(e) for e in store.list_all()]


@app.get("/api/v1/experiments/{exp_id}", response_model=ExperimentResponse)
def get_experiment(exp_id: str):
    exp = store.get(exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return _to_response(exp)


@app.get("/api/v1/experiments/{exp_id}/status", response_model=ExperimentStatusResponse)
def get_status(exp_id: str):
    exp = store.get(exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return ExperimentStatusResponse(
        id=exp.id,
        status=exp.status.value,
        progress=exp.progress,
        stage=exp.stage,
        progress_log=exp.progress_log,
        error=exp.error,
    )


@app.get("/api/v1/experiments/{exp_id}/profile")
def get_profile(exp_id: str):
    exp = store.get(exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    if exp.status != ExperimentStatus.COMPLETED:
        raise HTTPException(400, "Experiment not completed")
    return {
        "profile": exp.result.get("semantic_profile", {}),
        "quality_score": exp.result.get("quality_score"),
    }


@app.get("/api/v1/experiments/{exp_id}/models")
def get_models(exp_id: str):
    exp = store.get(exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return {
        "best_model": exp.result.get("best_model_name"),
        "models": exp.result.get("model_results", []),
        "metrics": exp.result.get("best_metrics", {}),
    }


@app.get("/api/v1/experiments/{exp_id}/evaluation")
def get_evaluation(exp_id: str):
    exp = store.get(exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return {
        "metrics": exp.result.get("best_metrics", {}),
        "shap": exp.result.get("shap_summary", {}),
        "generated_features": exp.result.get("generated_features", []),
    }


@app.get("/api/v1/experiments/{exp_id}/report")
def get_report(exp_id: str, format: str = "eda"):
    exp = store.get(exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    if format == "analysis":
        path = exp.result.get("llm_report")
        if path and Path(path).exists():
            return FileResponse(path, media_type="text/markdown")
        raise HTTPException(404, "Analysis report not ready")
    eda = exp.result.get("eda_report")
    if eda and Path(eda).exists():
        return FileResponse(eda, media_type="text/html")
    raise HTTPException(404, "Report not ready")


@app.get("/api/v1/experiments/{exp_id}/errors")
def get_errors(exp_id: str):
    exp = store.get(exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return exp.result.get("error_analysis", {})


@app.post("/api/v1/experiments/{exp_id}/deploy")
def deploy_experiment(exp_id: str):
    exp = store.get(exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    if exp.status != ExperimentStatus.COMPLETED:
        raise HTTPException(400, "Experiment must be completed before deployment")
    artifact_dir = Path(exp.result.get("artifact_dir", ""))
    if not artifact_dir.exists():
        raise HTTPException(400, "Artifacts not found")
    try:
        result = deployment_service.deploy(
            exp_id, artifact_dir, exp.task_description, exp.target_column
        )
        exp.result["deployment"] = result
        return result
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/v1/experiments/{exp_id}/predict")
def predict(exp_id: str, body: dict):
    if not deployment_service.is_deployed(exp_id):
        raise HTTPException(400, "Model not deployed. POST /deploy first.")
    try:
        return deployment_service.predict(exp_id, body)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/v1/experiments/{exp_id}/monitoring")
def get_monitoring(exp_id: str):
    exp = store.get(exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return deployment_service.get_monitoring(exp_id)


@app.get("/api/v1/experiments/{exp_id}/model-info")
def get_model_info(exp_id: str):
    if not deployment_service.is_deployed(exp_id):
        raise HTTPException(400, "Model not deployed. POST /deploy first.")
    try:
        return deployment_service.get_model_info(exp_id)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/v1/experiments/{exp_id}/model-card")
def get_model_card(exp_id: str):
    exp = store.get(exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    deployment = exp.result.get("deployment", {})
    card = deployment.get("model_card")
    if card and Path(card).exists():
        return FileResponse(card, media_type="text/markdown")
    artifact = Path(exp.result.get("artifact_dir", "")) / "model_card.md"
    if artifact.exists():
        return FileResponse(artifact, media_type="text/markdown")
    raise HTTPException(404, "Model card not found")


def _to_response(exp) -> ExperimentResponse:
    return ExperimentResponse(
        id=exp.id,
        name=exp.name,
        target_column=exp.target_column,
        task_description=exp.task_description,
        status=exp.status.value,
        created_at=exp.created_at,
        progress=exp.progress,
        stage=exp.stage,
        progress_log=exp.progress_log,
        error=exp.error,
        result=exp.result,
    )


def create_app() -> FastAPI:
    return app
