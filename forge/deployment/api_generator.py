"""Generate standalone FastAPI serving application."""

from __future__ import annotations

from pathlib import Path

SERVING_APP_TEMPLATE = '''"""Auto-generated FORGE model serving API."""

from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model

from forge.deployment.inference import ModelServer

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
server = ModelServer(ARTIFACT_DIR)
info = server.model_info()

fields = {col: (Any, ...) for col in info.get("input_columns", [])}
PredictRequest = create_model("PredictRequest", **fields) if fields else BaseModel

class BatchRequest(BaseModel):
    records: list[dict[str, Any]]

app = FastAPI(title="FORGE Model API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": info.get("model_name")}


@app.get("/model-info")
def model_info():
    return info


@app.post("/predict")
def predict(body: dict[str, Any]):
    try:
        return server.predict(body)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/predict/batch")
def predict_batch(body: BatchRequest):
    try:
        return {"predictions": server.predict(body.records)}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''

DOCKERFILE_TEMPLATE = """FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
COPY forge/ forge/
RUN pip install --no-cache-dir .

COPY deployment/artifacts/ artifacts/
COPY deployment/serve.py .

EXPOSE 8080
CMD ["python", "serve.py"]
"""

REQUIREMENTS_PIN = """pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
joblib>=1.3
fastapi>=0.109
uvicorn[standard]>=0.27
"""


class APIGenerator:
    """Generates deployment files for a trained model."""

    def generate(self, deployment_dir: Path) -> list[str]:
        deployment_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = deployment_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        (deployment_dir / "serve.py").write_text(SERVING_APP_TEMPLATE)
        (deployment_dir / "Dockerfile").write_text(DOCKERFILE_TEMPLATE)
        (deployment_dir / "requirements.txt").write_text(REQUIREMENTS_PIN)

        compose = f"""services:
  model-api:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./artifacts:/app/artifacts:ro
"""
        (deployment_dir / "docker-compose.yml").write_text(compose)
        return ["serve.py", "Dockerfile", "requirements.txt", "docker-compose.yml"]
