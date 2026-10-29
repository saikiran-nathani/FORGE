"""The public-API limits: trial cap, one job at a time, upload size cap, CORS parsing."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from forge.api import settings
from forge.api.app import app
from forge.api.services import experiment_service
from forge.api.services.experiment_service import ExperimentStatus

CSV = b"a,b,y\n1,2,0\n3,4,1\n5,6,0\n"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    store = experiment_service.store
    monkeypatch.setattr(store, "base_dir", tmp_path)
    monkeypatch.setattr(store, "_experiments", {})
    started = []
    # Record the job instead of training a model.
    monkeypatch.setattr(store, "run_async", lambda exp, **kw: started.append((exp, kw)))
    c = TestClient(app)  # no `with`: skips the lifespan, so no demo seed is loaded
    c.started = started
    return c


def _upload(client, body=CSV, trials=5):
    return client.post(
        "/api/v1/experiments",
        files={"file": ("data.csv", io.BytesIO(body), "text/csv")},
        data={"target_column": "y", "trials": str(trials)},
    )


def test_valid_upload_starts_one_job(client, tmp_path):
    r = _upload(client)
    assert r.status_code == 200, r.text
    exp_id = r.json()["id"]
    assert (tmp_path / exp_id / "dataset.csv").read_bytes() == CSV
    assert len(client.started) == 1 and client.started[0][1]["trials"] == 5


def test_trials_above_cap_rejected(client, monkeypatch):
    monkeypatch.setenv("FORGE_MAX_TRIALS", "10")
    assert _upload(client, trials=11).status_code == 422
    assert _upload(client, trials=0).status_code == 422
    assert client.started == []


def test_second_job_while_one_is_active_gets_429(client):
    assert _upload(client).status_code == 200  # stays PENDING: run_async is stubbed
    r = _upload(client)
    assert r.status_code == 429
    assert r.headers.get("retry-after") == "120"
    assert len(client.started) == 1


def test_capacity_frees_up_when_the_job_finishes(client):
    first = _upload(client).json()["id"]
    experiment_service.store.get(first).status = ExperimentStatus.COMPLETED
    assert _upload(client).status_code == 200


def test_oversized_upload_rejected_and_cleaned_up(client, monkeypatch, tmp_path):
    monkeypatch.setenv("FORGE_MAX_UPLOAD_MB", "1")
    big = b"a,b,y\n" + b"1,2,0\n" * 200_000  # ~1.2 MB
    r = _upload(client, body=big)
    assert r.status_code == 413
    assert client.started == []
    assert experiment_service.store.list_all() == []  # rejected job does not hold the slot
    assert list(tmp_path.iterdir()) == []              # and leaves nothing on disk
    assert _upload(client).status_code == 200


def test_cors_origins_parsing(monkeypatch):
    monkeypatch.delenv("FORGE_CORS_ORIGINS", raising=False)
    assert settings.cors_origins() == ["*"]
    monkeypatch.setenv("FORGE_CORS_ORIGINS", " https://a.netlify.app/ , http://localhost:5173 ")
    assert settings.cors_origins() == ["https://a.netlify.app", "http://localhost:5173"]


def test_bad_limit_value_fails_loudly(monkeypatch):
    monkeypatch.setenv("FORGE_MAX_UPLOAD_MB", "lots")
    with pytest.raises(ValueError, match="FORGE_MAX_UPLOAD_MB"):
        settings.max_upload_bytes()
    monkeypatch.setenv("FORGE_MAX_CONCURRENT_RUNS", "0")
    with pytest.raises(ValueError, match=">= 1"):
        settings.max_concurrent_runs()
