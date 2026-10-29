"""Runtime limits for the public API, read from the environment.

The API can run on a shared machine and be reachable from the internet, where
anyone can upload a dataset and start a training job. These settings bound
what one visitor can cost the host. Defaults are safe for local development;
a deployment sets them in its service environment.

    FORGE_CORS_ORIGINS        comma-separated origins allowed to call the API
                              ("*" allows any; default for local development)
    FORGE_MAX_UPLOAD_MB       largest accepted dataset upload        (default 25)
    FORGE_MAX_TRIALS          upper bound on HPO trials per model     (default 30)
    FORGE_MAX_CONCURRENT_RUNS training jobs allowed at the same time  (default 1)
"""

from __future__ import annotations

import os


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def cors_origins() -> list[str]:
    raw = os.environ.get("FORGE_CORS_ORIGINS", "*")
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return origins or ["*"]


def max_upload_bytes() -> int:
    return _int_env("FORGE_MAX_UPLOAD_MB", 25) * 1024 * 1024


def max_trials() -> int:
    return _int_env("FORGE_MAX_TRIALS", 30)


def max_concurrent_runs() -> int:
    return _int_env("FORGE_MAX_CONCURRENT_RUNS", 1)
