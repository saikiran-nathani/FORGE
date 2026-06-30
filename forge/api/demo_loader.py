"""Load the committed demo seed into the API at startup.

When ``forge/api/demo/`` contains a prebuilt seed (see scripts/build_demo_seed.py),
this registers it as a COMPLETED experiment and deploys its model, so the live
app shows a real, pre-trained experiment instantly and the prediction Playground
works without anyone having to train a model. No-op if the seed is absent.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from forge.api.services.deployment_service import deployment_service
from forge.api.services.experiment_service import Experiment, ExperimentStatus, store

DEMO_DIR = Path(__file__).resolve().parent / "demo"


def load_demo_seed() -> bool:
    """Returns True if a demo seed was loaded."""
    result_path = DEMO_DIR / "result.json"
    meta_path = DEMO_DIR / "meta.json"
    artifacts = DEMO_DIR / "artifacts"
    if not (result_path.exists() and meta_path.exists() and artifacts.exists()):
        return False

    meta = json.loads(meta_path.read_text())
    exp_id = meta.get("experiment_id", "demo")
    if store.get(exp_id):  # already loaded (e.g. reload)
        return True

    result = json.loads(result_path.read_text())
    # Recompute path-typed fields against the committed location (ignore stored paths).
    result["artifact_dir"] = str(artifacts)
    eda = artifacts / "eda_report.html"
    analysis = artifacts / "analysis_report.md"
    result["eda_report"] = str(eda) if eda.exists() else ""
    result["llm_report"] = str(analysis) if analysis.exists() else ""

    exp = Experiment(
        id=exp_id,
        name=meta.get("name", "Demo experiment"),
        task_description=meta.get("task_description", ""),
        target_column=meta.get("target_column", "target"),
        status=ExperimentStatus.COMPLETED,
        created_at=meta.get("created_at", datetime.now(timezone.utc).isoformat()),
        dataset_path=DEMO_DIR / "demo_dataset.csv",
        output_dir=artifacts,
        result=result,
        progress="Complete",
    )
    store.add(exp)

    # Deploy the model so the Playground (/predict, /model-info) works immediately.
    try:
        deployment_service.deploy(exp_id, artifacts, exp.task_description, exp.target_column)
        exp.result["deployment"] = {"status": "deployed", "demo": True}
    except Exception as exc:  # demo still shows results even if predict is unavailable
        print(f"[demo] auto-deploy failed: {exc}")

    print(f"[demo] loaded seeded experiment '{exp_id}' ({exp.result.get('best_model_name')})")
    return True
