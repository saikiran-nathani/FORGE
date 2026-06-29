"""Airflow batch prediction DAG template."""

from __future__ import annotations

from pathlib import Path

DAG_TEMPLATE = '''"""FORGE batch prediction DAG — generated automatically."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd

ARTIFACT_DIR = "{{ artifact_dir }}"
INPUT_PATH = "{{ input_path }}"
OUTPUT_PATH = "{{ output_path }}"


def score_batch():
    from forge.deployment.inference import ModelServer
    server = ModelServer(ARTIFACT_DIR)
    df = pd.read_csv(INPUT_PATH)
    records = df.to_dict(orient="records")
    predictions = server.predict(records)
    out = pd.DataFrame(predictions)
    out.to_csv(OUTPUT_PATH, index=False)


with DAG(
    dag_id="forge_batch_scoring",
    schedule_interval="{{ schedule }}",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:
    PythonOperator(
        task_id="score_batch",
        python_callable=score_batch,
    )
'''


class BatchPipelineGenerator:
    """Generates Airflow DAG for scheduled batch scoring."""

    def generate(
        self,
        deployment_dir: Path,
        input_path: str = "/data/input.csv",
        output_path: str = "/data/predictions.csv",
        schedule: str = "@daily",
    ) -> Path:
        dag_dir = deployment_dir / "airflow_dags"
        dag_dir.mkdir(parents=True, exist_ok=True)
        content = DAG_TEMPLATE.replace("{{ artifact_dir }}", str(deployment_dir / "artifacts"))
        content = content.replace("{{ input_path }}", input_path)
        content = content.replace("{{ output_path }}", output_path)
        content = content.replace("{{ schedule }}", schedule)
        dag_path = dag_dir / "forge_batch_scoring.py"
        dag_path.write_text(content)
        return dag_path
