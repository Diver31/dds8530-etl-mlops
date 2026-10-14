import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pendulum
from airflow.sdk import dag, task

ROOT = Path("/opt/project")
REPORTS = ROOT / "reports"


def read_audit(filename):
    return json.loads((REPORTS / filename).read_text())


def get_json(url):
    with urlopen(url, timeout=15) as response:
        return json.load(response)


@dag(
    dag_id="dds8530_pipeline_verification",
    description="Verify existing retail ETL and MLOps outputs",
    schedule=None,
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    tags=["DDS8530", "ETL", "MLOps", "validation"],
)
def pipeline_verification():

    @task
    def verify_source():
        profile = read_audit("source_profile.json")
        archive = ROOT / "data/raw/online_retail.zip"

        with archive.open("rb") as file:
            digest = hashlib.file_digest(
                file, "sha256"
            ).hexdigest()

        assert digest == profile["source_archive_sha256"]
        assert profile["row_count"] == profile["csv_row_count"]

        print("Source integrity: PASS")
        print("Source records:", profile["row_count"])

    @task
    def verify_etl():
        source = read_audit("source_profile.json")
        curation = read_audit(
            "transaction_curation_audit.json"
        )
        classification = read_audit(
            "merchandise_classification_audit.json"
        )
        weekly = read_audit(
            "weekly_aggregation_audit.json"
        )
        uk = read_audit("uk_weekly_audit.json")
        features = read_audit("ml_feature_audit.json")

        assert source["row_count"] == curation["input_rows"]
        assert curation["candidate_sales_rows"] == (
            classification["input_records"]
        )
        assert (
            classification["classification_counts"][
                "merchandise_candidate"
            ] == weekly["input_rows"]
        )
        assert uk["source_merchandise_rows"] == (
            weekly["input_rows"]
        )
        assert features["source_product_week_rows"] == (
            uk["uk_product_week_rows"]
        )
        assert features["validation"] == "PASS"

        print("ETL audit reconciliation: PASS")

    @task
    def verify_database_load_audit():
        weekly = read_audit(
            "weekly_aggregation_audit.json"
        )
        loaded = read_audit("postgres_load_audit.json")

        assert loaded["reconciliation"] == "PASS"
        assert loaded["records"] == (
            weekly["observed_product_week_rows"]
        )
        assert loaded["total_units"] == (
            weekly["complete_week_units"]
        )
        assert loaded["products"] == (
            weekly["distinct_products"]
        )

        print("Database load audit: PASS")
        print("Loaded records:", loaded["records"])

    @task
    def verify_model_artifact_audit():
        evaluation = read_audit(
            "final_model_evaluation.json"
        )
        artifact = read_audit(
            "model_artifact_validation.json"
        )

        assert evaluation["status"] == "PASS"
        assert artifact["model_reload"] == "PASS"
        assert artifact["prediction_reproducibility"] == "PASS"
        assert artifact["model_uri"] == (
            evaluation["final_model_uri"]
        )

        print("MLflow model artifact audit: PASS")

    @task
    def verify_live_services():
        api = "http://host.docker.internal:8000"
        health = get_json(f"{api}/health")
        model = get_json(f"{api}/model-info")
        evaluation = read_audit(
            "final_model_evaluation.json"
        )

        assert health["status"] == "ready"
        assert model["model_uri"] == (
            evaluation["final_model_uri"]
        )

        query = urlencode({
            "query": 'up{job="dds8530-flask-api"}'
        })

        result = get_json(
            "http://host.docker.internal:9090"
            f"/api/v1/query?{query}"
        )

        values = result["data"]["result"]

        assert values and values[0]["value"][1] == "1"

        print("Flask API: PASS")
        print("Prometheus scrape: UP")

    a = verify_source()
    b = verify_etl()
    c = verify_database_load_audit()
    d = verify_model_artifact_audit()
    e = verify_live_services()

    a >> b >> c >> d >> e


pipeline_verification()
