import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pendulum
from airflow.sdk import dag, task, get_current_context

PROJECT = Path("/opt/project")
RUNS = PROJECT / "airflow_runs"
PYTHON = "/home/airflow/dds8530-venv/bin/python"

@dag(
    dag_id="dds8530_etl_execution",
    description="Execute Dask ETL and PostgreSQL refresh",
    schedule="0 3 * * 1",
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["DDS8530", "ETL", "Dask", "PostgreSQL"],
)
def execute_etl():

    @task
    def prepare_workspace():
        context = get_current_context()
        token = hashlib.sha256(
            context["run_id"].encode()
        ).hexdigest()[:16]

        root = RUNS / token
        data_dir = root / "data/processed"
        reports_dir = root / "reports"

        data_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        source = (
            PROJECT / "data/processed/merchandise_sales.csv"
        )
        link = data_dir / "merchandise_sales.csv"

        if not source.is_file():
            raise FileNotFoundError(source)

        if not link.is_symlink():
            if link.exists():
                raise RuntimeError(
                    "Unexpected file in execution workspace."
                )
            link.symlink_to(source)

        if link.resolve() != source.resolve():
            raise RuntimeError("Source link mismatch.")

        for name in (
            "source_profile.json",
            "merchandise_classification_audit.json",
        ):
            shutil.copy2(
                PROJECT / "reports" / name,
                reports_dir / name,
            )

        print("Workspace:", root)
        print("Original source preserved: YES")
        print("Workspace preparation: PASS")
        return str(root)

    @task
    def aggregate_with_dask(workspace):
        result = subprocess.run(
            [
                PYTHON,
                str(PROJECT / "scripts/aggregate_weekly.py"),
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=600,
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError("Dask transformation failed.")

        audit = json.loads(
            (Path(workspace) /
             "reports/weekly_aggregation_audit.json").read_text()
        )

        original = json.loads(
            (PROJECT /
             "reports/weekly_aggregation_audit.json").read_text()
        )

        for field in (
            "input_rows",
            "distinct_products",
            "observed_product_week_rows",
            "complete_week_units",
        ):
            if audit[field] != original[field]:
                raise RuntimeError(
                    f"Dask reconciliation failed: {field}"
                )

        print("Original vs Airflow aggregation: PASS")
        return workspace

    @task
    def load_postgresql(workspace):
        result = subprocess.run(
            [
                PYTHON,
                str(PROJECT /
                    "scripts/airflow_load_weekly.py"),
                workspace,
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=600,
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError("PostgreSQL load failed.")

        return workspace

    @task
    def verify_committed_database(workspace):
        result = subprocess.run(
            [
                PYTHON,
                str(PROJECT /
                    "scripts/airflow_load_weekly.py"),
                workspace,
                "--verify-only",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=120,
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                "Independent database validation failed."
            )

        print("AIRFLOW ETL EXECUTION: PASS")


    @task
    def prepare_model_inputs(workspace):
        root = Path(workspace)

        inputs = [
            (
                PROJECT / "data/staging/uk_holidays_2010_2011.csv",
                root / "data/staging/uk_holidays_2010_2011.csv",
            ),
            (
                PROJECT / "reports/model_source_assessment.json",
                root / "reports/model_source_assessment.json",
            ),
        ]

        for source, destination in inputs:
            if not source.is_file():
                raise FileNotFoundError(source)

            destination.parent.mkdir(
                parents=True, exist_ok=True
            )

            if destination.exists():
                if destination.read_bytes() != source.read_bytes():
                    raise RuntimeError(
                        "Existing input differs from validated source."
                    )
            else:
                shutil.copy2(source, destination)

        print("Model inputs prepared: PASS")
        return workspace

    @task
    def aggregate_uk_sales(workspace):
        result = subprocess.run(
            [
                PYTHON,
                str(PROJECT / "scripts/build_uk_weekly.py"),
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=900,
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError("UK aggregation failed.")

        audit = json.loads(
            (
                Path(workspace) / "reports/uk_weekly_audit.json"
            ).read_text()
        )

        reference = json.loads(
            (
                PROJECT / "reports/uk_weekly_audit.json"
            ).read_text()
        )

        for key in (
            "uk_transaction_rows",
            "uk_product_week_rows",
            "uk_units_complete_weeks",
        ):
            if audit[key] != reference[key]:
                raise RuntimeError(
                    f"UK aggregation mismatch: {key}"
                )

        print("UK aggregation reconciliation: PASS")
        return workspace

    @task
    def engineer_forecasting_features(workspace):
        result = subprocess.run(
            [
                PYTHON,
                str(PROJECT / "scripts/build_ml_features.py"),
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=900,
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError("Feature engineering failed.")

        audit = json.loads(
            (
                Path(workspace) / "reports/ml_feature_audit.json"
            ).read_text()
        )

        reference = json.loads(
            (
                PROJECT / "reports/ml_feature_audit.json"
            ).read_text()
        )

        for key in (
            "eligible_products",
            "model_ready_rows",
            "train_rows",
            "test_rows",
        ):
            if audit[key] != reference[key]:
                raise RuntimeError(
                    f"Feature reconciliation mismatch: {key}"
                )

        print("Forecasting features: PASS")
        return workspace

    @task
    def train_with_mlflow(workspace):
        result = subprocess.run(
            [
                PYTHON,
                str(PROJECT / "scripts/train_mlflow.py"),
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=900,
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError("MLflow training failed.")

        audit = json.loads(
            (
                Path(workspace)
                / "reports/model_validation_audit.json"
            ).read_text()
        )

        if audit["held_out_test_rows"] != 19296:
            raise RuntimeError("Unexpected holdout size.")

        if audit["final_test_evaluated"] is not False:
            raise RuntimeError("Final holdout was evaluated.")

        print("MLflow experiment tracking: PASS")
        return workspace

    @task
    def verify_automated_model(workspace):
        result = subprocess.run(
            [
                PYTHON,
                str(
                    PROJECT /
                    "scripts/verify_airflow_mlflow.py"
                ),
                workspace,
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=300,
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                "Automated model verification failed."
            )

        print("AIRFLOW MLOPS EXECUTION: PASS")

    workspace = prepare_workspace()
    aggregated = aggregate_with_dask(workspace)
    loaded = load_postgresql(aggregated)

    verified = verify_committed_database(loaded)
    prepared = prepare_model_inputs(loaded)

    # Model processing cannot begin until SQL validation passes.
    verified >> prepared

    uk = aggregate_uk_sales(prepared)
    featured = engineer_forecasting_features(uk)
    trained = train_with_mlflow(featured)
    verify_automated_model(trained)


execute_etl()
