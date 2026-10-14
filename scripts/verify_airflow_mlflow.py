from pathlib import Path
from hashlib import sha256
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# Load only the model produced by this trusted local pipeline.
os.environ["MLFLOW_ALLOW_PICKLE_DESERIALIZATION"] = "true"

import mlflow
import mlflow.sklearn

root = Path(sys.argv[1]).resolve()
runs = Path("/opt/project/airflow_runs").resolve()

if root.parent != runs:
    raise RuntimeError("Unexpected execution workspace.")

reports = root / "reports"
audit = json.loads(
    (reports / "model_validation_audit.json").read_text()
)
features = json.loads(
    (reports / "ml_feature_audit.json").read_text()
)

reference = json.loads(
    Path("/opt/project/reports/ml_feature_audit.json").read_text()
)

for key in (
    "eligible_products",
    "train_rows",
    "test_rows",
    "model_ready_rows",
):
    if features[key] != reference[key]:
        raise RuntimeError(f"Feature mismatch: {key}")

if audit["held_out_test_rows"] != reference["test_rows"]:
    raise RuntimeError("Final holdout count mismatch.")

if (reports / "final_model_evaluation.json").exists():
    raise RuntimeError(
        "Final holdout evaluation unexpectedly executed."
    )

dataset = root / "data/processed/uk_model_ready.csv"

with dataset.open("rb") as file:
    digest = sha256(file.read()).hexdigest()

if digest != audit["dataset_sha256"]:
    raise RuntimeError("Dataset fingerprint mismatch.")

model_uri = audit["mlflow_model_uri"]

if not re.fullmatch(r"models:/m-[0-9a-f]{32}", model_uri):
    raise RuntimeError("Unexpected MLflow model identifier.")

tracking_db = root / "mlflow_tracking.db"

if not tracking_db.is_file():
    raise RuntimeError("MLflow tracking database missing.")

mlflow.set_tracking_uri(
    f"sqlite:///{tracking_db.as_posix()}"
)

model = mlflow.sklearn.load_model(model_uri)

columns = [
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_prev_4",
    "week_sin",
    "week_cos",
    "uk_holiday_week",
]

df = pd.read_csv(dataset)
development = df.loc[df["split"] == "train"]

sample = development[columns].head(5).astype("float64")
predictions = model.predict(sample)

if len(predictions) != 5:
    raise RuntimeError("Incorrect prediction count.")

if not np.isfinite(predictions).all():
    raise RuntimeError("Invalid predictions.")

if (predictions < 0).any():
    raise RuntimeError("Negative forecast quantities.")

result = {
    "status": "PASS",
    "workspace": root.name,
    "eligible_products": features["eligible_products"],
    "development_rows": features["train_rows"],
    "holdout_rows_preserved": features["test_rows"],
    "mlflow_model_uri": model_uri,
    "mlflow_model_run": audit["model_run_id"],
    "model_reload": "PASS",
    "prediction_check": "PASS",
    "final_holdout_retested": False,
}

output = reports / "airflow_mlops_validation.json"
output.write_text(json.dumps(result, indent=2))

print("\n=== AUTOMATED MLOPS VALIDATION ===")
print("Workspace:", root.name)
print("Eligible products:", result["eligible_products"])
print("Development rows:", result["development_rows"])
print("Holdout rows preserved:", result["holdout_rows_preserved"])
print("MLflow model:", model_uri)
print("Model reload: PASS")
print("Prediction check: PASS")
print("Final holdout not rerun: PASS")
print("Audit:", output)
print("AUTOMATED MLOPS VALIDATION: PASS")
