from pathlib import Path
from hashlib import sha256
import json
import os

import numpy as np
import pandas as pd

# Permit loading only our own locally generated model.
# Serialized models must never be loaded from untrusted sources.
os.environ["MLFLOW_ALLOW_PICKLE_DESERIALIZATION"] = "true"

import mlflow
import mlflow.sklearn

ROOT = Path.cwd()
AUDIT = ROOT / "reports/final_model_evaluation.json"
SOURCE = ROOT / "data/processed/uk_model_ready.csv"
PREDICTIONS = ROOT / "data/processed/final_test_predictions.csv"
OUTPUT = ROOT / "reports/model_artifact_validation.json"

result = json.loads(AUDIT.read_text())
model_uri = result["final_model_uri"]

if not model_uri.startswith("models:/m-"):
    raise ValueError("Unexpected MLflow model URI.")

with SOURCE.open("rb") as file:
    actual_hash = sha256(file.read()).hexdigest()

if actual_hash != result["dataset_sha256"]:
    raise ValueError("Model input dataset fingerprint mismatch.")

mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow_tracking.db').as_posix()}"
)

print("Loading saved MLflow model...")
model = mlflow.sklearn.load_model(model_uri)

FEATURES = [
    "lag_1", "lag_2", "lag_4",
    "rolling_prev_4", "week_sin",
    "week_cos", "uk_holiday_week",
]

dataset = pd.read_csv(SOURCE, dtype={"StockCode": "string"})
test = dataset.loc[
    dataset["split"] == "test"
].reset_index(drop=True)

saved = pd.read_csv(
    PREDICTIONS, dtype={"StockCode": "string"}
)

if len(test) != result["test_rows"] or len(saved) != len(test):
    raise RuntimeError("Holdout row counts do not match.")

if not np.array_equal(
    test["StockCode"].to_numpy(),
    saved["StockCode"].to_numpy(),
):
    raise RuntimeError("Product identifiers do not match.")

if not np.array_equal(
    test["week_start"].to_numpy(),
    saved["week_start"].to_numpy(),
):
    raise RuntimeError("Forecast dates do not match.")

# Sample 25 different positions from the preserved holdout.
positions = np.linspace(
    0, len(test) - 1, 25, dtype=int
)

inputs = test.iloc[positions][FEATURES].astype("float64")
original = saved.iloc[positions]["model_prediction"].to_numpy()

reloaded = model.predict(inputs)

if not np.isfinite(reloaded).all():
    raise RuntimeError("Model returned invalid predictions.")

if not np.allclose(
    reloaded, original, rtol=1e-6, atol=1e-6
):
    raise RuntimeError("Reloaded predictions do not match.")

maximum_difference = float(
    np.max(np.abs(reloaded - original))
)

verification = {
    "status": "PASS",
    "model_uri": model_uri,
    "model_run_id": result["model_run_id"],
    "test_rows_verified_for_alignment": len(test),
    "sample_predictions_compared": len(positions),
    "maximum_absolute_prediction_difference": maximum_difference,
    "dataset_fingerprint": "PASS",
    "model_reload": "PASS",
    "prediction_reproducibility": "PASS",
}

OUTPUT.write_text(json.dumps(verification, indent=2))

print("\n=== MODEL ARTIFACT VALIDATION ===")
print("Model loaded: PASS")
print("Model URI:", model_uri)
print("Holdout row alignment:", len(test), "PASS")
print("Predictions compared:", len(positions))
print("Maximum prediction difference:", maximum_difference)
print("Reproducibility: PASS")
print("Audit:", OUTPUT)
