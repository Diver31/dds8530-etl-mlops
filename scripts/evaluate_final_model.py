from pathlib import Path
from hashlib import sha256
import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = Path("data/processed/uk_model_ready.csv")
FEATURE_AUDIT = Path("reports/ml_feature_audit.json")
VALIDATION_AUDIT = Path("reports/model_validation_audit.json")
OUTPUT = Path("reports/final_model_evaluation.json")
PREDICTIONS = Path("data/processed/final_test_predictions.csv")

# Prevent accidental repeated use of the final holdout.
if OUTPUT.exists():
    raise RuntimeError(
        "A final evaluation already exists. "
        "Review it before running another test."
    )

features_audit = json.loads(FEATURE_AUDIT.read_text())
validation = json.loads(VALIDATION_AUDIT.read_text())

with DATA.open("rb") as file:
    fingerprint = sha256(file.read()).hexdigest()

if fingerprint != validation["dataset_sha256"]:
    raise ValueError("Dataset changed since model validation.")

if validation["final_test_evaluated"] is not False:
    raise ValueError("Final test status is unexpected.")

df = pd.read_csv(DATA, parse_dates=["week_start"])

FEATURES = [
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_prev_4",
    "week_sin",
    "week_cos",
    "uk_holiday_week",
]

development = df.loc[df["split"] == "train"].copy()
test = df.loc[df["split"] == "test"].copy()

if len(development) != features_audit["train_rows"]:
    raise ValueError("Development count mismatch.")

if len(test) != features_audit["test_rows"]:
    raise ValueError("Holdout count mismatch.")

if len(df) != features_audit["model_ready_rows"]:
    raise ValueError("Total modeling rows mismatch.")

if development.empty or test.empty:
    raise ValueError("Empty modeling partition.")

if development["week_start"].max() >= test["week_start"].min():
    raise ValueError("Chronological leakage detected.")

if test["week_start"].nunique() != 12:
    raise ValueError("Expected twelve holdout weeks.")

if str(test["week_start"].min().date()) != features_audit["test_first_week"]:
    raise ValueError("Unexpected holdout start date.")

if str(test["week_start"].max().date()) != features_audit["test_last_week"]:
    raise ValueError("Unexpected holdout end date.")

X_train = development[FEATURES].astype("float64")
y_train = development["units_sold"].to_numpy(dtype="float64")

X_test = test[FEATURES].astype("float64")
y_test = test["units_sold"].to_numpy(dtype="float64")

if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Invalid development features.")

if not np.isfinite(X_test.to_numpy()).all():
    raise ValueError("Invalid holdout features.")

if (y_train < 0).any() or (y_test < 0).any():
    raise ValueError("Negative sales quantities.")

def calculate_metrics(actual, predicted):
    predicted = np.asarray(predicted, dtype="float64")

    if not np.isfinite(predicted).all():
        raise ValueError("Non-finite predictions.")

    if (predicted < 0).any():
        raise ValueError("Negative predictions.")

    total_actual = float(actual.sum())

    if total_actual <= 0:
        raise ValueError("Invalid WAPE denominator.")

    return {
        "test_mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "test_rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "test_wape_pct": float(
            100 * np.abs(actual - predicted).sum()
            / total_actual
        ),
    }

# Baseline: assume next week equals the previous week.
baseline_predictions = X_test["lag_1"].to_numpy()
baseline_metrics = calculate_metrics(
    y_test, baseline_predictions
)

# Refit the validation-selected configuration on all
# development weeks. Do not retune using the test set.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.07,
        l2_regularization=1.0,
        random_state=42,
    )),
])

model.fit(X_train, y_train)
model_predictions = model.predict(X_test)

model_metrics = calculate_metrics(
    y_test, model_predictions
)

# Save row-level predictions for independent validation.
predictions = test[
    ["StockCode", "week_start", "units_sold"]
].copy()

predictions = predictions.rename(
    columns={"units_sold": "actual_units"}
)

predictions["baseline_prediction"] = baseline_predictions
predictions["model_prediction"] = model_predictions

predictions["baseline_abs_error"] = np.abs(
    y_test - baseline_predictions
)

predictions["model_abs_error"] = np.abs(
    y_test - model_predictions
)

predictions.to_csv(
    PREDICTIONS,
    index=False,
    date_format="%Y-%m-%d",
)

# Continue recording experiments in the existing MLflow DB.
tracking_db = Path("mlflow_tracking.db").resolve()

mlflow.set_tracking_uri(
    f"sqlite:///{tracking_db.as_posix()}"
)

mlflow.set_experiment(
    "DDS8530_UK_Sales_Forecasting"
)

common = {
    "development_rows": len(development),
    "holdout_rows": len(test),
    "holdout_weeks": 12,
    "evaluation_type": "rolling_one_week_ahead",
}

with mlflow.start_run(
    run_name="final_previous_week_baseline"
) as run:
    baseline_run_id = run.info.run_id

    mlflow.log_params({
        **common,
        "algorithm": "previous_week_sales",
    })

    mlflow.log_metrics(baseline_metrics)
    mlflow.set_tag("dataset_sha256", fingerprint)
    mlflow.set_tag("evaluation_phase", "final_holdout")

with mlflow.start_run(
    run_name="final_poisson_gradient_boosting"
) as run:
    model_run_id = run.info.run_id

    mlflow.log_params({
        **common,
        "algorithm": "HistGradientBoostingRegressor",
        "loss": "poisson",
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "learning_rate": 0.07,
        "l2_regularization": 1.0,
        "normalization": "StandardScaler_training_only",
    })

    mlflow.log_metrics(model_metrics)
    mlflow.set_tag("dataset_sha256", fingerprint)
    mlflow.set_tag("evaluation_phase", "final_holdout")

    saved = mlflow.sklearn.log_model(
        sk_model=model,
        name="uk_sales_forecaster_final",
        input_example=X_train.head(3),
        skops_trusted_types=[
            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
        ],
    )

    result = {
        "status": "PASS",
        "dataset_sha256": fingerprint,
        "development_rows": len(development),
        "test_rows": len(test),
        "test_weeks": 12,
        "baseline": baseline_metrics,
        "gradient_boosting": model_metrics,
        "baseline_run_id": baseline_run_id,
        "model_run_id": model_run_id,
        "final_model_uri": saved.model_uri,
        "model_configuration": "Fixed using validation results",
        "test_evaluated": True,
        "forecast_assumption": (
            "Previous week's actual sales available "
            "for each one-week-ahead prediction"
        ),
    }

    OUTPUT.write_text(json.dumps(result, indent=2))

    mlflow.log_artifact(
        str(OUTPUT), artifact_path="evaluation"
    )
    mlflow.log_artifact(
        str(PREDICTIONS), artifact_path="evaluation"
    )

print("\n=== FINAL HOLDOUT EVALUATION ===")
print("Development observations:", len(development))
print("Test observations:", len(test))
print("Test weeks:", test["week_start"].nunique())

print("\nPrevious-week baseline:")
for key, value in baseline_metrics.items():
    print(f"{key}: {value:.3f}")

print("\nGradient boosting:")
for key, value in model_metrics.items():
    print(f"{key}: {value:.3f}")

mae_change = (
    100
    * (baseline_metrics["test_mae"] - model_metrics["test_mae"])
    / baseline_metrics["test_mae"]
)

print(f"\nMAE improvement: {mae_change:.2f}%")
print("Final model URI:", saved.model_uri)
print("MLflow model run:", model_run_id)
print("Predictions:", PREDICTIONS)
print("Audit:", OUTPUT)
print("Final evaluation: COMPLETE")
