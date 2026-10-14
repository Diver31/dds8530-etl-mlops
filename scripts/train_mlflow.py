from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

SOURCE = Path("data/processed/uk_model_ready.csv")
AUDIT = Path("reports/ml_feature_audit.json")
OUTPUT = Path("reports/model_validation_audit.json")

audit = json.loads(AUDIT.read_text())

with SOURCE.open("rb") as file:
    data_hash = sha256(file.read()).hexdigest()

df = pd.read_csv(SOURCE, parse_dates=["week_start"])

if len(df) != audit["model_ready_rows"]:
    raise ValueError("Model row count mismatch.")

if df["split"].value_counts().to_dict() != {
    "train": audit["train_rows"],
    "test": audit["test_rows"],
}:
    raise ValueError("Training/test split mismatch.")

FEATURES = [
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_prev_4",
    "week_sin",
    "week_cos",
    "uk_holiday_week",
]

# Use development data only.
dev = df.loc[df["split"] == "train"].copy()

# Reserve the final eight development weeks for validation.
val_start = (
    pd.Timestamp(audit["train_last_week"])
    - pd.Timedelta(weeks=7)
)

train = dev.loc[dev["week_start"] < val_start].copy()
valid = dev.loc[dev["week_start"] >= val_start].copy()

if train.empty or valid.empty:
    raise ValueError("Empty training or validation partition.")

if valid["week_start"].nunique() != 8:
    raise ValueError("Expected eight validation weeks.")

if train["week_start"].max() >= valid["week_start"].min():
    raise ValueError("Training/validation overlap.")

if valid["week_start"].max() >= pd.Timestamp(
    audit["test_first_week"]
):
    raise ValueError("Validation overlaps final holdout.")

X_train = train[FEATURES].astype("float64")
X_valid = valid[FEATURES].astype("float64")
y_train = train["units_sold"].to_numpy(dtype="float64")
y_valid = valid["units_sold"].to_numpy(dtype="float64")

if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Invalid training features.")

if not np.isfinite(X_valid.to_numpy()).all():
    raise ValueError("Invalid validation features.")

if np.any(y_train < 0) or np.any(y_valid < 0):
    raise ValueError("Negative sales target.")

def metrics(actual, predicted):
    if not np.isfinite(predicted).all():
        raise ValueError("Non-finite predictions.")

    if np.any(predicted < 0):
        raise ValueError("Negative predicted quantities.")

    denominator = float(np.sum(actual))

    if denominator <= 0:
        raise ValueError("Cannot calculate WAPE.")

    return {
        "val_mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "val_rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "val_wape_pct": float(
            100 * np.sum(np.abs(actual - predicted))
            / denominator
        ),
    }

# Use a local database for experiment tracking.
tracking_db = Path("mlflow_tracking.db").resolve()
tracking_uri = f"sqlite:///{tracking_db.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("DDS8530_UK_Sales_Forecasting")

common_params = {
    "train_rows": len(train),
    "validation_rows": len(valid),
    "validation_start": str(val_start.date()),
    "validation_end": audit["train_last_week"],
    "forecast_type": "rolling_one_week_ahead",
}

# Experiment 1: previous-week baseline.
baseline_pred = X_valid["lag_1"].to_numpy()
baseline_metrics = metrics(y_valid, baseline_pred)

with mlflow.start_run(run_name="previous_week_baseline") as run:
    baseline_run = run.info.run_id
    mlflow.log_params({
        **common_params,
        "algorithm": "previous_week_sales",
    })
    mlflow.log_metrics(baseline_metrics)
    mlflow.set_tag("dataset_sha256", data_hash)

# Experiment 2: train a nonnegative regression model.
pipeline = Pipeline([
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

pipeline.fit(X_train, y_train)
model_pred = pipeline.predict(X_valid)
model_metrics = metrics(y_valid, model_pred)

with mlflow.start_run(run_name="poisson_gradient_boosting") as run:
    model_run = run.info.run_id

    mlflow.log_params({
        **common_params,
        "algorithm": "HistGradientBoostingRegressor",
        "loss": "poisson",
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "learning_rate": 0.07,
        "l2_regularization": 1.0,
        "normalization": "StandardScaler_training_only",
    })

    mlflow.log_metrics(model_metrics)
    mlflow.set_tag("dataset_sha256", data_hash)

    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="poisson_forecaster",
        input_example=X_train.head(3),
        skops_trusted_types=[
            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
        ],
    )

comparison = {
    "dataset_sha256": data_hash,
    "train_rows": len(train),
    "validation_rows": len(valid),
    "held_out_test_rows": audit["test_rows"],
    "validation_start": str(val_start.date()),
    "validation_end": audit["train_last_week"],
    "baseline": baseline_metrics,
    "gradient_boosting": model_metrics,
    "baseline_run_id": baseline_run,
    "model_run_id": model_run,
    "mlflow_model_uri": model_info.model_uri,
    "final_test_evaluated": False,
}

OUTPUT.write_text(json.dumps(comparison, indent=2))

print("\n=== MLFLOW MODEL VALIDATION ===")
print("Training rows:", len(train))
print("Validation rows:", len(valid))
print("Final test rows held out:", audit["test_rows"])
print("Validation start:", val_start.date())

print("\nPrevious-week baseline:")
print("MAE:", round(baseline_metrics["val_mae"], 3))
print("RMSE:", round(baseline_metrics["val_rmse"], 3))
print("WAPE:", round(baseline_metrics["val_wape_pct"], 2), "%")

print("\nPoisson gradient boosting:")
print("MAE:", round(model_metrics["val_mae"], 3))
print("RMSE:", round(model_metrics["val_rmse"], 3))
print("WAPE:", round(model_metrics["val_wape_pct"], 2), "%")

print("\nMLflow experiment: DDS8530_UK_Sales_Forecasting")
print("Baseline run:", baseline_run)
print("Model run:", model_run)
print("Saved model URI:", model_info.model_uri)
print("Final test set untouched: YES")
print("Validation audit:", OUTPUT)
print("Experiment tracking: COMPLETE")
