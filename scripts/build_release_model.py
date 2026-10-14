"""Create a versioned, approved-configuration serving artifact.

Re-trains the configuration fixed during the historical development-validation
phase on the original 40 development weeks. Never uses final-test outcomes to
fit, tune, approve, or select the model. Outputs a trusted build artifact, not
an export of the earlier, machine-local MLflow model URI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "assets/uk_model_ready.csv"
OUTPUT = ROOT / "model_release"
EXPECTED_DATA_SHA256 = "b6c5959b1dd208548fb69f2b62732e0ec0728414f0d91c08eebecbd3718d1b96"
FEATURES = ["lag_1", "lag_2", "lag_4", "rolling_prev_4", "week_sin", "week_cos", "uk_holiday_week"]
PARAMETERS = dict(loss="poisson", max_iter=100, max_leaf_nodes=15,
                  learning_rate=0.07, l2_regularization=1.0, random_state=42)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    if not INPUT.is_file():
        raise RuntimeError("Derived, attributed model-ready dataset is missing")
    checksum = digest(INPUT)
    if checksum != EXPECTED_DATA_SHA256:
        raise RuntimeError("Input differs from the validated historical modeling extract")

    frame = pd.read_csv(INPUT, dtype={"StockCode": "string"}, parse_dates=["week_start"])
    assert len(frame) == 75875, "Unexpected model panel length"
    assert list(sorted(frame["split"].unique())) == ["test", "train"]
    develop = frame.loc[frame["split"] == "train"].copy()
    holdout = frame.loc[frame["split"] == "test"].copy()
    assert len(develop) == 56579 and len(holdout) == 19296
    assert develop["week_start"].max() < holdout["week_start"].min()
    assert develop["week_start"].max().strftime("%Y-%m-%d") == "2011-09-05"
    assert holdout["week_start"].min().strftime("%Y-%m-%d") == "2011-09-12"
    assert holdout["week_start"].nunique() == 12
    assert len(set(FEATURES) - set(develop.columns)) == 0
    assert develop[FEATURES + ["units_sold"]].notna().all().all()
    assert (develop["units_sold"] >= 0).all()

    # Re-evaluate the FIXED configuration on development-period validation only.
    validation_start = pd.Timestamp("2011-07-18")
    train = develop.loc[develop["week_start"] < validation_start]
    validation = develop.loc[develop["week_start"] >= validation_start]
    assert len(train) == 43715 and len(validation) == 12864
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingRegressor(**PARAMETERS)),
    ])
    candidate.fit(train[FEATURES].astype(float), train["units_sold"])
    val_predictions = candidate.predict(validation[FEATURES].astype(float))
    val_mae = float(mean_absolute_error(validation["units_sold"], val_predictions))
    baseline_mae = float(mean_absolute_error(validation["units_sold"], validation["lag_1"]))
    if not np.isfinite(val_mae) or val_mae >= baseline_mae:
        raise RuntimeError("Model fails the prespecified development-validation baseline gate")

    # Final serving artifact: refit the same configuration using ONLY development rows.
    approved = Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingRegressor(**PARAMETERS)),
    ])
    approved.fit(develop[FEATURES].astype(float), develop["units_sold"])
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_file = output / "approved_model.joblib"
    joblib.dump(approved, model_file, compress=3)
    digest_model = digest(model_file)
    manifest = {
        "artifact_kind": "retrained_locked_configuration_from_development_data",
        "source_data": "Chen (2015), Online Retail, UCI ML Repository, CC BY 4.0",
        "dataset_sha256": checksum,
        "model_sha256": digest_model,
        "model_id": "dds8530-" + digest_model[:16],
        "feature_order": FEATURES,
        "hyperparameters": PARAMETERS,
        "train_rows_final": len(develop),
        "validation_train_rows": len(train),
        "validation_rows": len(validation),
        "validation_mae": val_mae,
        "validation_baseline_mae": baseline_mae,
        "holdout_rows_excluded_from_training": len(holdout),
        "historical_holdout_not_rerun": True,
        "sklearn_version": sklearn.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Verify serialized artifact against the in-memory model without using test outcomes.
    reloaded = joblib.load(model_file)
    sample = validation[FEATURES].head(20).astype(float)
    if not np.allclose(approved.predict(sample), reloaded.predict(sample), atol=1e-12, rtol=1e-12):
        raise RuntimeError("Released artifact did not reproduce saved predictions")
    print("RELEASE MODEL BUILD: PASS")
    print("Source SHA-256:", checksum)
    print("Artifact SHA-256:", digest_model)
    print("Development validation MAE:", round(val_mae, 4))
    print("Development baseline MAE:", round(baseline_mae, 4))
    print("Final holdout NOT used in training or approval:", len(holdout), "rows")
    print("Saved:", output)


if __name__ == "__main__":
    main()
