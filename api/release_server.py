"""Containerized model-serving application for approved, locally built artifacts.

Loads only a build-produced model with a checksum-matching manifest. The joblib
format must NEVER be used to load arbitrary externally supplied models.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import math
import os

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = Path(os.environ.get("DDS8530_MODEL_DIR", str(ROOT / "model_release"))).resolve()
MANIFEST_FILE = MODEL_DIRECTORY / "manifest.json"
MODEL_FILE = MODEL_DIRECTORY / "approved_model.joblib"


def _read_model():
    if not (MANIFEST_FILE.is_file() and MODEL_FILE.is_file()):
        raise RuntimeError("Approved model artifact is unavailable")
    manifest = json.loads(MANIFEST_FILE.read_text())
    with MODEL_FILE.open("rb") as fd:
        check = sha256(fd.read()).hexdigest()
    if check != manifest.get("model_sha256"):
        raise RuntimeError("Model integrity check failed")
    if manifest.get("artifact_kind") != "retrained_locked_configuration_from_development_data":
        raise RuntimeError("Unrecognized model artifact policy")
    if not isinstance(manifest.get("feature_order"), list) or len(manifest["feature_order"]) != 7:
        raise RuntimeError("Invalid model feature manifest")
    # Image artifact is built from trusted project code, never from HTTP input.
    model = joblib.load(MODEL_FILE)
    return manifest, model


MANIFEST, MODEL = _read_model()
FEATURES = MANIFEST["feature_order"]
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8192
COUNTS = Counter("dds8530_predictions", "Prediction requests by outcome", ["status"])
LATENCY = Histogram("dds8530_prediction_duration_seconds", "Prediction request duration in seconds")


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError("JSON object required")
    missing = sorted(set(FEATURES) - set(payload))
    unknown = sorted(set(payload) - set(FEATURES))
    if missing or unknown:
        raise ValueError(f"Missing features: {missing}; unexpected features: {unknown}")
    values = {}
    for field in FEATURES:
        raw = payload[field]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{field} must be a number")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        if (field.startswith("lag_") or field == "rolling_prev_4") and value < 0:
            raise ValueError(f"{field} cannot be negative")
        if field in ("week_sin", "week_cos") and not -1.000001 <= value <= 1.000001:
            raise ValueError(f"{field} must be between -1 and 1")
        if field == "uk_holiday_week" and value not in (0, 1):
            raise ValueError("uk_holiday_week must be 0 or 1")
        values[field] = value
    return pd.DataFrame([values], columns=FEATURES)


@app.get("/health")
def health():
    return jsonify({"status": "ready", "model_loaded": True})


@app.get("/model-info")
def model_info():
    return jsonify({
        "model_id": MANIFEST["model_id"],
        "feature_order": FEATURES,
        "forecast_type": "sequential_one_week_ahead_recorded_unit_sales",
        "development_rows": MANIFEST["train_rows_final"],
    })


@app.post("/predict")
def predict():
    start = perf_counter()
    state = "error"
    try:
        if not request.is_json:
            raise ValueError("Content-Type must be application/json")
        inputs = validate(request.get_json(silent=True))
        result = float(MODEL.predict(inputs)[0])
        if not math.isfinite(result) or result < 0:
            raise RuntimeError("Invalid prediction")
        state = "success"
        return jsonify({"prediction_units": result, "prediction_units_rounded": int(round(result)),
                        "model_id": MANIFEST["model_id"]})
    except ValueError as exc:
        state = "validation_error"
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction error")
        return jsonify({"error": "Prediction unavailable"}), 500
    finally:
        COUNTS.labels(status=state).inc()
        LATENCY.observe(perf_counter() - start)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)
