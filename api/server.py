from pathlib import Path
from time import perf_counter
import json
import math
import os
import re

from flask import Flask, Response, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
import pandas as pd

# Load only the previously verified, locally generated model.
# MLflow requires explicit permission to deserialize trusted artifacts.
os.environ["MLFLOW_ALLOW_PICKLE_DESERIALIZATION"] = "true"

import mlflow
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[1]
AUDIT = json.loads(
    (ROOT / "reports/final_model_evaluation.json").read_text()
)

MODEL_URI = AUDIT["final_model_uri"]

if not re.fullmatch(r"models:/m-[0-9a-f]{32}", MODEL_URI):
    raise RuntimeError("Unexpected model identifier.")

tracking_db = ROOT / "mlflow_tracking.db"
if not tracking_db.is_file():
    raise RuntimeError("Local MLflow tracking database missing.")

mlflow.set_tracking_uri(
    f"sqlite:///{tracking_db.as_posix()}"
)
model = mlflow.sklearn.load_model(MODEL_URI)

FEATURES = [
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_prev_4",
    "week_sin",
    "week_cos",
    "uk_holiday_week",
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8192

REQUEST_COUNT = Counter(
    "dds8530_predictions",
    "Prediction requests by outcome",
    ["status"],
)
REQUEST_DURATION = Histogram(
    "dds8530_prediction_duration_seconds",
    "Prediction request processing duration",
)


@app.get("/health")
def health():
    return jsonify({
        "status": "ready",
        "model_loaded": True,
    })


@app.get("/model-info")
def model_info():
    return jsonify({
        "model_uri": MODEL_URI,
        "model_run_id": AUDIT["model_run_id"],
        "required_features": FEATURES,
        "forecast_type": "one_week_ahead",
        "prediction_unit": "recorded_sales_units",
    })


def validate_features(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = sorted(set(FEATURES) - set(payload))
    extra = sorted(set(payload) - set(FEATURES))

    if missing or extra:
        raise ValueError(
            f"Missing features: {missing}; unexpected features: {extra}"
        )

    values = {}

    for name in FEATURES:
        value = payload[name]

        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise ValueError(f"{name} must be numeric.")

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

        if name.startswith("lag_") or name == "rolling_prev_4":
            if value < 0:
                raise ValueError(
                    f"{name} cannot be negative."
                )

        if name in ("week_sin", "week_cos"):
            if not -1.000001 <= value <= 1.000001:
                raise ValueError(
                    f"{name} must be between -1 and 1."
                )

        if name == "uk_holiday_week" and value not in (0, 1):
            raise ValueError(
                "uk_holiday_week must be 0 or 1."
            )

        values[name] = value

    return pd.DataFrame([values], columns=FEATURES)


@app.post("/predict")
def predict():
    started = perf_counter()
    status = "error"

    try:
        if not request.is_json:
            raise ValueError("Content-Type must be application/json.")

        features = validate_features(
            request.get_json(silent=True)
        )

        prediction = float(model.predict(features)[0])

        if not math.isfinite(prediction) or prediction < 0:
            raise RuntimeError("Model produced an invalid prediction.")

        status = "success"

        return jsonify({
            "prediction_units": prediction,
            "prediction_units_rounded": int(round(prediction)),
            "model_uri": MODEL_URI,
        })

    except ValueError as exc:
        status = "validation_error"
        return jsonify({"error": str(exc)}), 400

    except Exception:
        app.logger.exception("Prediction request failed")
        return jsonify({
            "error": "Prediction unavailable."
        }), 500

    finally:
        REQUEST_COUNT.labels(status=status).inc()
        REQUEST_DURATION.observe(
            perf_counter() - started
        )


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        content_type=CONTENT_TYPE_LATEST,
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False,
        use_reloader=False,
    )
