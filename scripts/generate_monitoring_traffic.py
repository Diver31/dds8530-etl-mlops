from pathlib import Path
from datetime import datetime, timezone
import json
import math
import re
import time

import pandas as pd
import requests

ROOT = Path.cwd()
BASE = "http://127.0.0.1:8000"
OUTPUT = ROOT / "reports/monitoring_traffic_audit.json"

session = requests.Session()

health = session.get(f"{BASE}/health", timeout=10)
health.raise_for_status()

if health.json().get("status") != "ready":
    raise RuntimeError("Prediction API is not ready.")

info = session.get(f"{BASE}/model-info", timeout=10)
info.raise_for_status()

features = info.json()["required_features"]

data = pd.read_csv(
    ROOT / "data/processed/uk_model_ready.csv"
)

test = data.loc[data["split"] == "test"]

if len(test) < 75:
    raise RuntimeError("Insufficient test examples.")

samples = test.head(75).reset_index(drop=True)

def counters():
    response = session.get(f"{BASE}/metrics", timeout=10)
    response.raise_for_status()

    matches = re.findall(
        r'^dds8530_predictions_total'
        r'\{status="([^"]+)"\}\s+([0-9.eE+-]+)',
        response.text,
        flags=re.MULTILINE,
    )

    return {
        name: float(value)
        for name, value in matches
    }

before = counters()
started = time.perf_counter()

successful = 0
rejected = 0

print("=== GENERATING MONITORED TRAFFIC ===")
print("Expected successful requests: 75")
print("Expected rejected requests: 15")
print("Starting controlled test...")

for batch in range(15):
    for position in range(5):
        index = batch * 5 + position

        payload = {
            name: float(samples.iloc[index][name])
            for name in features
        }

        response = session.post(
            f"{BASE}/predict",
            json=payload,
            timeout=15,
        )
        response.raise_for_status()

        prediction = response.json()["prediction_units"]

        if not math.isfinite(prediction) or prediction < 0:
            raise RuntimeError("Invalid model prediction.")

        successful += 1
        time.sleep(0.4)

    # Deliberately omit a required feature.
    invalid = dict(payload)
    invalid.pop("lag_1")

    response = session.post(
        f"{BASE}/predict",
        json=invalid,
        timeout=15,
    )

    if response.status_code != 400:
        raise RuntimeError(
            "Invalid request was not rejected."
        )

    rejected += 1
    time.sleep(0.4)

    if (batch + 1) % 5 == 0:
        print(
            f"Progress: {successful} successful, "
            f"{rejected} rejected"
        )

print("Waiting for Prometheus to collect metrics...")
time.sleep(6)

after = counters()
elapsed = time.perf_counter() - started

success_delta = int(
    after.get("success", 0)
    - before.get("success", 0)
)

rejected_delta = int(
    after.get("validation_error", 0)
    - before.get("validation_error", 0)
)

if success_delta != successful:
    raise RuntimeError("Successful request counter mismatch.")

if rejected_delta != rejected:
    raise RuntimeError("Rejected request counter mismatch.")

# Verify Prometheus can still scrape the API.
prometheus = session.get(
    "http://127.0.0.1:9090/api/v1/query",
    params={
        "query": 'up{job="dds8530-flask-api"}'
    },
    timeout=10,
)
prometheus.raise_for_status()

values = prometheus.json()["data"]["result"]

scrape_up = bool(
    values and values[0]["value"][1] == "1"
)

if not scrape_up:
    raise RuntimeError("Prometheus scrape target is down.")

audit = {
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "test_type": "Controlled historical feature replay",
    "model_uri": info.json()["model_uri"],
    "successful_requests": successful,
    "rejected_requests": rejected,
    "success_counter_delta": success_delta,
    "rejected_counter_delta": rejected_delta,
    "elapsed_seconds": round(elapsed, 2),
    "prometheus_scrape": "UP",
    "validation": "PASS",
    "limitations": (
        "Synthetic request activity; not a production "
        "load or model accuracy evaluation."
    ),
}

OUTPUT.write_text(json.dumps(audit, indent=2))

print("\n=== MONITORING TRAFFIC RESULTS ===")
print("Successful requests:", successful)
print("Rejected requests:", rejected)
print("Successful counter delta:", success_delta)
print("Rejected counter delta:", rejected_delta)
print("Duration (seconds):", round(elapsed, 2))
print("Prometheus scrape: UP")
print("Request counter reconciliation: PASS")
print("Audit:", OUTPUT)
print("MONITORING TRAFFIC VALIDATION: PASS")
