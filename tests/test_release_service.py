"""Contract and live inference tests for the reproducibly built container model."""
from pathlib import Path
import json
import math
import os
import unittest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "model_release/approved_model.joblib"
MANIFEST = ROOT / "model_release/manifest.json"

@unittest.skipUnless(ARTIFACT.is_file() and MANIFEST.is_file(),
                     "Release artifact not built in this environment")
class ReleaseServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from api.release_server import app, MANIFEST, FEATURES
        cls.client = app.test_client()
        cls.manifest = MANIFEST
        cls.features = FEATURES
        cls.payload = {"lag_1": 10.0, "lag_2": 11.0, "lag_4": 12.0,
                       "rolling_prev_4": 11.0, "week_sin": 0.5,
                       "week_cos": 0.5, "uk_holiday_week": 0}

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["model_loaded"])

    def test_model_provenance(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["model_id"], self.manifest["model_id"])
        self.assertEqual(response.json["feature_order"], self.features)

    def test_positive_prediction(self):
        response = self.client.post("/predict", json=self.payload)
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertTrue(math.isfinite(response.json["prediction_units"]))
        self.assertGreaterEqual(response.json["prediction_units"], 0)

    def test_bad_request(self):
        payload = dict(self.payload)
        payload.pop("lag_1")
        self.assertEqual(self.client.post("/predict", json=payload).status_code, 400)
        self.assertEqual(self.client.post("/predict", json={"lag_1": 3}).status_code, 400)

    def test_metrics(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"dds8530_predictions_total", response.data)


if __name__ == "__main__":
    unittest.main()
