import unittest
from pathlib import Path

import pandas as pd

from api.server import app, FEATURES

ROOT = Path(__file__).resolve().parents[1]


class PredictionAPITests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

        data = pd.read_csv(
            ROOT / "data/processed/uk_model_ready.csv"
        )

        test = data.loc[
            data["split"] == "test"
        ].reset_index(drop=True)

        saved = pd.read_csv(
            ROOT / "data/processed/final_test_predictions.csv"
        )

        cls.example = {
            name: float(test.iloc[0][name])
            for name in FEATURES
        }
        cls.expected = float(
            saved.iloc[0]["model_prediction"]
        )

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ready")

    def test_model_info(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json["required_features"], FEATURES
        )

    def test_prediction_matches_saved_model(self):
        response = self.client.post(
            "/predict", json=self.example
        )
        self.assertEqual(
            response.status_code, 200,
            response.get_json()
        )
        self.assertAlmostEqual(
            response.json["prediction_units"],
            self.expected,
            places=7,
        )

    def test_missing_feature_rejected(self):
        invalid = dict(self.example)
        invalid.pop("lag_1")
        response = self.client.post(
            "/predict", json=invalid
        )
        self.assertEqual(response.status_code, 400)

    def test_negative_lag_rejected(self):
        invalid = dict(self.example)
        invalid["lag_1"] = -10
        response = self.client.post(
            "/predict", json=invalid
        )
        self.assertEqual(response.status_code, 400)

    def test_metrics(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"dds8530_predictions_total",
            response.data,
        )


if __name__ == "__main__":
    unittest.main()
