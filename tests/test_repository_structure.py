"""Offline repository provenance, structure, and source contract tests."""
from pathlib import Path
import hashlib
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]

class RepositoryContractTests(unittest.TestCase):
    def test_expected_implementation(self):
        needed = [
            'README.md', '.env.example', 'compose.yaml', 'compose.override.yaml',
            'api/server.py', 'api/release_server.py', 'Dockerfile',
            'scripts/ingest_uci_online_retail.py', 'scripts/curate_transactions.py',
            'scripts/aggregate_weekly.py', 'scripts/build_uk_weekly.py',
            'scripts/build_ml_features.py', 'scripts/train_mlflow.py',
            'scripts/evaluate_final_model.py', 'scripts/build_release_model.py',
            'scripts/publish_to_s3.py',
            'orchestration/compose.yaml', 'orchestration/Dockerfile',
            'orchestration/dags/dds8530_etl_execution.py',
            'monitoring/prometheus.yml',
            '.github/workflows/ci.yml', '.github/workflows/deploy-aws.yml',
            'evidence/final_model_evaluation.json',
            'docs/figures/airflow_nine_tasks.png',
            'docs/figures/mlflow_validation_comparison.png',
            'docs/figures/grafana_live_monitoring.png',
            'submission/Prince_DDS8530_Assignment5.docx',
        ]
        for name in needed:
            with self.subTest(path=name):
                self.assertTrue((ROOT / name).is_file(), name)

    def test_all_python_files_compile_without_side_effects(self):
        for parent in ('api', 'scripts', 'tests', 'orchestration/dags'):
            for path in (ROOT / parent).rglob('*.py'):
                with self.subTest(path=str(path.relative_to(ROOT))):
                    compile(path.read_text(), str(path), 'exec')

    def test_derived_dataset_has_known_fingerprint(self):
        path = ROOT / 'assets/uk_model_ready.csv'
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            'b6c5959b1dd208548fb69f2b62732e0ec0728414f0d91c08eebecbd3718d1b96',
        )

    def test_evidence_reconciliation(self):
        def read(name):
            return json.loads((ROOT / 'evidence' / name).read_text())
        source = read('source_profile.json')
        classified = read('merchandise_classification_audit.json')
        weekly = read('weekly_aggregation_audit.json')
        features = read('ml_feature_audit.json')
        evaluation = read('final_model_evaluation.json')
        airflow = read('airflow_mlops_validation.json')
        assert source['row_count'] == source['csv_row_count'] == 541909
        assert classified['input_records'] == 530104
        self.assertEqual(sum(classified['classification_counts'].values()), 530104)
        self.assertEqual(classified['classification_counts']['merchandise_candidate'], 527758)
        self.assertEqual(weekly['observed_product_week_rows'], 96063)
        self.assertEqual(weekly['complete_week_units'], 5241234)
        self.assertEqual(features['train_rows'] + features['test_rows'], features['model_ready_rows'])
        self.assertEqual(features['test_rows'], 19296)
        self.assertFalse(airflow['final_holdout_retested'])
        self.assertLess(evaluation['gradient_boosting']['test_mae'], evaluation['baseline']['test_mae'])
        self.assertEqual(evaluation['test_rows'], 19296)
        self.assertTrue(evaluation['test_evaluated'])

    def test_monitored_configuration(self):
        dashboard = json.loads((ROOT / 'monitoring/grafana/provisioning/dashboards/json/forecast_api.json').read_text())
        self.assertEqual(len(dashboard['panels']), 6)
        self.assertEqual(dashboard['uid'], 'dds8530-api-monitor')

    def test_sensitive_paths_are_gitignored(self):
        exclusions = (ROOT / '.gitignore').read_text().splitlines()
        for entry in ('.env', '.venv/', '/data/', '/reports/',
                      '/model_release/', '/mlflow_tracking.db*',
                      'orchestration/db.env', '/orchestration/runs/'):
            with self.subTest(ignore=entry):
                self.assertIn(entry, exclusions)

if __name__ == '__main__':
    unittest.main()
