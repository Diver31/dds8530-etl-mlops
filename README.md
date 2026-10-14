# Retail Forecasting: Python ETL, MLOps, and AWS Deployment Workflow

**DDS8530 — Big Data Integration | National University | October 2026**

This repository contains an implemented, audited retail forecasting laboratory using Python, pandas, Dask, SQLAlchemy/PostgreSQL, scikit-learn, MLflow, Apache Airflow, Flask, Prometheus, and Grafana. A separate, approval-gated GitHub Actions workflow provides the configuration for an AWS ECR/App Runner deployment. **Local operation was demonstrated; a live AWS-hosted version was not deployed or verified during the reported experiment.**

## Research objective and design

Can weekly recorded merchandise sales for sufficiently active products be forecast more accurately than a previous-week naive baseline? The forecasting demonstration uses the public [UCI Online Retail dataset](https://doi.org/10.24432/C5BW33), which contains 541,909 transaction records from December 2010 to December 2011. The model targets *positive recorded UK merchandise units per product-week*, not unconstrained customer demand or net accounting revenue. Dates are historic and outputs are not current inventory recommendations.

**Data lineage.** The initial extract included a structured Excel/CSV transaction source (including free-text product descriptions), a relational PostgreSQL source read through SQLAlchemy, and a historical public-holiday REST API (Nager.Date). The descriptions informed an explicit reviewed stock-code classification rather than serving as unvalidated model features. The source archive was fingerprinted; cancellations, nonpositive quantity, and nonpositive price rows were excluded from gross positive recorded sales. Financial charges and vouchers were classified separately; ambiguous manual entries were held for review; possible legitimate repeated invoice lines were retained. The model-ready dataset reflects these documented decisions.

**Dataset preparation.** Dask processed the sales data in four partitions; 52 complete reporting weeks were retained and aggregated for 3,782 globally observed products. UK-specific aggregation yielded 93,608 observed product-week rows. Product eligibility was determined using **only the 40-week development period**: at least 20 observed sales weeks, resulting in 1,608 selected products. Missing observed records after the first product appearance were imputed as zero *recorded sales*, not proof of available stock or zero latent demand. The last 12 weeks remained outside development and model selection. Features include previous-week lags, a four-week trailing mean, cyclical week-of-year indicators, and nationwide UK holiday indicators. UK regional holidays were not indiscriminately applied nationally. The final 12-week period contained zero nationwide holidays under the API's selected `global` flag.

**Learning and evaluation.** A previous-week baseline was compared with a scikit-learn `HistGradientBoostingRegressor(loss="poisson")` under an eight-week time-ordered development validation. The chosen fixed configuration was refit on all 56,579 development observations, then evaluated on 19,296 final chronological holdout observations as a *rolling one-week-ahead* protocol. Each next week's lag values presume that the immediately preceding week's actual recorded sales have become available. This is not a 12-week recursive forecast made at a single origin.

| Test measure | Previous-week baseline | Gradient boosting |
|---|---:|---:|
| MAE (units per product-week) | 39.838 | **35.038** |
| RMSE (units) | 123.250 | **106.444** |
| WAPE | 77.424% | **68.096%** |

The final model reduces MAE by **12.05%**, but WAPE remains high. A production inventory use case would require external and temporal validation, stock availability data, richer covariates, and error monitoring across product segments. The numbers above are **recorded results from the author's October 2026 local execution**, not values inferred from a simulated AWS deployment.

## Architecture and evidence

```text
UCI historic transaction archive ──> pandas cleaning / classification ──┐
                                                                         v
                                                 Dask weekly partitions / aggregation
                                                                         |
                                                           SQLAlchemy -> PostgreSQL
                                                                         |
                             UK geographic isolation / lagged features + UK holidays API
                                                                         |
                                  scikit-learn training and evaluation -> MLflow
                                                                         |
                                       locally verified Flask model-serving API
                                                                         |
                                       Prometheus scrapes -> Grafana monitoring

Airflow 3: executed nine dependent ETL/MLflow tasks using Docker-mounted inputs.
GitHub Actions: CI and manually approved ECR/App Runner CD workflows are defined.
AWS hosted model endpoint: NOT verified at the time of this project report.
```

| Evidence | Observed result |
|---|---|
| Independent SQL reconciliation | 96,063 original and Airflow product-week rows; **zero differing records** |
| Airflow orchestration | All nine execution tasks green/success |
| MLflow model tracking | Development metrics, model identifiers, and artifact reloading verified |
| Model artifact reproducibility | 25 sampled holdout predictions reproduced to maximum 1.42 × 10^-14 absolute difference in the original local environment |
| Local HTTP API | Six in-process Flask tests and a separate live HTTP test passed |
| Monitoring | Grafana displayed availability 1, 76 cumulative successes, 16 cumulative rejected requests, and a 24.1 ms displayed P95 estimate during controlled replay |

The repository includes supporting run-specific evidence under [`evidence/`](evidence/) and the following **actual captured screenshots**:

- [Successful nine-task Airflow run](docs/figures/airflow_nine_tasks.png)
- [MLflow side-by-side validation](docs/figures/mlflow_validation_comparison.png)
- [Grafana during monitored prediction replay](docs/figures/grafana_live_monitoring.png)

These are real local observations, **not** cloud deployment benchmarks. A single 0.32-second Dask aggregation provides no controlled evidence of speedup over pandas.

## Repository layout

| Directory/file | Purpose |
|---|---|
| `api/server.py` | Original MLflow-based local service; requires the original local MLflow artifacts |
| `api/release_server.py` | Deployment service loading a checksum-checked, CI-generated model artifact |
| `scripts/` | Original ETL, validation, feature, modeling, and audit scripts; S3 upload and release build |
| `orchestration/dags/` | Airflow verification and executable nine-task ETL/MLOps DAGs |
| `monitoring/` | Prometheus scrape and six-panel Grafana dashboard configuration |
| `assets/uk_model_ready.csv` | Derived, attributed model-ready public-data fixture (no customer IDs) to make automated release builds reproducible |
| `evidence/` | Contemporaneous validation audit summaries from successful local runs |
| `docs/` | Scientific results, assumptions, figures, AWS status, data-use and architecture details |
| `deployment/requirements.txt`, `Dockerfile` | Pinned release environment and Docker image definition |
| `.github/workflows/ci.yml` | Source integrity, deterministic configuration, model rebuild, API contract, and Docker build checks |
| `.github/workflows/deploy-aws.yml` | **Manual, approval-gated** AWS ECR publish and App Runner deployment request (requires configured AWS resources and permissions) |
| `submission/Prince_DDS8530_Assignment5.docx` | Five-page body, APA-formatted academic report, with separate title and references |

## Reproduce the historical ETL on a Mac/Linux development machine

Python **3.13** and Docker Compose are required. Historical pipeline and source credentials are kept out of version control. `requirements.txt` corresponds to the local development ETL/MLOps environment. A fresh clone starts with no user passwords or raw transaction files.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Configure real unique local values for POSTGRES_PASSWORD and GRAFANA_ADMIN_PASSWORD in .env.
docker compose up -d --wait postgres
python scripts/ingest_uci_online_retail.py
python scripts/curate_transactions.py
python scripts/review_stock_codes.py
python scripts/classify_merchandise.py
python scripts/aggregate_weekly.py
python scripts/load_weekly_postgres.py
python scripts/extract_holidays.py
python scripts/assess_model_sources.py
python scripts/build_uk_weekly.py
python scripts/build_ml_features.py
python scripts/train_mlflow.py
```

The scripts operate from the repository root and write generated data under `data/`, audit reports under `reports/`, and local MLflow artifacts to the current workspace. These generated files are excluded from GitHub. **Do not rerun the one-time final test as a model-selection exercise.** On a clean independent recreation only, `python scripts/evaluate_final_model.py` performs the fixed-hyperparameter, one-time holdout evaluation and intentionally refuses to replace an existing evaluation report. `python scripts/verify_mlflow_artifact.py` checks its saved model.

Launch the original local model-based API only after creating its MLflow database and final model:

```bash
python -m unittest discover -s tests -p test_api.py -v
python -m api.server
```

For MLflow dashboard: `mlflow ui --backend-store-uri "sqlite:///$PWD/mlflow_tracking.db" --host 127.0.0.1 --port 5000`. Docker Compose starts Prometheus (`localhost:9090`) and Grafana (`localhost:3000`); the local API (`localhost:8000`) must be running for active metrics. Use separate shell processes for these services.

For Airflow, copy the matching database credentials into the local, ignored `orchestration/db.env` using the supplied example, then:

```bash
docker compose -f orchestration/compose.yaml up -d --build
```

Open `http://127.0.0.1:8080`, leave the historical weekly schedule **paused**, and manually trigger `dds8530_etl_execution` once. This DAG intentionally uses an independent target `etl.fact_weekly_sales_airflow` and isolated run-local MLflow tracking stores. Its successful execution **does not imply new source-data availability**.

## Portable model release and automated checks

The upstream MLflow registry and SQLite DB are intentionally absent from GitHub. To permit a fresh checkout to build/test an API anyway, a **separate locked-configuration model release** is trained from the attributed, derived fixture. This artifact is **newly generated**, not an unverified byte-for-byte copy of the previously logged MLflow model. The build reads only development records; it rejects altered fixture fingerprints, tests the fixed configuration against a development-only naive baseline, saves an immutable SHA-256 manifest, reloads the artifact, and verifies predictions.

```bash
python -m pip install -r deployment/requirements.txt
python scripts/build_release_model.py
python -m unittest discover -s tests -p test_release_service.py -v
docker build -t dds8530-forecast-api:local .
```

`model_release/` is generated at build time and gitignored. `api/release_server.py` loads *only* a model generated by the trusted release pipeline and validates its checksum before deserialization. The model uses joblib, which is not safe for untrusted arbitrary uploads. Public API clients cannot supply serialized models. Port 8000 exposes `/health`, `/model-info`, `/predict`, and `/metrics`, and request features must already be computed and ordered according to the manifest. The API does **not** implement a production-grade product/time feature store, authentication, or drift detection.

The GitHub **CI** workflow runs repository checks, an independent locked-model rebuild, Flask endpoint tests, and image building. The **CD** workflow is *defined* but does not run without manual dispatch, a protected `production` environment, and configured AWS IAM/ECR resources (an initial `bootstrap_ecr` run seeds ECR before App Runner is created). See [AWS deployment scope and prerequisites](docs/DEPLOYMENT_STATUS.md). If AWS S3 is available, `scripts/publish_to_s3.py` implements an **opt-in** encrypted weekly export (default dry run); no AWS write is claimed in this assignment run.

## Methodological constraints and data use

- User-derived and public source values are attributed to **Chen (2015), *Online Retail* [Data set], UCI Machine Learning Repository**, DOI [10.24432/C5BW33](https://doi.org/10.24432/C5BW33), licensed **CC BY 4.0**. The included `assets/uk_model_ready.csv` is a filtered, aggregated and engineered derivative, without CustomerID, invoice number, person names, or transaction-level narratives. Its original repository authorship and modification/aggregation are documented here.
- `Nager.Date` historical holiday flags are a separate external API; nationally flagged holidays do not represent all regional holidays. There were no nationally flagged holiday weeks in the test window. Holiday importance is not causally established.
- Dataset-derived quantities are observed recorded gross-positive-sale quantities, not recovered latent demand after stockouts, product unavailability, or customer returns. No external business deployment or scaling validation was performed.
- See [validation tables](docs/RESULTS.md) and the [academic submission](submission/Prince_DDS8530_Assignment5.docx) for measured results and peer-reviewed scholarly context.

### Scholarly sources

Akidau et al. (2015), *Proceedings of the VLDB Endowment*, 8(12), 1792–1803. https://doi.org/10.14778/2824032.2824076

Kreuzberger, Kühl, and Hirschl (2023), *IEEE Access*, 11, 31866–31879. https://doi.org/10.1109/ACCESS.2023.3262138

Sculley et al. (2015), *Advances in Neural Information Processing Systems*, 28, 2503–2511. https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems
