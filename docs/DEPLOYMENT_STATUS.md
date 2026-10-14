# CI/CD and AWS deployment: implementation boundary

**Verified:** Docker-based ETL, Airflow execution, MLflow model training, local Flask service, Prometheus collection, and Grafana display. **Not verified:** an actual AWS App Runner endpoint or successful GitHub Actions AWS release. No AWS-hosted application was created by the reported local experiment.

The repository includes an executable, manually initiated GitHub Actions workflow (`.github/workflows/deploy-aws.yml`) for the controlled *release process*: re-create the fixed-configuration model artifact from the CC BY 4.0 derived development dataset, validate its HTTP contract, containerize, authenticate to AWS with short-lived OIDC credentials, push to Amazon ECR, and request an App Runner deployment. This is a **deployment configuration**, not a claim that AWS resources exist or that the workflow has passed in a live AWS account.

## Required AWS infrastructure and authorization

A connected AWS account must already have:

1. A private Amazon ECR repository for the Docker image, with a retained and authorized `production` image tag.
2. After the first ECR image is published, an App Runner service configured to pull from that repository **using the `production` tag** and with automatic image deployments disabled; its ECR access role must be authorized. The container listens on port **8000**; the health endpoint is `/health`.
3. A GitHub OIDC IAM role that restricts its trust conditions to this specific repository and the protected `production` environment and permits only the necessary ECR write operations and `apprunner:StartDeployment` on the selected App Runner service. The policy must not grant administrator or wildcard blanket permissions.
4. A protected GitHub Actions `production` environment with required reviewer approval, regional and service settings, and these nonsecret environment variables: `AWS_REGION`, `AWS_ROLE_TO_ASSUME`, `ECR_REPOSITORY`, and `APP_RUNNER_SERVICE_ARN`.
5. A documented cost/retention plan. ECR, App Runner, log retention, and network transfer may incur charges even for a demonstration. Disable or delete cloud services when evidence is no longer needed.

The first approved dispatch uses `bootstrap_ecr` to publish a candidate image, **without calling App Runner**. An App Runner service can then be created using the published `production` tag; subsequent approved dispatches use `deploy_service`. Both modes require explicit authorization, and the ECR repository must already exist. The deployment run must be **manually dispatched and explicitly approved**. The workflow does not write raw customer records or original UCI spreadsheets to AWS. Its image contains an approved *retrained* artifact and application code; it is not a transfer of the existing local MLflow tracking database.

## Verification criteria for any future AWS execution

A successful GitHub workflow run and ECR push are necessary but **not sufficient** evidence of a healthy deployed model. After deployment, independently verify App Runner health, send a valid test request to `/predict`, validate the returned model ID, check error handling and logs, and record a screenshot of the live AWS endpoint/service revision. Do not treat `start-deployment` acknowledgement alone as a completed rollout. Any externally exposed prediction API requires appropriate access controls before use with non-public data; this educational service has **no user authentication**.

## Optional encrypted S3 export

`scripts/publish_to_s3.py` prepares a source-fingerprinted weekly aggregate upload to an existing private S3 bucket with SSE-S3 and an integrity metadata check, but it defaults to dry run. An actual upload requires an explicit `--execute`, authorized credentials, and a bucket and region. The recorded experiment does not establish an S3 upload for Assignment 5.

## Operational limitations

This is a reproducible lab, not a continuously learning production system. The historical dataset ends in 2011, the Airflow weekly DAG remains paused, the API expects precomputed feature inputs, authentication and drift monitoring are not implemented, and the test-set WAPE of 68.096% remains high. The monitored 24.1 ms P95 figure was observed during controlled local replay; it is not an AWS service-level performance guarantee.
