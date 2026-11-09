
# Cats vs Dogs — Deployment Architecture

## Overview

This project implements an end-to-end MLOps workflow for binary
image classification using PyTorch, FastAPI, Docker, DVC,
MLflow, and GitHub Actions.

## Model

- Architecture: MobileNetV2
- Framework: PyTorch
- Classes: Cat and Dog
- Validation accuracy: 98.00%
- Test accuracy: 99.33%
- Production checkpoint: models/best_model.pth
- Model versioning: SHA-256 fingerprint

## Local Production Deployment

Build the Linux AMD64 image:

```bash
docker build --platform linux/amd64 \
  -f Dockerfile.cloud -t cats-dogs-cloud:1.0 .
```

Start the deployment:

```bash
./scripts/deploy_local.sh
```

Verify the deployment:

```bash
./scripts/smoke_test.sh
```

## API Endpoints

- GET /health — API and model status
- POST /predict — Image classification
- GET /metrics — Prediction counters and latency
- GET /docs — Interactive FastAPI documentation

## CI Pipeline

GitHub Actions automatically performs:

1. Python unit and API tests
2. Offline Cloud Run configuration validation
3. Docker image build with a synthetic checkpoint
4. Container startup and health verification
5. Synthetic-image inference smoke test
6. Monitoring counter verification

The synthetic CI checkpoint is not a trained classifier.
The real production checkpoint is excluded from Git.

## Model Lifecycle

Training produces a candidate checkpoint.

Candidate models are evaluated before promotion.
Production models are protected by promotion safeguards,
checkpoint backups, and rollback support.

## Future Cloud Deployment

Target: Google Cloud Run

Configuration: deploy/cloudrun.yaml

Cloud deployment has not been activated.
No Google Cloud billing or cloud resources are required
for the current workflow.

Before any future deployment, validate authentication,
artifact storage, resource limits, and potential costs.

## Cost Policy

The project currently uses local Docker and GitHub Actions.

Google Cloud deployment is intentionally disabled to avoid
cloud charges. GitHub Actions usage is subject to GitHub's
plan allowances.
