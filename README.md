# Surrogate Model Service

A production-shaped REST API for training, serving, and evaluating surrogate models — fast ML approximations that replace expensive numerical simulations.

[![CI](https://github.com/pierre-yves-lelievre/surrogate-model-service/actions/workflows/ci.yml/badge.svg)](https://github.com/pierre-yves-lelievre/surrogate-model-service/actions/workflows/ci.yml)

---

## Overview

Numerical simulations — finite-element solvers, CFD runs, physics engines — can take minutes to hours per call. A surrogate model is a cheap ML approximation trained on a corpus of simulation outputs. Once trained, it reduces a multi-hour evaluation to milliseconds while preserving enough accuracy for optimisation loops, uncertainty quantification, or real-time control.

This service exposes that workflow as a REST API: submit a training dataset (feature matrix + target vector), get back a `job_id` immediately, poll until the model is ready, then run inference. Every trained model is stored on disk alongside a reproducibility manifest that records library versions, random state, and dataset shape. Evaluation results accumulate in the same sidecar, so you can track model quality over time without a separate experiment-tracking database.

---

## Quick start

### Docker (recommended)

```bash
docker compose up --build
```

### Local

```bash
uv sync --all-groups   # install all deps including dev
make run               # uvicorn with --reload on :8000
```

### Train → poll → predict

```bash
# 1. Submit a training job — returns immediately
curl -s -X POST localhost:8000/train \
  -H 'Content-Type: application/json' \
  -d '{"features":[[1],[2],[3],[4],[5]],"targets":[2,4,6,8,10]}' | jq
# {
#   "job_id": "a1b2c3d4-e5f6-...",
#   "status": "pending"
# }

# 2. Poll until succeeded
curl -s localhost:8000/train/a1b2c3d4-e5f6-... | jq
# {
#   "job_id": "a1b2c3d4-...",
#   "status": "succeeded",
#   "model_id": "f7e8d9c0-...",
#   "error": null,
#   "created_at": "2024-01-01T00:00:00Z",
#   "started_at": "2024-01-01T00:00:00.012Z",
#   "completed_at": "2024-01-01T00:00:01.340Z"
# }

# 3. Predict
curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"f7e8d9c0-...","features":[[3],[6],[9]]}' | jq
# {
#   "inference_id": "c1d2e3f4-...",
#   "model_id": "f7e8d9c0-...",
#   "predictions": [5.94, 11.88, 17.82],
#   "timestamp": "2024-01-01T00:00:05Z"
# }
```

---

## API reference

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Service health + dependency checks |
| `POST` | `/train` | Submit async training job |
| `GET` | `/train/{job_id}` | Poll training job status |
| `POST` | `/predict` | Run inference on a trained model |
| `POST` | `/evaluate` | Evaluate model + persist metrics |
| `GET` | `/evaluate/{model_id}` | Retrieve all evaluation records |

Full interactive docs at **`http://localhost:8000/docs`** (Swagger UI) and **`/redoc`**.

---

### `GET /health`

```bash
curl -s localhost:8000/health | jq
```
```json
{
  "status": "healthy",
  "checks": {
    "storage_writable": true,
    "models_count": 3,
    "active_jobs": 0
  },
  "version": "0.1.0",
  "uptime_seconds": 142.7
}
```

Returns `503` with `"status": "degraded"` if `storage_writable` is `false`.

---

### `POST /train`

```bash
curl -s -X POST localhost:8000/train \
  -H 'Content-Type: application/json' \
  -d '{
    "features": [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]],
    "targets":  [1.2, 2.4, 3.6, 4.8]
  }' | jq
```
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending"
}
```

Returns `202 Accepted`. Training runs in a `BackgroundTask`; poll `/train/{job_id}` until `succeeded` or `failed`.

---

### `GET /train/{job_id}`

```bash
curl -s localhost:8000/train/a1b2c3d4-e5f6-7890-abcd-ef1234567890 | jq
```
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "succeeded",
  "model_id": "f7e8d9c0-1234-5678-abcd-ef0987654321",
  "error": null,
  "created_at": "2024-01-01T10:00:00Z",
  "started_at": "2024-01-01T10:00:00.015Z",
  "completed_at": "2024-01-01T10:00:01.412Z"
}
```

`status` transitions: `pending → running → succeeded | failed`. Returns `404` for unknown job IDs.

---

### `POST /predict`

```bash
curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "f7e8d9c0-1234-5678-abcd-ef0987654321",
    "features": [[2.5, 1.25], [3.5, 1.75]]
  }' | jq
```
```json
{
  "inference_id": "c1d2e3f4-5678-90ab-cdef-123456789012",
  "model_id": "f7e8d9c0-1234-5678-abcd-ef0987654321",
  "predictions": [3.01, 4.19],
  "timestamp": "2024-01-01T10:01:00Z"
}
```

Predictions are deterministic: `RandomForestRegressor(random_state=42).predict()` is fully reproducible on the same fitted model.

---

### `POST /evaluate`

```bash
curl -s -X POST localhost:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "f7e8d9c0-1234-5678-abcd-ef0987654321",
    "features": [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]],
    "targets":  [1.2, 2.4, 3.6, 4.8]
  }' | jq
```
```json
{
  "model_id": "f7e8d9c0-1234-5678-abcd-ef0987654321",
  "metrics": {
    "mse": 0.0041,
    "rmse": 0.0640,
    "mae": 0.0512,
    "r2": 0.9991
  },
  "evaluated_at": "2024-01-01T10:02:00Z"
}
```

---

### `GET /evaluate/{model_id}`

```bash
curl -s localhost:8000/evaluate/f7e8d9c0-1234-5678-abcd-ef0987654321 | jq
```
```json
{
  "model_id": "f7e8d9c0-1234-5678-abcd-ef0987654321",
  "evaluations": [
    {
      "metrics": {"mse": 0.0041, "rmse": 0.0640, "mae": 0.0512, "r2": 0.9991},
      "evaluated_at": "2024-01-01T10:02:00Z"
    }
  ]
}
```

---

## Running tests

```bash
make test
```

Seven tests in ~1 second. The philosophy: cover every happy path end-to-end and every error path that has a custom HTTP status code — not chasing line coverage. The fixture patches `n_estimators=10` (vs 100 in production) for an 18x speed-up with no loss of contract coverage.

---

## Architecture

```
HTTP request
     │
     ▼
┌──────────────┐   BackgroundTask   ┌─────────────────┐
│  FastAPI app  │ ─────────────────▶ │  _run_training   │
│  (api.py)     │                    │  (trains model)  │
└──────┬───────┘                    └────────┬────────┘
       │                                     │
       │  Depends(get_store)                 │ store.save()
       ▼                                     ▼
┌──────────────┐              ┌──────────────────────────┐
│  ModelStore   │              │  ./models/               │
│  (storage.py) │◀────────────▶│    {model_id}.joblib     │
└──────────────┘              │    {model_id}.json        │
       │                      └──────────────────────────┘
       │  Depends(get_jobs)
       ▼
┌──────────────┐
│   JobStore    │  (in-memory, thread-safe)
│   (jobs.py)   │
└──────────────┘
```

**Training request lifecycle.** `POST /train` validates the request body via Pydantic (shape checks, length parity), creates a `Job` record in `JobStore`, registers `_run_training` as a `BackgroundTask`, and returns `202` with the `job_id` — all before a single tree is grown. The background task transitions the job to `running`, calls `train_model()` which fits a `RandomForestRegressor`, generates a `model_id`, calls `store.save()` to write the `.joblib` and `.json` sidecar atomically, then transitions to `succeeded`. Any exception is caught, stored as `job.error`, and the job transitions to `failed` — it never propagates out of the background task.

---

## Design decisions and trade-offs

**Why FastAPI.** Automatic OpenAPI generation, Pydantic-native request/response validation, and native async support. The `Depends()` injection system made it trivial to swap the `ModelStore` root in tests without touching production code. There was no reason to choose anything else for a service this shape.

**Why `BackgroundTasks` over Celery.** `BackgroundTasks` runs in the same process, requires zero infrastructure, and is the right tool for a single-server demo. The swap path to a real queue is documented below and is a clean interface boundary: `_run_training` is already a plain function that takes explicit arguments — no shared state, no globals. Dropping it behind SQS means changing one call site.

**Why an in-memory `JobStore`.** Same reasoning — zero infrastructure, transparent code. The limitation is documented in the module docstring: jobs don't survive restart. The interface (`create`, `update`, `get`, `count_active`) maps directly to a SQL table with one row per job. The diff to Postgres is small and isolated to `jobs.py`.

**Why filesystem storage with joblib.** The simplest credible artifact store. `ModelStore` is a class with five methods. Nothing in `api.py` knows about file paths. Swapping to S3 means writing a new `ModelStore` implementation — the interface contract is stable and tested.

**Why a flat package with no extra abstractions.** Every interface I considered, I asked: "is there a second implementation today?" For `ModelStore`, the answer is no (yet). For `JobStore`, the answer is no (yet). So I wrote concrete classes. The diff to introduce an `AbstractModelStore` protocol is small when the second implementation arrives and the need is proven.

**Why RandomForest.** Fast to train, well-understood, requires no feature scaling, and `random_state=42` makes predictions fully reproducible. The brief de-emphasised the modelling choice — the interesting part of this service is the API, the job lifecycle, and the storage contract.

**Reproducibility.** Every model is saved with a JSON sidecar recording `sklearn_version`, `python_version`, `random_state`, `n_samples`, and `n_features`. The same binary weights + same library version → identical predictions. Evaluation results accumulate in the same sidecar, so model quality is traceable without a separate database.

---

## Quality choices

- **Structured JSON logging** via structlog. Every log line carries `request_id` (injected by middleware), making it trivial to trace a single request across log aggregators.
- **Custom exception hierarchy** (`ServiceError` → `ModelNotFoundError`, etc.) maps cleanly to HTTP status codes and returns a consistent `{"detail": "...", "code": "..."}` shape. Client code can branch on `code` without parsing error messages.
- **All timestamps UTC, ISO 8601.** No timezone ambiguity anywhere in the system.
- **Training data shapes logged, never values.** Privacy by default — the model learns from the data, the logs don't need to.
- **Pydantic validators reject malformed datasets before any work begins.** Feature matrix row-length consistency and features/targets length parity are checked at deserialization time, not mid-training.
- **Pre-commit hooks** running `ruff check --fix` and `ruff format` on every commit.
- **GitHub Actions CI** running lint + format check + full test suite on every push and PR.

---

## What I didn't build — and how I would

**Durable job store.** Replace `JobStore` with a Postgres-backed implementation behind the same four-method interface. Jobs become a table: `job_id`, `status`, `model_id`, `error`, `created_at`, `started_at`, `completed_at`. Schema managed by Alembic. The API code is unchanged.

**Real task queue.** The API publishes a message to SQS on `POST /train`. A separate worker container (same image, different `CMD`) polls the queue, calls `_run_training`, and writes to S3. Workers autoscale on queue depth via an ECS Service with a CloudWatch alarm. Training failures go to a DLQ. This decouples API latency from training duration and survives API restarts.

**Multi-instance API.** With state externalised (jobs in Postgres, models in S3), run any number of API replicas behind an ALB. Add an LRU in-process model cache on the inference path — keyed by `model_id` — to avoid a cold S3 fetch per prediction.

**Authentication.** API key middleware as a first step (header → DynamoDB lookup). OAuth 2.0 / JWT via AWS Cognito or Auth0 for production, with scopes separating read (`predict`) from write (`train`, `evaluate`).

**Multi-tenancy.** Scope `model_id` by `org_id`; namespace S3 keys as `{org_id}/{model_id}.joblib`. Row-level security in Postgres for the job table. The current UUID-keyed design is compatible — it just needs a prefix.

**Model registry and versioning.** Use MLflow as a model registry. Each `train` call registers a run with parameters, metrics, and the artifact path. `model_id` becomes `registered_name:version`, enabling rollback to a previous version without retraining.

**MLOps monitoring.** Log every inference's feature vector and prediction to S3 (append-only Parquet partitioned by date). Run Evidently on a schedule to detect feature drift and prediction distribution shift. Prometheus metrics (request latency, queue depth, training duration, error rate) scraped by Grafana. PagerDuty alerts on training failure rate and P99 prediction latency.

**IaC.** Terraform for the full stack: VPC, ECS Fargate (API + worker), ALB, SQS + DLQ, S3, RDS Aurora Postgres, IAM roles, CloudWatch log groups. Each environment (`dev`, `staging`, `prod`) is a Terraform workspace. CI builds the image, pushes to ECR, and updates the ECS task definition — no SSH, no manual steps.

**Cancellation, retries, idempotency.** Cancellation: a `DELETE /train/{job_id}` endpoint that sets `status=cancelled` and sends a cancellation signal via a threading `Event`. Retries: SQS message visibility timeout + DLQ after N failures. Idempotency: the client provides `job_id` at submission; the server treats re-submissions of the same `job_id` as no-ops.

**Large datasets.** For datasets that don't fit in a JSON body, accept `{"s3_uri": "s3://bucket/key.parquet"}` as an alternative to inline `features`/`targets`. The worker downloads and parses the file before training. This keeps the API stateless and avoids HTTP body limits.

---

## Project structure

```
surrogate-model-service/
├── app/
│   ├── api.py            # all route handlers
│   ├── config.py         # Settings via pydantic-settings
│   ├── errors.py         # exception hierarchy + FastAPI handlers
│   ├── evaluation.py     # compute_metrics()
│   ├── inference.py      # predict()
│   ├── jobs.py           # Job dataclass + in-memory JobStore
│   ├── logging_setup.py  # structlog JSON config + request_id contextvar
│   ├── main.py           # FastAPI app, lifespan, middleware
│   ├── schemas.py        # Pydantic request/response models
│   ├── storage.py        # ModelStore + reproducibility manifest
│   └── training.py       # train_model()
├── tests/
│   ├── conftest.py       # fixtures: tmp_models_dir, client, wait_for_job
│   └── test_api.py       # 7 integration tests
├── models/               # runtime artefacts (gitignored)
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile            # multi-stage build
├── docker-compose.yml
├── Makefile
├── pyproject.toml        # deps, ruff config, pytest config
└── uv.lock
```

---

## License

MIT — see [LICENSE](LICENSE).
