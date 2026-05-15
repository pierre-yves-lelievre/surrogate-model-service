import uuid
from datetime import UTC, datetime
from functools import lru_cache

from fastapi import APIRouter, BackgroundTasks, Depends

from app.config import settings
from app.errors import JobNotFoundError
from app.jobs import Job, JobStore
from app.logging_setup import get_logger
from app.schemas import JobStatusResponse, TrainRequest, TrainResponse
from app.storage import ModelStore, build_manifest
from app.training import train_model

router = APIRouter()
log = get_logger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────

_job_store = JobStore()


@lru_cache(maxsize=1)
def get_store() -> ModelStore:
    return ModelStore(settings.models_dir)


def get_jobs() -> JobStore:
    return _job_store


# ── Background task ───────────────────────────────────────────────────────────


def _run_training(
    job_id: str,
    features: list[list[float]],
    targets: list[float],
    jobs: JobStore,
    store: ModelStore,
) -> None:
    jobs.update(job_id, status="running", started_at=datetime.now(UTC))
    try:
        model = train_model(features, targets)
        model_id = str(uuid.uuid4())
        manifest = build_manifest(
            n_samples=len(features),
            n_features=len(features[0]),
        )
        store.save(model_id, model, manifest)
        jobs.update(
            job_id,
            status="succeeded",
            model_id=model_id,
            completed_at=datetime.now(UTC),
        )
        log.info("job_succeeded", job_id=job_id, model_id=model_id)
    except Exception as exc:
        jobs.update(
            job_id,
            status="failed",
            error=str(exc),
            completed_at=datetime.now(UTC),
        )
        log.exception("job_failed", job_id=job_id, error=str(exc))


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post(
    "/train",
    response_model=TrainResponse,
    status_code=202,
    summary="Submit a training job",
    description=(
        "Enqueues an async training job for a RandomForestRegressor. "
        "Returns immediately with a `job_id` you can poll via `GET /train/{job_id}`."
    ),
    responses={
        202: {
            "content": {
                "application/json": {"example": {"job_id": "a1b2c3d4-...", "status": "pending"}}
            }
        }
    },
)
async def train(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    store: ModelStore = Depends(get_store),
    jobs: JobStore = Depends(get_jobs),
) -> TrainResponse:
    job_id = str(uuid.uuid4())
    jobs.create(job_id)
    background_tasks.add_task(_run_training, job_id, request.features, request.targets, jobs, store)
    return TrainResponse(job_id=job_id, status="pending")


@router.get(
    "/train/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll a training job",
    description=(
        "Returns the current status of a training job. "
        "Poll until `status` is `succeeded` (provides `model_id`) or `failed` (provides `error`)."
    ),
    responses={
        404: {
            "content": {
                "application/json": {
                    "example": {"detail": "Job not found.", "code": "job_not_found"}
                }
            }
        }
    },
)
async def get_job(
    job_id: str,
    jobs: JobStore = Depends(get_jobs),
) -> JobStatusResponse:
    job: Job | None = jobs.get(job_id)
    if job is None:
        raise JobNotFoundError(f"Job '{job_id}' not found.")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        model_id=job.model_id,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
