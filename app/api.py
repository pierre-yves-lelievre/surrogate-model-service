import uuid
from functools import lru_cache

from fastapi import APIRouter, Depends

from app.config import settings
from app.schemas import TrainRequest, TrainResponse
from app.storage import ModelStore, build_manifest
from app.training import train_model

router = APIRouter()


@lru_cache(maxsize=1)
def get_store() -> ModelStore:
    return ModelStore(settings.models_dir)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/train", response_model=TrainResponse, status_code=201)
async def train(request: TrainRequest, store: ModelStore = Depends(get_store)):
    model = train_model(request.features, request.targets)

    model_id = str(uuid.uuid4())
    manifest = build_manifest(
        n_samples=len(request.features),
        n_features=len(request.features[0]),
    )
    store.save(model_id, model, manifest)

    return TrainResponse(job_id=model_id, status="completed")
