import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request

from app.api import router
from app.config import settings
from app.logging_setup import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    log = get_logger(__name__)
    log.info("startup", version=settings.app_version, models_dir=str(settings.models_dir))
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    yield
    log.info("shutdown")


app = FastAPI(title="Surrogate Model Service", version=settings.app_version, lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=req_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


app.include_router(router)


def main():
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
