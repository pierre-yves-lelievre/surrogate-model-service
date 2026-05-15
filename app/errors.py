from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ServiceError(Exception):
    code: str = "service_error"
    status_code: int = 500
    default_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None):
        self.message = message or self.default_message
        super().__init__(self.message)


class ModelNotFoundError(ServiceError):
    code = "model_not_found"
    status_code = 404
    default_message = "Model not found."


class JobNotFoundError(ServiceError):
    code = "job_not_found"
    status_code = 404
    default_message = "Job not found."


class InvalidDatasetError(ServiceError):
    code = "invalid_dataset"
    status_code = 422
    default_message = "Invalid dataset."


class TrainingFailedError(ServiceError):
    code = "training_failed"
    status_code = 500
    default_message = "Model training failed."


async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "code": exc.code},
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    detail = errors[0]["msg"] if errors else "Validation error."
    return JSONResponse(
        status_code=422,
        content={"detail": detail, "code": "validation_error"},
    )
