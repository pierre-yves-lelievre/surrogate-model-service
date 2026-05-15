from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.errors import InvalidDatasetError


class TrainRequest(BaseModel):
    features: list[list[float]] = Field(
        ..., json_schema_extra={"example": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}
    )
    targets: list[float] = Field(..., json_schema_extra={"example": [1.0, 2.0, 3.0]})
    model_name: str | None = Field(None, json_schema_extra={"example": "my_model"})

    @model_validator(mode="after")
    def validate_dataset(self) -> "TrainRequest":
        if not self.features:
            raise InvalidDatasetError("features must not be empty.")
        if len(self.features) != len(self.targets):
            raise InvalidDatasetError(
                f"features and targets length mismatch: {len(self.features)} vs {len(self.targets)}."
            )
        row_len = len(self.features[0])
        for i, row in enumerate(self.features):
            if len(row) != row_len:
                raise InvalidDatasetError(
                    f"Row {i} has {len(row)} features, expected {row_len}."
                )
        return self


class TrainResponse(BaseModel):
    job_id: str = Field(..., json_schema_extra={"example": "a1b2c3d4-..."})
    status: str = Field(..., json_schema_extra={"example": "completed"})


class JobStatusResponse(BaseModel):
    job_id: str = Field(..., json_schema_extra={"example": "a1b2c3d4-..."})
    status: str = Field(..., json_schema_extra={"example": "running"})
    model_id: str | None = Field(None, json_schema_extra={"example": "e5f6g7h8-..."})
    error: str | None = Field(None, json_schema_extra={"example": None})
    created_at: datetime = Field(..., json_schema_extra={"example": "2024-01-01T00:00:00Z"})
    started_at: datetime | None = Field(None, json_schema_extra={"example": "2024-01-01T00:00:01Z"})
    completed_at: datetime | None = Field(None, json_schema_extra={"example": "2024-01-01T00:00:05Z"})


class PredictRequest(BaseModel):
    model_id: str = Field(..., json_schema_extra={"example": "e5f6g7h8-..."})
    features: list[list[float]] = Field(
        ..., json_schema_extra={"example": [[1.0, 2.0], [3.0, 4.0]]}
    )

    @model_validator(mode="after")
    def validate_features(self) -> "PredictRequest":
        if not self.features:
            raise InvalidDatasetError("features must not be empty.")
        row_len = len(self.features[0])
        for i, row in enumerate(self.features):
            if len(row) != row_len:
                raise InvalidDatasetError(
                    f"Row {i} has {len(row)} features, expected {row_len}."
                )
        return self


class PredictResponse(BaseModel):
    inference_id: str = Field(..., json_schema_extra={"example": "f1e2d3c4-..."})
    model_id: str = Field(..., json_schema_extra={"example": "e5f6g7h8-..."})
    predictions: list[float] = Field(..., json_schema_extra={"example": [1.0, 2.0]})
    timestamp: datetime = Field(..., json_schema_extra={"example": "2024-01-01T00:00:00Z"})


class EvaluateRequest(BaseModel):
    model_id: str = Field(..., json_schema_extra={"example": "e5f6g7h8-..."})
    features: list[list[float]] = Field(
        ..., json_schema_extra={"example": [[1.0, 2.0], [3.0, 4.0]]}
    )
    targets: list[float] = Field(..., json_schema_extra={"example": [1.0, 2.0]})

    @model_validator(mode="after")
    def validate_dataset(self) -> "EvaluateRequest":
        if not self.features:
            raise InvalidDatasetError("features must not be empty.")
        if len(self.features) != len(self.targets):
            raise InvalidDatasetError(
                f"features and targets length mismatch: {len(self.features)} vs {len(self.targets)}."
            )
        row_len = len(self.features[0])
        for i, row in enumerate(self.features):
            if len(row) != row_len:
                raise InvalidDatasetError(
                    f"Row {i} has {len(row)} features, expected {row_len}."
                )
        return self


class EvaluateResponse(BaseModel):
    model_id: str = Field(..., json_schema_extra={"example": "e5f6g7h8-..."})
    metrics: dict[str, float] = Field(..., json_schema_extra={"example": {"r2": 0.99, "rmse": 0.01}})
    evaluated_at: datetime = Field(..., json_schema_extra={"example": "2024-01-01T00:00:00Z"})


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    checks: dict[str, Any] = Field(..., json_schema_extra={"example": {"storage": "ok"}})
    version: str = Field(..., json_schema_extra={"example": "0.1.0"})
    uptime_seconds: float = Field(..., json_schema_extra={"example": 42.0})
