"""
In-memory job store for tracking async training jobs.

NOTE: this store is ephemeral — all jobs are lost on restart.
A future phase will replace this with a Postgres-backed implementation.
"""

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

JobStatus = Literal["pending", "running", "succeeded", "failed"]


@dataclass
class Job:
    job_id: str
    status: JobStatus = "pending"
    model_id: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None


class JobStore:
    """Thread-safe in-memory store for training job records."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str) -> Job:
        """Create a new Job with status=pending and register it in the store."""
        job = Job(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def update(self, job_id: str, **fields) -> None:
        """Update arbitrary fields on an existing job; silently ignores unknown job IDs."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def get(self, job_id: str) -> Job | None:
        """Return the Job for the given ID, or None if not found."""
        with self._lock:
            return self._jobs.get(job_id)

    def count_active(self) -> int:
        """Return the number of jobs currently in pending or running state."""
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status in {"pending", "running"})
