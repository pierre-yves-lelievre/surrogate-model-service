import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn
from sklearn.base import BaseEstimator

from app.errors import ModelNotFoundError
from app.logging_setup import get_logger

log = get_logger(__name__)


class ModelStore:
    """Filesystem-backed store for trained model artefacts and their JSON sidecars."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _model_path(self, model_id: str) -> Path:
        return self.root / f"{model_id}.joblib"

    def _manifest_path(self, model_id: str) -> Path:
        return self.root / f"{model_id}.json"

    def save(self, model_id: str, model: BaseEstimator, manifest: dict) -> None:
        """Persist a fitted model as a .joblib file and write the manifest sidecar."""
        joblib.dump(model, self._model_path(model_id))
        with self._manifest_path(model_id).open("w") as f:
            json.dump({**manifest, "evaluations": []}, f, indent=2)
        log.info("model_saved", model_id=model_id)

    def load(self, model_id: str) -> BaseEstimator:
        """Load and return a fitted model by ID; raises ModelNotFoundError if missing."""
        path = self._model_path(model_id)
        if not path.exists():
            raise ModelNotFoundError(f"Model '{model_id}' not found.")
        return joblib.load(path)

    def load_manifest(self, model_id: str) -> dict:
        """Load and return the JSON manifest sidecar; raises ModelNotFoundError if missing."""
        path = self._manifest_path(model_id)
        if not path.exists():
            raise ModelNotFoundError(f"Manifest for model '{model_id}' not found.")
        with path.open() as f:
            return json.load(f)

    def exists(self, model_id: str) -> bool:
        """Return True if the model artefact file exists on disk."""
        return self._model_path(model_id).exists()

    def list_models(self) -> list[str]:
        """Return a sorted list of all model IDs present in the store."""
        return [p.stem for p in sorted(self.root.glob("*.joblib"))]

    def is_writable(self) -> bool:
        """Return True if the store root directory is writable."""
        try:
            probe = self.root / ".write_probe"
            probe.touch()
            probe.unlink()
            return True
        except OSError:
            return False

    def save_evaluation(self, model_id: str, metrics: dict) -> None:
        """Append an evaluation record (metrics + timestamp) to the model's manifest sidecar."""
        manifest = self.load_manifest(model_id)
        manifest.setdefault("evaluations", []).append(
            {"metrics": metrics, "evaluated_at": datetime.now(UTC).isoformat()}
        )
        with self._manifest_path(model_id).open("w") as f:
            json.dump(manifest, f, indent=2)

    def get_evaluations(self, model_id: str) -> list[dict]:
        """Return all evaluation records stored in the model's manifest sidecar."""
        return self.load_manifest(model_id).get("evaluations", [])


def build_manifest(
    n_samples: int,
    n_features: int,
    random_state: int = 42,
    model_type: str = "RandomForestRegressor",
) -> dict:
    """Build a reproducibility manifest capturing library versions and dataset shape."""
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "random_state": random_state,
        "n_samples": n_samples,
        "n_features": n_features,
        "model_type": model_type,
    }
