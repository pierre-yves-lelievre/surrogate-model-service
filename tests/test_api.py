import json

from tests.conftest import wait_for_job

FEATURES = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
TARGETS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _train_and_wait(client):
    r = client.post("/train", json={"features": FEATURES, "targets": TARGETS})
    assert r.status_code == 202
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "succeeded"
    return job["model_id"]


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["checks"]["storage_writable"] is True
    assert "models_count" in body["checks"]
    assert "active_jobs" in body["checks"]
    assert "uptime_seconds" in body
    assert "version" in body


def test_train_then_predict_roundtrip(client):
    model_id = _train_and_wait(client)

    predict_features = [[2.0], [4.0], [6.0]]
    r = client.post("/predict", json={"model_id": model_id, "features": predict_features})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == len(predict_features)
    assert all(isinstance(p, float) for p in body["predictions"])


def test_predictions_are_reproducible(client):
    model_id = _train_and_wait(client)

    payload = {"model_id": model_id, "features": [[3.0], [5.0]]}
    p1 = client.post("/predict", json=payload).json()["predictions"]
    p2 = client.post("/predict", json=payload).json()["predictions"]
    assert p1 == p2


def test_predict_unknown_model_returns_404(client):
    r = client.post("/predict", json={"model_id": "no-such-model", "features": [[1.0]]})
    assert r.status_code == 404
    assert r.json()["code"] == "model_not_found"


def test_evaluate_and_retrieve(client):
    model_id = _train_and_wait(client)

    r = client.post(
        "/evaluate",
        json={"model_id": model_id, "features": FEATURES, "targets": TARGETS},
    )
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    assert metrics["r2"] > 0.8  # sensible fit on near-linear data

    r = client.get(f"/evaluate/{model_id}")
    assert r.status_code == 200
    evals = r.json()["evaluations"]
    assert len(evals) == 1
    assert evals[0]["metrics"]["r2"] == metrics["r2"]


def test_invalid_dataset_returns_422(client):
    r = client.post(
        "/train",
        json={"features": [[1.0], [2.0], [3.0]], "targets": [1.0, 2.0]},  # length mismatch
    )
    assert r.status_code == 422
    assert r.json()["code"] == "invalid_dataset"


def test_reproducibility_manifest_written(client, tmp_models_dir):
    model_id = _train_and_wait(client)

    sidecar = tmp_models_dir / f"{model_id}.json"
    assert sidecar.exists(), "JSON sidecar not written"

    manifest = json.loads(sidecar.read_text())
    for key in ("random_state", "sklearn_version", "python_version", "n_samples", "n_features"):
        assert key in manifest, f"missing manifest key: {key}"

    assert manifest["n_samples"] == len(FEATURES)
    assert manifest["n_features"] == 1
    assert manifest["random_state"] == 42
