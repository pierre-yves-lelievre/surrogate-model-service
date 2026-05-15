def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["checks"]["storage_writable"] is True
    assert "models_count" in body["checks"]
    assert "active_jobs" in body["checks"]
    assert "uptime_seconds" in body
    assert "version" in body
