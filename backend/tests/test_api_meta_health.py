from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "docs" in body
    assert "meta" in body


def test_meta_endpoint_contains_xai_sections():
    res = client.get("/api/meta")
    assert res.status_code == 200
    body = res.json()
    assert body["name"]
    assert body["version"]
    assert "shap_guide" in body
    assert "endpoint_groups" in body
    assert "xai_diagnostics" in body["endpoint_groups"]
