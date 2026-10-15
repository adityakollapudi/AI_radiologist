from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["online"] is True
    assert data["status"] == "healthy"
    assert "device" in data
    assert "version" in data
    assert set(data["available_modalities"]) == {"xray", "ct", "mri"}

def test_api_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["online"] is True
    assert data["status"] == "healthy"

def test_modalities_endpoint():
    response = client.get("/modalities")
    assert response.status_code == 200
    data = response.json()
    assert "modalities" in data
    assert "xray" in data["modalities"]
    assert "ct" in data["modalities"]
    assert "mri" in data["modalities"]
    
    xray_info = data["modalities"]["xray"]
    assert ".png" in xray_info["supported_extensions"]
    assert len(xray_info["tasks"]) > 0

def test_models_specs_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "xray_chest" in data
    assert "ct" in data
    assert "mri" in data
