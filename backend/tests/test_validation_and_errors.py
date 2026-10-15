import io
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_unsupported_modality():
    file_content = b"fake image bytes"
    files = {"file": ("test.png", file_content, "image/png")}
    response = client.post("/predict/ultrasound", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "Unsupported modality" in data["error"]

def test_invalid_file_extension():
    file_content = b"Some plain text document"
    files = {"file": ("document.txt", file_content, "text/plain")}
    response = client.post("/predict/xray", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "Unsupported file format" in data["error"]

def test_empty_file():
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/predict/xray", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "empty" in data["error"].lower()

def test_corrupted_image_content():
    corrupted_bytes = b"Not a valid PNG or JPEG header content"
    files = {"file": ("corrupted.png", corrupted_bytes, "image/png")}
    response = client.post("/predict/ct", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "corrupted" in data["error"].lower() or "not a recognized" in data["error"].lower()
