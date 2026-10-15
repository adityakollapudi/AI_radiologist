import os
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings

client = TestClient(app)

REPO_ROOT = settings.REPO_ROOT

def test_real_xray_inference_smoke():
    xray_path = REPO_ROOT / "Pneumonia_Detection" / "Chest-X-ray-radiograph-with-Pneumonia.png"
    if not xray_path.exists():
        print(f"Skipping X-ray test: image not found at {xray_path}")
        return

    with open(xray_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/predict/xray",
        files={"file": (xray_path.name, file_bytes, "image/png")}
    )
    assert response.status_code == 200, f"Expected 200 but got {response.status_code}: {response.text}"
    data = response.json()
    assert data["status"] == "success"
    assert data["modality"] == "xray"
    assert data["routed_to"] == "chest"
    assert data["prediction"] in ("PNEUMONIA", "NORMAL")
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["gradcam_png_base64"].startswith("data:image/png;base64,")
    assert data["original_png_base64"].startswith("data:image/png;base64,")

def test_real_ct_inference_smoke():
    ct_path = REPO_ROOT / "CT_Router" / "test_brain.png"
    if not ct_path.exists():
        print(f"Skipping CT test: image not found at {ct_path}")
        return

    with open(ct_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/predict/ct",
        files={"file": (ct_path.name, file_bytes, "image/png")}
    )
    assert response.status_code == 200, f"Expected 200 but got {response.status_code}: {response.text}"
    data = response.json()
    assert data["status"] == "success"
    assert data["modality"] == "ct"
    assert data["routed_to"] == "ct_brain_bleed"
    assert "probabilities" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["gradcam_png_base64"].startswith("data:image/png;base64,")

def test_real_mri_inference_smoke():
    mri_path = REPO_ROOT / "MRI_Router" / "with dementia.jpeg"
    if not mri_path.exists():
        print(f"Skipping MRI test: image not found at {mri_path}")
        return

    with open(mri_path, "rb") as f:
        file_bytes = f.read()

    # Test explicit dementia task
    response = client.post(
        "/predict/mri",
        files={"file": (mri_path.name, file_bytes, "image/jpeg")},
        data={"subtype": "dementia"}
    )
    assert response.status_code == 200, f"Expected 200 but got {response.status_code}: {response.text}"
    data = response.json()
    assert data["status"] == "success"
    assert data["modality"] == "mri"
    assert data["routed_to"] == "mri_dementia"
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["gradcam_png_base64"].startswith("data:image/png;base64,")
