import io
from pathlib import Path
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
import pypdf

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.services.pdf_service import generate_medical_pdf_report
from backend.app.schemas.report import ReportRequest
from backend.app.core.errors import MedicalAIException

client = TestClient(app)
REPO_ROOT = settings.REPO_ROOT

def test_pdf_endpoint_missing_original_image():
    """Verify missing original image triggers controlled 400 error."""
    payload = {
        "modality": "xray",
        "prediction": "PNEUMONIA",
        "confidence": 0.94,
        "original_png_base64": None,
        "gradcam_png_base64": None
    }
    response = client.post("/report/pdf", json=payload)
    assert response.status_code == 400
    assert "Original scan image is missing or invalid" in response.text

def test_pdf_endpoint_with_missing_gradcam():
    """Verify that when Grad-CAM is unavailable, PDF generates successfully with 2-column layout."""
    # Use real X-ray to generate original image
    xray_path = REPO_ROOT / "Pneumonia_Detection" / "Chest-X-ray-radiograph-with-Pneumonia.png"
    assert xray_path.exists(), f"Image not found at {xray_path}"

    with open(xray_path, "rb") as f:
        file_bytes = f.read()

    # Get real inference from /predict/xray
    pred_res = client.post("/predict/xray", files={"file": (xray_path.name, file_bytes, "image/png")})
    assert pred_res.status_code == 200
    pred_data = pred_res.json()

    # Deliberately remove gradcam to test missing Grad-CAM handling
    pred_data["gradcam_png_base64"] = None

    response = client.post("/report/pdf", json=pred_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "Content-Disposition" in response.headers
    assert response.headers["Content-Disposition"].startswith("attachment; filename=\"MEDVISION_AI_Report_")

    pdf_bytes = response.content
    assert pdf_bytes.startswith(b"%PDF-")

    # Read with PyPDF or check text
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1
    full_text = "".join([page.extract_text() for page in reader.pages])

    assert "MEDVISION AI" in full_text
    assert "PREDICTION SUMMARY" in full_text
    assert "SCAN COMPARISON" in full_text
    assert "Grad-CAM visualization is unavailable" in full_text
    assert "LIMITATIONS AND DISCLAIMER" in full_text
    assert "REPORT METADATA" in full_text

def test_real_xray_pdf_generation():
    """Verify end-to-end PDF report generation for real X-Ray result."""
    xray_path = REPO_ROOT / "Pneumonia_Detection" / "Chest-X-ray-radiograph-with-Pneumonia.png"
    assert xray_path.exists()

    with open(xray_path, "rb") as f:
        file_bytes = f.read()

    pred_res = client.post("/predict/xray", files={"file": (xray_path.name, file_bytes, "image/png")})
    assert pred_res.status_code == 200
    pred_data = pred_res.json()

    assert pred_data["original_png_base64"] is not None
    assert pred_data["gradcam_png_base64"] is not None

    response = client.post("/report/pdf", json=pred_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"

    pdf_bytes = response.content
    assert pdf_bytes.startswith(b"%PDF-")

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1
    full_text = "".join([page.extract_text() for page in reader.pages])

    assert "MEDVISION AI" in full_text
    clean_pred = "".join(pred_data["prediction"].split())
    clean_text = "".join(full_text.split())
    assert clean_pred in clean_text
    assert "ORIGINAL INPUT SCAN" in full_text
    assert "GRAD-CAM VISUALIZATION" in full_text
    assert "CLINICAL RESEARCH DISCLAIMER" in full_text
    assert "REPORT METADATA" in full_text

    # Persist artifact for inspection
    out_dir = REPO_ROOT / "backend" / "tests" / "generated_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "xray_report.pdf").write_bytes(pdf_bytes)

def test_real_ct_pdf_generation():
    """Verify end-to-end PDF report generation for real CT Scan result."""
    ct_path = REPO_ROOT / "CT_Router" / "test_brain.png"
    assert ct_path.exists()

    with open(ct_path, "rb") as f:
        file_bytes = f.read()

    pred_res = client.post("/predict/ct", files={"file": (ct_path.name, file_bytes, "image/png")})
    assert pred_res.status_code == 200
    pred_data = pred_res.json()

    assert pred_data["original_png_base64"] is not None

    response = client.post("/report/pdf", json=pred_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"

    pdf_bytes = response.content
    assert pdf_bytes.startswith(b"%PDF-")

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1
    full_text = "".join([page.extract_text() for page in reader.pages])

    assert "MEDVISION AI" in full_text
    clean_pred = "".join(pred_data["prediction"].split())
    clean_text = "".join(full_text.split())
    assert clean_pred in clean_text
    assert "SCAN COMPARISON" in full_text

    # Persist artifact for inspection
    out_dir = REPO_ROOT / "backend" / "tests" / "generated_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ct_report.pdf").write_bytes(pdf_bytes)

def test_real_mri_pdf_generation():
    """Verify end-to-end PDF report generation for real Brain MRI result."""
    mri_path = REPO_ROOT / "MRI_Router" / "with dementia.jpeg"
    assert mri_path.exists()

    with open(mri_path, "rb") as f:
        file_bytes = f.read()

    pred_res = client.post(
        "/predict/mri",
        files={"file": (mri_path.name, file_bytes, "image/jpeg")},
        data={"subtype": "dementia"}
    )
    assert pred_res.status_code == 200
    pred_data = pred_res.json()

    assert pred_data["original_png_base64"] is not None

    response = client.post("/report/pdf", json=pred_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"

    pdf_bytes = response.content
    assert pdf_bytes.startswith(b"%PDF-")

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1
    full_text = "".join([page.extract_text() for page in reader.pages])

    assert "MEDVISION AI" in full_text
    clean_pred = "".join(pred_data["prediction"].split())
    clean_text = "".join(full_text.split())
    assert clean_pred in clean_text
    assert "SCAN COMPARISON" in full_text

    # Persist artifact for inspection
    out_dir = REPO_ROOT / "backend" / "tests" / "generated_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mri_report.pdf").write_bytes(pdf_bytes)
