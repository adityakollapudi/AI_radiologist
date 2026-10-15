import io
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app.schemas.prediction import PredictionResponse

client = TestClient(app)

def create_dummy_png_bytes():
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_xray_mocked_adapter():
    dummy_response = PredictionResponse(
        modality="xray",
        routed_to="chest",
        prediction="NORMAL",
        probabilities={"NORMAL": 0.94, "PNEUMONIA": 0.06},
        confidence=0.94,
        threshold=0.50,
        model_name="Mocked X-Ray Specialist",
        framework="PyTorch 2.x",
        input_size="300x300 px",
        dataset="NIH ChestX-ray14",
        metrics={"Accuracy": "95%"},
        gradcam_png_base64="data:image/png;base64,mockcam",
        original_png_base64="data:image/png;base64,mockorig",
        disclaimer="Research use only",
        warnings=[],
        status="success"
    )

    with patch("backend.app.api.routes.xray.XRayService.get_instance") as mock_get_inst:
        mock_service = MagicMock()
        mock_service.predict.return_value = dummy_response
        mock_get_inst.return_value = mock_service

        png_bytes = create_dummy_png_bytes()
        response = client.post(
            "/predict/xray",
            files={"file": ("test.png", png_bytes, "image/png")},
            data={"subtype": "chest"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["modality"] == "xray"
        assert data["prediction"] == "NORMAL"
        assert data["confidence"] == 0.94
        assert "NORMAL" in data["probabilities"]
        assert data["status"] == "success"
        assert data["request_id"] is not None

def test_ct_mocked_adapter():
    dummy_response = PredictionResponse(
        modality="ct",
        routed_to="ct_axial_slice",
        prediction="COVID-19",
        probabilities={"NORMAL": 0.05, "PNEUMONIA": 0.15, "COVID-19": 0.80},
        confidence=0.80,
        threshold=0.50,
        model_name="Mocked CT Specialist",
        framework="PyTorch 2.x",
        input_size="256x256 px",
        dataset="COVIDx CT",
        metrics={"Accuracy": "96%"},
        gradcam_png_base64="data:image/png;base64,mockcam",
        original_png_base64="data:image/png;base64,mockorig",
        disclaimer="Research use only",
        warnings=[],
        status="success"
    )

    with patch("backend.app.api.routes.ct.CTService.get_instance") as mock_get_inst:
        mock_service = MagicMock()
        mock_service.predict.return_value = dummy_response
        mock_get_inst.return_value = mock_service

        png_bytes = create_dummy_png_bytes()
        response = client.post(
            "/predict/ct",
            files={"file": ("test.png", png_bytes, "image/png")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["modality"] == "ct"
        assert data["prediction"] == "COVID-19"
        assert data["confidence"] == 0.80

def test_mri_mocked_adapter():
    dummy_response = PredictionResponse(
        modality="mri",
        routed_to="brain_mri",
        prediction="NO TUMOR",
        probabilities={"NO TUMOR": 0.92, "TUMOR": 0.08},
        confidence=0.92,
        threshold=0.50,
        model_name="Mocked MRI Screener",
        framework="PyTorch 2.x",
        input_size="240x240 px",
        dataset="BraTS Benchmark",
        metrics={"Accuracy": "91%"},
        gradcam_png_base64="data:image/png;base64,mockcam",
        original_png_base64="data:image/png;base64,mockorig",
        disclaimer="Research use only",
        warnings=[],
        status="success"
    )

    with patch("backend.app.api.routes.mri.MRIService.get_instance") as mock_get_inst:
        mock_service = MagicMock()
        mock_service.predict.return_value = dummy_response
        mock_get_inst.return_value = mock_service

        png_bytes = create_dummy_png_bytes()
        response = client.post(
            "/predict/mri",
            files={"file": ("test.png", png_bytes, "image/png")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["modality"] == "mri"
        assert data["prediction"] == "NO TUMOR"
        assert data["confidence"] == 0.92
