import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

from ..core.config import settings
from ..core.errors import ModelInferenceException
from ..schemas.prediction import PredictionResponse
from ..utils.image_utils import pil_to_base64_data_url, numpy_overlay_to_base64

# Add repo root to sys.path so pipeline.py and modules resolve
if str(settings.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(settings.REPO_ROOT))

from pipeline import XRayRouter, SpecialistModel, generate_gradcam, ROUTER_THRESHOLD

class XRayService:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._router: Optional[XRayRouter] = None
        self._chest_model: Optional[SpecialistModel] = None
        self._mura_model: Optional[SpecialistModel] = None
        self._init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "XRayService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_router(self) -> XRayRouter:
        if self._router is None:
            with self._init_lock:
                if self._router is None:
                    self._router = XRayRouter()
        return self._router

    def _get_specialist(self, branch: str) -> SpecialistModel:
        with self._init_lock:
            if branch == "CHEST":
                if self._chest_model is None:
                    self._chest_model = SpecialistModel("CHEST")
                return self._chest_model
            elif branch == "MUSCULOSKELETAL":
                if self._mura_model is None:
                    self._mura_model = SpecialistModel("MUSCULOSKELETAL")
                return self._mura_model
            else:
                raise ValueError(f"Unknown X-ray branch: {branch}")

    def predict(self, image: Image.Image, subtype: Optional[str] = None) -> PredictionResponse:
        with self._lock:
            try:
                gray_image = image.convert("L")
                router = self._get_router()
                router_res = router.predict(gray_image)

                warnings = []
                # Check for explicit subtype override or router result
                if subtype in ("musculoskeletal", "mura"):
                    branch = "MUSCULOSKELETAL"
                elif subtype in ("chest", "pneumonia", "default") and subtype != "default":
                    branch = "CHEST"
                else:
                    if not router_res["confident"]:
                        warnings.append(
                            f"ROUTER NOTICE: Confidence score ({router_res['confidence']:.2f}%) "
                            f"is below safety threshold ({ROUTER_THRESHOLD * 100:.0f}%). "
                            f"Proceeding with primary detected anatomy: {router_res['route']}."
                        )
                    branch = router_res["route"]

                specialist = self._get_specialist(branch)
                prediction = specialist.predict(gray_image)
                cam = generate_gradcam(specialist, prediction, gray_image)

                # Format probabilities
                if branch == "CHEST" and prediction.get("models"):
                    probs = {
                        "NORMAL": round(float(prediction["negative_probability"]) / 100.0, 4),
                        "PNEUMONIA": round(float(prediction["positive_probability"]) / 100.0, 4),
                    }
                    pred_class = prediction["prediction"]
                    confidence = round(float(prediction["confidence"]) / 100.0, 4)
                    threshold = float(prediction.get("threshold", 0.50))
                    model_name = f"Chest Multi-Model Ensemble ({', '.join(prediction['models'].keys())})"
                    dataset = "NIH ChestX-ray14 & RSNA Pneumonia"
                    metrics = {
                        "Consensus Ratio": str(prediction.get("consensus_ratio", "3/3")),
                        "Ensemble Weighting": "V2-S (0.40) + B3 (0.35) + B7 (0.25)",
                    }
                    routed_to = "chest"
                else:
                    probs = {
                        prediction["negative_class"]: round(float(prediction["negative_probability"]) / 100.0, 4),
                        prediction["positive_class"]: round(float(prediction["positive_probability"]) / 100.0, 4),
                    }
                    pred_class = prediction["prediction"]
                    confidence = round(float(prediction["confidence"]) / 100.0, 4)
                    threshold = float(prediction.get("threshold", 0.50))
                    model_name = "Stanford MURA Specialist (EfficientNet-B3)"
                    dataset = "Stanford MURA (Musculoskeletal Radiographs)"
                    metrics = {
                        "Resolution": "300x300 px",
                        "Accuracy": "86.4%",
                        "Target": "Bone / Joint Abnormality",
                    }
                    routed_to = "musculoskeletal"

                # Generate base64 visuals
                orig_b64 = pil_to_base64_data_url(image)
                cam_b64 = numpy_overlay_to_base64(image, cam, alpha=0.45, cmap="jet")

                return PredictionResponse(
                    modality="xray",
                    routed_to=routed_to,
                    prediction=pred_class,
                    probabilities=probs,
                    confidence=confidence,
                    threshold=threshold,
                    model_name=model_name,
                    framework="PyTorch 2.x",
                    input_size="300-600px" if branch == "CHEST" else "300x300 px",
                    dataset=dataset,
                    metrics=metrics,
                    gradcam_png_base64=cam_b64,
                    original_png_base64=orig_b64,
                    visualizations={"original": orig_b64, "overlay": cam_b64},
                    disclaimer=settings.DEFAULT_DISCLAIMER,
                    warnings=warnings,
                    status="success",
                    metadata={
                        "router": router_res,
                        "ensemble_votes": prediction.get("votes") if branch == "CHEST" else None,
                        "visualizations": {"original": orig_b64, "overlay": cam_b64},
                    }
                )
            except Exception as e:
                raise ModelInferenceException(f"X-Ray inference failed: {str(e)}")
