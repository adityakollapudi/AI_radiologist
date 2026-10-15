import sys
import uuid
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

from ..core.config import settings
from ..core.errors import ModelInferenceException
from ..schemas.prediction import PredictionResponse
from ..utils.image_utils import pil_to_base64_data_url

# Add CT directories to sys.path
CT_ROOT = settings.REPO_ROOT / "CT_Router"
COVID_DIR = CT_ROOT / "Covid19-CT-Scan--Detection"
RSNA_DIR = CT_ROOT / "RSNA2019_Intracranial-Hemorrhage-Detection"

for p in (str(settings.REPO_ROOT), str(CT_ROOT), str(COVID_DIR), str(RSNA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from CT_Router.src.inference import MedicalCTPredictor

class CTService:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._predictor: Optional[MedicalCTPredictor] = None
        self._init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "CTService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_predictor(self) -> MedicalCTPredictor:
        if self._predictor is None:
            with self._init_lock:
                if self._predictor is None:
                    device_str = "cuda" if settings.DEVICE.type == "cuda" else "cpu"
                    self._predictor = MedicalCTPredictor(device=device_str)
        return self._predictor

    def predict(self, image: Image.Image, subtype: Optional[str] = None) -> PredictionResponse:
        with self._lock:
            temp_path = None
            try:
                predictor = self._get_predictor()

                # Save image temporarily for CT pipeline inference
                temp_filename = f"ct_{uuid.uuid4().hex}.png"
                temp_path = settings.TEMP_DIR / temp_filename
                image.save(temp_path, format="PNG")

                res = predictor.predict(
                    temp_path,
                    gradcam_dir=None,
                    multi_layer=True
                )

                router_res = res.get("router", {})
                anatomy = router_res.get("prediction", "UNKNOWN")
                expert_res = res.get("expert_result", {})
                warnings = []

                if router_res.get("uncertain"):
                    warnings.append(
                        "ROUTER UNCERTAINTY BAND: CT slice confidence fell within manual review zone. "
                        "Proceeding with primary detected anatomy triage."
                    )

                probs = {}
                metrics = {}

                if anatomy == "CHEST_CT":
                    routed_to = "ct_axial_slice"
                    diagnosis = expert_res.get("diagnosis", "Normal").upper()
                    raw_probs = expert_res.get("probabilities", {})
                    for k, v in raw_probs.items():
                        probs[k.upper()] = round(float(v), 4)

                    conf = float(expert_res.get("confidence", 0.0))
                    if conf <= 1.0 and conf > 0:
                        conf = round(conf, 4)
                    else:
                        conf = round(max(probs.values()) if probs else 0.90, 4)

                    model_name = "Chest CT COVID-19/Pneumonia Specialist (EfficientNet-B0)"
                    dataset = "COVIDx CT-2A / CC-CCII Multicenter Cohort"
                    metrics = {
                        "Architecture": "EfficientNet-B0 + Multi-Layer LayerCAM",
                        "Entropy": f"{expert_res.get('entropy', 0.0):.4f}",
                        "Top-1 Margin": str(expert_res.get("top1_margin_pct", "N/A")),
                    }

                else:  # BRAIN_CT
                    routed_to = "ct_brain_bleed"
                    diag_title = expert_res.get("diag_title", expert_res.get("diagnosis", "No Hemorrhage"))
                    diagnosis = diag_title.upper()
                    raw_probs = expert_res.get("probabilities", {})
                    for k, v in raw_probs.items():
                        probs[k.upper()] = round(float(v), 4)

                    has_bleed = diagnosis != "NO ACUTE INTRACRANIAL HEMORRHAGE DETECTED" and "NO BLEED" not in diagnosis
                    if has_bleed:
                        warnings.append(
                            "CRITICAL FINDING: Acute intracranial hyperdensity detected consistent with hemorrhage. "
                            "Requires immediate neurosurgical and emergency radiology review."
                        )

                    conf = round(max(probs.values()) if probs else 0.95, 4)
                    model_name = "RSNA Intracranial Hemorrhage Specialist (SE-ResNeXt-101)"
                    dataset = "RSNA 2019 Intracranial Hemorrhage Challenge (CQ500)"
                    metrics = {
                        "Architecture": "SE-ResNeXt-101 (32x4d)",
                        "Grad-CAM Fused Stages": "Layer 1 to Layer 4 Native Projection",
                    }

                # Visualizations
                orig_b64 = pil_to_base64_data_url(image)
                cam_overlay = res.get("cam_overlay_pil")
                cam_b64 = pil_to_base64_data_url(cam_overlay) if cam_overlay else orig_b64

                return PredictionResponse(
                    modality="ct",
                    routed_to=routed_to,
                    prediction=diagnosis,
                    probabilities=probs,
                    confidence=conf,
                    threshold=0.50,
                    model_name=model_name,
                    framework="PyTorch 2.x",
                    input_size="256x256 px",
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
                        "layer_weights": res.get("gradcam_layer_weights", {}),
                        "clinical_summary": expert_res.get("clinical_summary"),
                        "visualizations": {"original": orig_b64, "overlay": cam_b64},
                    }
                )

            except Exception as e:
                raise ModelInferenceException(f"CT inference failed: {str(e)}")
            finally:
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
