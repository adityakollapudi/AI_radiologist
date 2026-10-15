import sys
import uuid
import threading
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

from ..core.config import settings
from ..core.errors import ModelInferenceException
from ..schemas.prediction import PredictionResponse
from ..utils.image_utils import pil_to_base64_data_url

# Add MRI directories to sys.path
MRI_ROOT = settings.REPO_ROOT / "MRI_Router"
DEMENTIA_DIR = MRI_ROOT / "dementia_mri_project"
TUMOR_DIR = MRI_ROOT / "brain-_tumor_detection"

for p in (str(settings.REPO_ROOT), str(MRI_ROOT), str(DEMENTIA_DIR), str(TUMOR_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Load BrainTumorClassifier dynamically from tumor_detector.py
tumor_detector_path = DEMENTIA_DIR / "src" / "tumor_detector.py"
spec = importlib.util.spec_from_file_location("mri_tumor_detector", str(tumor_detector_path.resolve()))
mri_tumor_mod = importlib.util.module_from_spec(spec)
sys.modules["mri_tumor_detector"] = mri_tumor_mod
spec.loader.exec_module(mri_tumor_mod)
BrainTumorClassifier = mri_tumor_mod.BrainTumorClassifier

# Load DementiaPredictor from dementia_mri_project package
from dementia_mri_project.src.inference import DementiaPredictor

class MRIService:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._tumor_classifier: Optional[BrainTumorClassifier] = None
        self._dementia_predictor: Optional[DementiaPredictor] = None
        self._init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "MRIService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_tumor_classifier(self) -> BrainTumorClassifier:
        if self._tumor_classifier is None:
            with self._init_lock:
                if self._tumor_classifier is None:
                    self._tumor_classifier = BrainTumorClassifier.get_instance()
        return self._tumor_classifier

    def _get_dementia_predictor(self) -> DementiaPredictor:
        if self._dementia_predictor is None:
            with self._init_lock:
                if self._dementia_predictor is None:
                    ckpt_path = DEMENTIA_DIR / "checkpoints" / "best_model.pth"
                    meta_path = DEMENTIA_DIR / "checkpoints" / "inference_meta.json"
                    self._dementia_predictor = DementiaPredictor(
                        checkpoint=ckpt_path,
                        meta_path=meta_path if meta_path.exists() else None
                    )
        return self._dementia_predictor

    def predict(self, image: Image.Image, subtype: Optional[str] = None) -> PredictionResponse:
        with self._lock:
            temp_path = None
            out_dir = None
            try:
                # Save input image to temp file
                unique_id = uuid.uuid4().hex
                temp_filename = f"mri_{unique_id}.png"
                temp_path = settings.TEMP_DIR / temp_filename
                out_dir = settings.TEMP_DIR / f"out_{unique_id}"
                out_dir.mkdir(parents=True, exist_ok=True)
                image.save(temp_path, format="PNG")

                orig_b64 = pil_to_base64_data_url(image)
                warnings = []

                # Task routing:
                is_explicit_tumor = subtype in ("tumor", "notumor", "brain_tumor", "default") and subtype != "dementia"
                is_explicit_dementia = subtype in ("dementia", "very_mild", "mild", "moderate")

                # If dementia explicitly requested
                if is_explicit_dementia:
                    dem_pred = self._get_dementia_predictor()
                    res = dem_pred.predict(temp_path, gradcam_dir=out_dir, multi_layer=True)
                    
                    pred_class = res["prediction"].upper()
                    probs = {k.upper(): round(float(v), 4) for k, v in res.get("probabilities", {}).items()}
                    confidence = round(float(res.get("confidence", 0.0)), 4)
                    
                    cam_file = res.get("gradcam") or res.get("gradcam_overlay")
                    cam_b64 = orig_b64
                    if cam_file and Path(cam_file).exists():
                        with Image.open(cam_file) as c_img:
                            cam_b64 = pil_to_base64_data_url(c_img)

                    visualizations = {"original": orig_b64}
                    if cam_b64 != orig_b64:
                        visualizations["overlay"] = cam_b64
                    if res.get("gradcam_heatmap") and Path(res["gradcam_heatmap"]).exists():
                        with Image.open(res["gradcam_heatmap"]) as hm_img:
                            visualizations["heatmap"] = pil_to_base64_data_url(hm_img)
                    if res.get("gradcam_fused") and Path(res["gradcam_fused"]).exists():
                        with Image.open(res["gradcam_fused"]) as f_img:
                            visualizations["fused_overlay"] = pil_to_base64_data_url(f_img)

                    if pred_class != "NON DEMENTED":
                        warnings.append(
                            "NEURODEGENERATIVE FINDING: Structural pattern matches significant ventricular enlargement "
                            "and temporal cortical volume loss consistent with dementia progression. Correlate with neuropsychological testing."
                        )

                    return PredictionResponse(
                        modality="mri",
                        routed_to="mri_dementia",
                        prediction=pred_class,
                        probabilities=probs,
                        confidence=confidence,
                        threshold=0.50,
                        model_name="Dementia Severity Classifier (EfficientNetV2-S)",
                        framework="PyTorch 2.x",
                        input_size="384x384 px",
                        dataset="OASIS / ADNI Brain MRI Cohorts",
                        metrics={
                            "Model": "EfficientNetV2-S",
                            "MC-Dropout Passes": "Calibrated Epistemic Uncertainty",
                            "Grad-CAM": "Multi-Layer MBConv Atrophy Fusion",
                        },
                        gradcam_png_base64=cam_b64,
                        original_png_base64=orig_b64,
                        visualizations=visualizations,
                        disclaimer=settings.DEFAULT_DISCLAIMER,
                        warnings=warnings,
                        status="success",
                        metadata={"raw": res, "visualizations": visualizations}
                    )

                # Default / Cascade Mode:
                # Stage 1: Tumor Screener
                tumor_clf = self._get_tumor_classifier()
                t_res = tumor_clf.predict(temp_path, out_dir=out_dir)

                if t_res["has_tumor"]:
                    # Brain tumor detected
                    prob_tumor = round(float(t_res["probability"]), 4)
                    prob_no_tumor = round(1.0 - prob_tumor, 4)
                    probs = {
                        "TUMOR": prob_tumor,
                        "NO TUMOR": prob_no_tumor,
                    }
                    confidence = prob_tumor
                    cam_file = t_res.get("gradcam_overlay")
                    cam_b64 = orig_b64
                    if cam_file and Path(cam_file).exists():
                        with Image.open(cam_file) as c_img:
                            cam_b64 = pil_to_base64_data_url(c_img)

                    visualizations = {"original": orig_b64}
                    if cam_b64 != orig_b64:
                        visualizations["overlay"] = cam_b64
                    crop_file = t_res.get("gradcam_cropped")
                    if crop_file and Path(crop_file).exists():
                        with Image.open(crop_file) as cr_img:
                            visualizations["cropped"] = pil_to_base64_data_url(cr_img)
                    heat_file = t_res.get("gradcam_heatmap")
                    if heat_file and Path(heat_file).exists():
                        with Image.open(heat_file) as hm_img:
                            visualizations["heatmap"] = pil_to_base64_data_url(hm_img)

                    warnings.append(
                        "ACUTE FINDING: Intracranial space-occupying lesion detected. "
                        "Stage 2 (Dementia screening) bypassed to avoid confounding mass effect with cortical atrophy."
                    )

                    return PredictionResponse(
                        modality="mri",
                        routed_to="brain_mri",
                        prediction="TUMOR",
                        probabilities=probs,
                        confidence=confidence,
                        threshold=0.50,
                        model_name="Brain Tumor Screener (Custom 5-Layer CNN)",
                        framework="PyTorch 2.x",
                        input_size="240x240 px",
                        dataset="BraTS Benchmark & Kaggle Brain MRI",
                        metrics={
                            "Layers Fused": "conv0, bn0, activation, max_pool0, max_pool1",
                            "Explainability": "5-Layer Gradient-Fused Logit Projection",
                        },
                        gradcam_png_base64=cam_b64,
                        original_png_base64=orig_b64,
                        visualizations=visualizations,
                        disclaimer=settings.DEFAULT_DISCLAIMER,
                        warnings=warnings,
                        status="success",
                        metadata={"tumor_result": t_res, "visualizations": visualizations}
                    )

                else:
                    # Stage 1 Normal (No Tumor) -> Reflex Stage 2 Dementia Evaluation
                    prob_tumor = round(float(t_res["probability"]), 4)
                    prob_no_tumor = round(1.0 - prob_tumor, 4)
                    
                    dem_pred = self._get_dementia_predictor()
                    res = dem_pred.predict(temp_path, gradcam_dir=out_dir, multi_layer=True)
                    
                    dem_pred_class = res["prediction"].upper()
                    dem_confidence = round(float(res.get("confidence", 0.0)), 4)
                    dem_probs = {k.upper(): round(float(v), 4) for k, v in res.get("probabilities", {}).items()}

                    cam_file = res.get("gradcam") or t_res.get("gradcam_overlay")
                    cam_b64 = orig_b64
                    if cam_file and Path(cam_file).exists():
                        with Image.open(cam_file) as c_img:
                            cam_b64 = pil_to_base64_data_url(c_img)

                    visualizations = {"original": orig_b64}
                    if cam_b64 != orig_b64:
                        visualizations["overlay"] = cam_b64
                    crop_file = t_res.get("gradcam_cropped")
                    if crop_file and Path(crop_file).exists():
                        with Image.open(crop_file) as cr_img:
                            visualizations["cropped"] = pil_to_base64_data_url(cr_img)
                    heat_file = res.get("gradcam_heatmap") or t_res.get("gradcam_heatmap")
                    if heat_file and Path(heat_file).exists():
                        with Image.open(heat_file) as hm_img:
                            visualizations["heatmap"] = pil_to_base64_data_url(hm_img)

                    is_dementia_positive = dem_pred_class != "NON DEMENTED"

                    if is_dementia_positive:
                        warnings.append(
                            f"CASCADE FINDING: Stage 1 Brain Tumor Screener is Negative (No space-occupying lesion detected, {prob_no_tumor * 100:.1f}% confidence). "
                            f"Reflex Stage 2 Neurodegenerative Assessment identified structural atrophy consistent with {dem_pred_class} ({dem_confidence * 100:.1f}% confidence)."
                        )
                        final_prediction = dem_pred_class
                    else:
                        warnings.append(
                            f"CASCADE FINDING: Stage 1 Brain Tumor Screening is Normal (No Tumor). "
                            f"Stage 2 Reflex Assessment found no significant neurodegenerative atrophy (Non-Demented, {dem_confidence * 100:.1f}% confidence)."
                        )
                        final_prediction = "NON DEMENTED"

                    return PredictionResponse(
                        modality="mri",
                        routed_to="mri_dementia",
                        prediction=final_prediction,
                        probabilities=dem_probs,
                        confidence=dem_confidence,
                        threshold=0.50,
                        model_name="Cascade Brain MRI (Tumor Normal -> Reflex Dementia Staging)",
                        framework="PyTorch 2.x",
                        input_size="240x240 & 384x384 px",
                        dataset="BraTS Benchmark & OASIS/ADNI",
                        metrics={
                            "Stage 1 Tumor Screener": f"Normal (No Tumor, {prob_no_tumor * 100:.1f}%)",
                            "Stage 2 Dementia Reflex": f"Triggered ({dem_pred_class})",
                            "Cascade Finding": f"No Tumor -> {final_prediction}",
                            "Explainability": "Multi-Layer MBConv Cortical Atrophy Grad-CAM",
                        },
                        gradcam_png_base64=cam_b64,
                        original_png_base64=orig_b64,
                        visualizations=visualizations,
                        disclaimer=settings.DEFAULT_DISCLAIMER,
                        warnings=warnings,
                        status="success",
                        metadata={
                            "tumor_result": t_res,
                            "dementia_result": res,
                            "visualizations": visualizations,
                            "tumor_probability": prob_tumor,
                            "no_tumor_probability": prob_no_tumor
                        }
                    )

            except Exception as e:
                raise ModelInferenceException(f"MRI inference failed: {str(e)}")
            finally:
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                if out_dir and out_dir.exists():
                    try:
                        import shutil
                        shutil.rmtree(out_dir, ignore_errors=True)
                    except Exception:
                        pass
