"""Brain CT Expert Model Adapter (Model 3).
Uses the trained SE-ResNeXt-101 model from RSNA2019_Intracranial-Hemorrhage-Detection to detect:
  - Any Hemorrhage
  - Subtypes: Epidural, Intraparenchymal, Intraventricular, Subarachnoid, Subdural
"""

import sys
import os
from pathlib import Path
from typing import Union, Dict, Any, Optional

import numpy as np
import cv2
import torch
from PIL import Image

CUR_DIR = Path(__file__).resolve().parent
BASE_DIR = CUR_DIR.parent
WORKSPACE_ROOT = BASE_DIR.parent if (BASE_DIR.parent / "RSNA2019_Intracranial-Hemorrhage-Detection").exists() else BASE_DIR
RSNA_DIR = WORKSPACE_ROOT / "RSNA2019_Intracranial-Hemorrhage-Detection"

# Add RSNA repo to sys.path so we can import its native modules
if str(RSNA_DIR) not in sys.path:
    sys.path.insert(0, str(RSNA_DIR))

# Possible checkpoint paths
CANDIDATE_CHECKPOINTS = [
    BASE_DIR / "models" / "brain" / "model_epoch_best_4.pth",
    BASE_DIR / "checkpoints" / "model_epoch_best_4.pth",
    WORKSPACE_ROOT / "models" / "brain" / "model_epoch_best_4.pth",
    WORKSPACE_ROOT / "ct_anatomy_project" / "models" / "brain" / "model_epoch_best_4.pth",
    WORKSPACE_ROOT / "ct_anatomy_project" / "checkpoints" / "model_epoch_best_4.pth",
    RSNA_DIR / "model_epoch_best_4.pth",
]

LABELS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
MEAN = np.array([0.456, 0.456, 0.456], dtype=np.float32)
STD = np.array([0.224, 0.224, 0.224], dtype=np.float32)


class BrainExpertModel:
    """Inference wrapper for Brain CT Intracranial Hemorrhage Expert (Model 3)."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        self.checkpoint_path = None
        if model_path:
            p = Path(model_path)
            if p.exists():
                self.checkpoint_path = p

        if self.checkpoint_path is None:
            for cand in CANDIDATE_CHECKPOINTS:
                if cand.exists():
                    self.checkpoint_path = cand
                    break

        self.device = device if device else None
        self.predictor = None
        self.is_loaded = False
        self.labels = list(LABELS)

        if self.checkpoint_path:
            self._load_model()

    def _load_model(self):
        try:
            from rsna_ich_predict_gradcam import ICHPredictor
            self.predictor = ICHPredictor(
                model_name="se_resnext101_32x4d",
                checkpoint=str(self.checkpoint_path),
                device=self.device,
            )
            self.is_loaded = True
        except Exception as e:
            print(f"[BrainExpertModel] Failed to load RSNA weights from {self.checkpoint_path}: {e}")
            self.is_loaded = False

    def predict(self, image_input: Union[str, Path, Image.Image], safe_threshold: float = 0.50) -> Dict[str, Any]:
        """Runs fast Brain CT forward inference to detect Intracranial Hemorrhage & subtypes."""
        if not self.is_loaded or self.predictor is None:
            return {
                "expert": "Brain CT Expert (Model 3)",
                "status": "NOT_LOADED",
                "message": "Brain model checkpoint not found. Ensure model_epoch_best_4.pth is present.",
            }

        image_path_str = str(image_input) if isinstance(image_input, (str, Path)) else "PIL.Image"

        try:
            # Load and normalize image for SE-ResNeXt
            if isinstance(image_input, (str, Path)):
                img = Image.open(str(image_input)).convert("RGB")
            elif isinstance(image_input, Image.Image):
                img = image_input.convert("RGB")
            else:
                img = Image.fromarray(image_input).convert("RGB")

            im_arr = np.array(img).astype(np.float32) / 255.0
            im_arr = cv2.resize(im_arr, (256, 256), interpolation=cv2.INTER_AREA)
            x_norm = (im_arr - MEAN) / STD
            x_tensor = torch.from_numpy(x_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.predictor.device)

            # Fast forward inference
            with torch.no_grad():
                logits = self.predictor.model(x_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            prob_dict = {lab: round(float(p), 6) for lab, p in zip(LABELS, probs)}
            any_p = prob_dict.get("any", 0.0)

            # Subtype risk calculation matching native rsna_ich_predict_gradcam.py
            subtypes = {k: v for k, v in prob_dict.items() if k != "any"}
            max_subtype, max_subtype_p = max(subtypes.items(), key=lambda item: item[1])
            overall_p = max(any_p, max_subtype_p)

            is_safe = overall_p < safe_threshold

            if is_safe:
                diag_title = f"[ SAFE ] - Brain is Safe / No Hemorrhage Detected  |  Normal Confidence: {(1.0 - overall_p) * 100:.2f}%"
                diagnosis = "No Acute Intracranial Hemorrhage Detected"
                summary = "Scan demonstrates no significant evidence of acute intracranial hemorrhage."
                confidence = 1.0 - overall_p
            else:
                diag_title = f"[ ALERT ] - Hemorrhage Detected: {max_subtype.upper()}  |  Risk Level: {overall_p * 100:.2f}%"
                diagnosis = f"Hemorrhage Detected ({max_subtype.capitalize()} dominant)"
                summary = f"Warning: Scan indicates elevated probability of acute intracranial hemorrhage ({max_subtype}: {max_subtype_p*100:.1f}%). Immediate radiologic review advised."
                confidence = overall_p

            return {
                "expert": "Brain CT Expert (Intracranial Hemorrhage Detection)",
                "status": "SUCCESS",
                "model_name": "se_resnext101_32x4d",
                "diagnosis": diagnosis,
                "diag_title": diag_title,
                "is_safe": is_safe,
                "overall_risk": round(overall_p, 4),
                "overall_risk_pct": f"{overall_p * 100:.2f}%",
                "target_class": max_subtype,
                "confidence": round(confidence, 4),
                "confidence_pct": f"{confidence * 100:.2f}%",
                "probabilities": prob_dict,
                "raw_probs": probs.tolist(),
                "clinical_summary": summary,
                "image_path": image_path_str,
            }
        except Exception as e:
            return {
                "expert": "Brain CT Expert (Model 3)",
                "status": "ERROR",
                "message": str(e),
                "image_path": image_path_str,
            }
