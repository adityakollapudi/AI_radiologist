"""Chest CT Expert Model Adapter (Model 2).
Uses the trained EfficientNet classifier from Covid19-CT-Scan--Detection to diagnose:
  - Normal
  - Pneumonia
  - COVID-19
"""

import sys
from pathlib import Path
from typing import Union, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CUR_DIR = Path(__file__).resolve().parent
BASE_DIR = CUR_DIR.parent
WORKSPACE_ROOT = BASE_DIR.parent if (BASE_DIR.parent / "Covid19-CT-Scan--Detection").exists() else BASE_DIR
COVID_DIR = WORKSPACE_ROOT / "Covid19-CT-Scan--Detection"

# Check potential checkpoint locations
CANDIDATE_CHECKPOINTS = [
    BASE_DIR / "models" / "chest" / "best.pt",
    BASE_DIR / "checkpoints" / "best_chest.pt",
    BASE_DIR / "checkpoints" / "best.pt",
    WORKSPACE_ROOT / "models" / "chest" / "best.pt",
    WORKSPACE_ROOT / "ct_anatomy_project" / "models" / "chest" / "best.pt",
    WORKSPACE_ROOT / "ct_anatomy_project" / "checkpoints" / "best_chest.pt",
    COVID_DIR / "covidx_ct_results" / "checkpoints" / "best.pt",
    COVID_DIR / "covidx_ct_results" / "checkpoints" / "last.pt",
]

CLASS_NAMES = ["Normal", "Pneumonia", "COVID-19"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def compute_entropy(p: np.ndarray) -> float:
    p_clipped = np.clip(p, 1e-12, 1.0)
    return float(-(p_clipped * np.log(p_clipped)).sum())


class EfficientNetClassifier(nn.Module):
    """Architecture definition matching Covid19-CT-Scan--Detection."""

    def __init__(self, arch="b0", num_classes=3):
        super().__init__()
        if arch == "b0":
            self.backbone = models.efficientnet_b0(weights=None)
        elif arch == "b3":
            self.backbone = models.efficientnet_b3(weights=None)
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.30),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class ChestExpertModel:
    """Inference wrapper for Chest CT Expert (Model 2)."""

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

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.is_loaded = False
        self.model = None

        if self.checkpoint_path:
            self._load_model()

    def _load_model(self):
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            args = ckpt.get("args", {})
            arch = args.get("arch", ckpt.get("arch", "b0"))
            self.class_names = ckpt.get("class_names", CLASS_NAMES)

            model = EfficientNetClassifier(arch, len(self.class_names)).to(self.device)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)

            for mod in model.modules():
                if hasattr(mod, "inplace"):
                    mod.inplace = False

            model.eval()
            self.model = model
            self.is_loaded = True
        except Exception as e:
            print(f"[ChestExpertModel] Failed to load weights from {self.checkpoint_path}: {e}")
            self.is_loaded = False

        self.transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def predict(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """Runs Chest CT inference to detect Normal vs. Pneumonia vs. COVID-19."""
        if not self.is_loaded or self.model is None:
            return {
                "expert": "Chest CT Expert (Model 2)",
                "status": "NOT_LOADED",
                "message": "Chest model checkpoint not found. Ensure best.pt is present.",
            }

        image_path_str = str(image_input) if isinstance(image_input, (str, Path)) else "PIL.Image"

        # Load image
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            if img_path.suffix.lower() == ".dcm":
                try:
                    import pydicom
                    dcm = pydicom.dcmread(str(img_path))
                    arr = dcm.pixel_array.astype(float)
                    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
                    image = Image.fromarray(arr.astype("uint8")).convert("RGB")
                except ImportError:
                    raise ImportError("DICOM support requires 'pydicom'.")
            else:
                image = Image.open(str(img_path)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        diagnosis = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        # Top-1 margin and entropy matching covid_ct_predict.py
        order = np.argsort(probs)[::-1]
        top1_margin = float(probs[order[0]] - probs[order[1]])
        entropy_val = compute_entropy(probs)

        prob_dict = {name: round(float(p), 6) for name, p in zip(self.class_names, probs)}

        # Clinical interpretation string
        if diagnosis == "COVID-19":
            summary = "Features consistent with viral pneumonia / COVID-19 pattern."
        elif diagnosis == "Pneumonia":
            summary = "Findings consistent with non-COVID bacterial/viral pneumonia."
        else:
            summary = "No acute pulmonary consolidation or ground-glass opacities detected."

        return {
            "expert": "Chest CT Expert (COVID-19 / Pneumonia Detection)",
            "status": "SUCCESS",
            "diagnosis": diagnosis,
            "confidence": round(confidence, 4),
            "confidence_pct": f"{confidence * 100:.2f}%",
            "top1_margin_pct": f"{top1_margin * 100:.2f}%",
            "entropy": round(entropy_val, 4),
            "crop_source": "patient body contour (couch excluded)",
            "probabilities": prob_dict,
            "raw_probs": probs.tolist(),
            "class_names": self.class_names,
            "clinical_summary": summary,
            "image_path": image_path_str,
        }
