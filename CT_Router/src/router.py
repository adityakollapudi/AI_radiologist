"""CT Anatomy Router Model Definition & Inference Logic (Model 1).
Routes CT slices into CHEST_CT or BRAIN_CT.
"""

from pathlib import Path
from typing import Union, Dict, Any, Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Find default model relative to project root or ct_anatomy_project
CUR_DIR = Path(__file__).resolve().parent
BASE_DIR = CUR_DIR.parent
WORKSPACE_ROOT = BASE_DIR.parent if (BASE_DIR.parent / "RSNA2019_Intracranial-Hemorrhage-Detection").exists() else BASE_DIR

CANDIDATE_ROUTER_PATHS = [
    BASE_DIR / "models" / "router" / "best_router.pt",
    BASE_DIR / "checkpoints" / "best_router.pt",
    WORKSPACE_ROOT / "models" / "router" / "best_router.pt",
    WORKSPACE_ROOT / "ct_anatomy_project" / "models" / "router" / "best_router.pt",
    WORKSPACE_ROOT / "ct_anatomy_project" / "checkpoints" / "best_router.pt",
]
DEFAULT_ROUTER_PATH = next((p for p in CANDIDATE_ROUTER_PATHS if p.exists()), CANDIDATE_ROUTER_PATHS[0])

# ImageNet normalization used during training
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (256, 256)


def build_router_model() -> nn.Module:
    """Builds EfficientNet-B0 with custom 2-class classification head."""
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),
    )
    return model


class CTAnatomyRouter:
    """Inference engine for CT Anatomy Router (Model 1)."""

    def __init__(self, model_path: Union[str, Path] = DEFAULT_ROUTER_PATH, device: Optional[str] = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            for cand in CANDIDATE_ROUTER_PATHS:
                if cand.exists():
                    self.model_path = cand
                    break
            else:
                raise FileNotFoundError(f"Router checkpoint not found at: {self.model_path}")

        # Choose hardware device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load weights and checkpoint metadata
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location=self.device)
        self.class_names = checkpoint.get("class_names", ["CHEST_CT", "BRAIN_CT"])
        self.threshold = checkpoint.get("threshold", 0.5)
        self.uncertain_band = checkpoint.get("uncertain_band", 0.45)

        # Initialize network and load state dictionary
        self.model = build_router_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])

    def preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Loads and pre-processes an image into a normalized 4D tensor (1, 3, 256, 256)."""
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file does not exist: {img_path}")

            # Check if DICOM file
            if img_path.suffix.lower() == ".dcm":
                try:
                    import pydicom
                    dcm = pydicom.dcmread(str(img_path))
                    arr = dcm.pixel_array.astype(float)
                    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
                    image = Image.fromarray(arr.astype("uint8")).convert("RGB")
                except ImportError:
                    raise ImportError("Reading .dcm files requires 'pydicom'. Run: pip install pydicom")
            else:
                image = Image.open(str(img_path)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """Runs router inference on a single image."""
        tensor = self.preprocess_image(image_input)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        prob_chest = float(probs[0])
        prob_brain = float(probs[1])

        # Classification based on calibrated threshold (default 0.5)
        if prob_brain >= self.threshold:
            predicted_class = "BRAIN_CT"
            confidence = prob_brain
        else:
            predicted_class = "CHEST_CT"
            confidence = prob_chest

        # Uncertainty check: |P(BRAIN) - threshold| < uncertain_band
        is_uncertain = abs(prob_brain - self.threshold) < self.uncertain_band
        status = "UNCERTAIN (Review Needed)" if is_uncertain else "ROUTED"

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "confidence_pct": f"{confidence * 100:.2f}%",
            "probabilities": {
                "CHEST_CT": round(prob_chest, 6),
                "BRAIN_CT": round(prob_brain, 6),
            },
            "status": status,
            "uncertain": is_uncertain,
            "image_path": str(image_input) if isinstance(image_input, (str, Path)) else "PIL.Image",
        }
