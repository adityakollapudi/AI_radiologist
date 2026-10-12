#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
router/router_model.py
======================
Architecture, preprocessing and checkpoint handling for the X-ray body-region
ROUTER used by pipeline.py.

The router answers exactly one question:

    "Which existing branch should process this image -- CHEST or MUSCULOSKELETAL?"

It does NOT predict pneumonia, fracture, normal or abnormal. Those belong to the
downstream models (Pneumonia_Detection and Foreign_Object_Detection).

DESIGN DECISIONS
----------------
Architecture : EfficientNet-B0 (5.3M params, 224x224). Deliberately lighter than
               the two EfficientNet-B3 disease models (10.7M each at 300x300).
               Distinguishing a thorax from a limb is a coarse, global shape task
               -- the silhouette alone is nearly sufficient -- so the capacity and
               resolution needed for subtle consolidation or a cortical step-off
               is wasted here. B0 at 224 also keeps the router's cost negligible
               relative to the downstream model it gates.

Output design: TWO logits + CrossEntropyLoss, softmax at inference.
               Chosen over a single BCE logit because routing needs a calibrated
               per-class confidence to compare against ROUTER_THRESHOLD, and
               softmax gives both class probabilities directly. This choice is
               applied consistently in train_router.py and here.

Preprocessing: matches the downstream convention -- convert("L") then replicate
               to 3 channels, ImageNet normalisation -- so one loaded PIL image
               can feed the router and then the downstream model without being
               re-encoded or re-read from disk.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.models import efficientnet_b0

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


# ==============================================================================
# ROUTER CONSTANTS
# ==============================================================================

CLASS_NAMES = {0: "CHEST", 1: "MUSCULOSKELETAL"}
CLASS_TO_INDEX = {"CHEST": 0, "MUSCULOSKELETAL": 1}
NUM_CLASSES = 2

ROUTER_IMAGE_SIZE = 224
ROUTER_MEAN = [0.485, 0.456, 0.406]
ROUTER_STD = [0.229, 0.224, 0.225]
ROUTER_DROPOUT = 0.2

# Below this max-softmax confidence the pipeline refuses to route automatically.
ROUTER_THRESHOLD = 0.80

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ==============================================================================
# MODEL
# ==============================================================================

class XRayRouter(nn.Module):
    """EfficientNet-B0 with a 2-class head.

    The backbone is stored as `self.backbone`, so every checkpoint key is
    prefixed "backbone." -- the same convention the two downstream models use.
    """

    def __init__(self, pretrained=True, dropout_p=ROUTER_DROPOUT,
                 num_classes=NUM_CLASSES):
        super().__init__()
        weights = None
        if pretrained:
            try:
                from torchvision.models import EfficientNet_B0_Weights
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            except Exception as e:
                print("[router] could not resolve pretrained weights ->", repr(e))

        try:
            self.backbone = efficientnet_b0(weights=weights)
            self.pretrained = weights is not None
        except Exception as e:
            print("[router] pretrained download failed ->", repr(e))
            print("[router] falling back to random init")
            self.backbone = efficientnet_b0(weights=None)
            self.pretrained = False

        in_features = self.backbone.classifier[1].in_features      # 1280 for B0
        self.in_features = in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        nn.init.zeros_(self.backbone.classifier[1].bias)
        nn.init.normal_(self.backbone.classifier[1].weight, std=0.01)

    def forward(self, x):
        return self.backbone(x)        # (B, 2) raw logits -- NO softmax inside

    @property
    def features(self):
        return self.backbone.features


# ==============================================================================
# PREPROCESSING
# ==============================================================================

def build_router_transform(image_size=ROUTER_IMAGE_SIZE, mean=None, std=None,
                           train=False):
    """Router transforms.

    Augmentation is conservative and anatomy-preserving. Note one deliberate
    difference from the downstream models: a horizontal flip IS acceptable here.
    The router's job is body-region identification, and a mirrored chest film is
    still unambiguously a chest film. The downstream pneumonia model must not
    flip (the thorax is not left-right symmetric and laterality matters for a
    finding), but that constraint does not apply to shape-level routing.
    """
    mean = mean or ROUTER_MEAN
    std = std or ROUTER_STD

    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.06, 0.06),
                scale=(0.92, 1.08),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.20, contrast=0.20),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # Deterministic -- validation, test, and every inference call.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_grayscale(image_path):
    """Open an image safely and return a single-channel PIL image.

    convert("L") first collapses any colour, matching what both downstream
    models do. The caller can then reuse this one PIL object for the router and
    the downstream model -- no second disk read, no re-encoding.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found:\n  {image_path}")
    try:
        with Image.open(image_path) as im:
            mode, size = im.mode, im.size
            gray = im.convert("L").copy()
    except Exception as e:
        raise RuntimeError(f"Could not decode the image file: {e}")
    return gray, mode, size


# ==============================================================================
# DATASET (module level -- Windows DataLoader workers use spawn and must pickle it)
# ==============================================================================

class RouterDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.paths = df["image_path"].values
        self.labels = df["label"].values.astype(np.int64)
        self.transform = transform
        self._warned = 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            with Image.open(path) as im:
                img = im.convert("L").copy()
        except Exception as e:
            if self._warned < 5:
                print(f"[router-dataset] unreadable: {path} ({e})")
                self._warned += 1
            img = Image.new("L", (ROUTER_IMAGE_SIZE, ROUTER_IMAGE_SIZE), 0)

        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long), idx


def router_seed_worker(worker_id):
    import random
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


# ==============================================================================
# CHECKPOINT
# ==============================================================================

def save_router_checkpoint(model, path, extra=None):
    """Self-sufficient checkpoint: everything inference needs lives inside it."""
    payload = {
        "model_state_dict": model.state_dict(),
        "model_name": "efficientnet_b0",
        "architecture": "torchvision.models.efficientnet_b0",
        "wrapper_class": "XRayRouter (backbone. prefix on all keys)",
        "role": "body-region router -- CHEST vs MUSCULOSKELETAL, NOT a disease model",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "class_to_index": CLASS_TO_INDEX,
        "image_size": ROUTER_IMAGE_SIZE,
        "normalization": {"mean": ROUTER_MEAN, "std": ROUTER_STD},
        "grayscale_handling": "PIL convert('L') then replicate to 3 channels",
        "dropout_p": ROUTER_DROPOUT,
        "loss": "CrossEntropyLoss",
        "activation": "softmax at inference only",
        "routing_threshold": ROUTER_THRESHOLD,
        "torch_version": torch.__version__,
        "disclaimer": ("Research/educational routing model. Decides which "
                       "downstream branch receives the image. It makes no "
                       "clinical claim of any kind."),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_router(checkpoint_path, device, verbose=True):
    """Rebuild the router from its checkpoint, with a verified weight load."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Router checkpoint not found:\n  {checkpoint_path}\n"
            "Train it first:  python router/train_router.py"
        )

    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck

    head_keys = [k for k in sd if "classifier" in k and k.endswith(".weight")]
    if not head_keys:
        raise RuntimeError("No classifier weights found in the router checkpoint.")
    n_out, n_in = sd[head_keys[-1]].shape
    if n_out != NUM_CLASSES:
        raise RuntimeError(
            f"Router checkpoint has {n_out} output neurons; this code expects "
            f"{NUM_CLASSES} (CHEST, MUSCULOSKELETAL)."
        )

    dropout_p = ck.get("dropout_p", ROUTER_DROPOUT) if isinstance(ck, dict) else ROUTER_DROPOUT
    model = XRayRouter(pretrained=False, dropout_p=dropout_p, num_classes=n_out)

    # Reconcile prefixes, then VERIFY. strict=False alone would turn a total key
    # mismatch into a silent no-op, leaving a random network that still returns
    # confident-looking probabilities.
    model_keys = set(model.state_dict().keys())
    if all(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    if not (set(sd) & model_keys):
        if all(("backbone." + k) in model_keys for k in list(sd)[:5]):
            sd = {"backbone." + k: v for k, v in sd.items()}
        elif all(k.startswith("backbone.") for k in sd):
            sd = {k[len("backbone."):]: v for k, v in sd.items()}

    res = model.load_state_dict(sd, strict=False)
    total = len(model.state_dict())
    loaded = total - len(res.missing_keys)
    if loaded == 0:
        raise RuntimeError(
            "NO router weights were loaded -- the model is randomly initialised "
            "and every routing decision would be meaningless."
        )
    if res.missing_keys:
        raise RuntimeError(
            f"{len(res.missing_keys)} router tensors did not load. Refusing to "
            "route with a partially initialised model. First missing key: "
            + res.missing_keys[0]
        )

    model.to(device).eval()

    cn = ck.get("class_names", CLASS_NAMES) if isinstance(ck, dict) else CLASS_NAMES
    cfg = {
        "class_names": {int(k): v for k, v in cn.items()},
        "image_size": ck.get("image_size", ROUTER_IMAGE_SIZE) if isinstance(ck, dict) else ROUTER_IMAGE_SIZE,
        "mean": (ck.get("normalization", {}) or {}).get("mean", ROUTER_MEAN) if isinstance(ck, dict) else ROUTER_MEAN,
        "std": (ck.get("normalization", {}) or {}).get("std", ROUTER_STD) if isinstance(ck, dict) else ROUTER_STD,
        "routing_threshold": ck.get("routing_threshold", ROUTER_THRESHOLD) if isinstance(ck, dict) else ROUTER_THRESHOLD,
        "model_name": ck.get("model_name", "efficientnet_b0") if isinstance(ck, dict) else "efficientnet_b0",
        "test_metrics": ck.get("test_metrics") if isinstance(ck, dict) else None,
        "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
        "n_train_images": ck.get("n_train_images") if isinstance(ck, dict) else None,
    }

    if verbose:
        print(f"[router] loaded {loaded}/{total} tensors from "
              f"{os.path.basename(checkpoint_path)}")
        print(f"[router] classes {cfg['class_names']} | input {cfg['image_size']} "
              f"| routing threshold {cfg['routing_threshold']:.2f}")
        if cfg.get("test_metrics"):
            tm = cfg["test_metrics"]
            print(f"[router] held-out accuracy at save time: "
                  f"{float(tm.get('accuracy', float('nan'))):.4f}")

    return model, cfg


# ==============================================================================
# ROUTING PREDICTION
# ==============================================================================

@torch.no_grad()
def predict_route(model, cfg, gray_image, device, transform=None):
    """Route ONE already-loaded grayscale PIL image.

    Takes a PIL image rather than a path on purpose: pipeline.py loads the file
    once and hands the same object to the router and then to the downstream
    model, so nothing is re-read or re-encoded between stages.

    Returns a dict with both class probabilities and the routing decision.
    """
    transform = transform or build_router_transform(
        cfg["image_size"], cfg["mean"], cfg["std"], train=False)

    x = transform(gray_image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x).float()
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    threshold = float(cfg.get("routing_threshold", ROUTER_THRESHOLD))

    return {
        "route": cfg["class_names"][idx],
        "route_index": idx,
        "confidence": round(confidence * 100, 2),
        "probabilities": {cfg["class_names"][i]: round(float(p) * 100, 2)
                          for i, p in enumerate(probs)},
        "logits": [round(float(v), 6) for v in logits[0].cpu().numpy()],
        "routing_threshold": threshold,
        "confident": confidence >= threshold,
    }
