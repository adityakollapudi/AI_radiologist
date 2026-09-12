#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pneumonia.py
=================
Single-image and batch inference + Grad-CAM for the model trained by
train_pneumonia.py.

Everything that can be read from the checkpoint IS read from the checkpoint --
class mapping, input size, normalisation statistics, letterbox flag and decision
threshold. None of it is re-declared here. That is deliberate: every value
re-typed in an inference script is a place where training and testing can drift
apart silently, and the resulting bug produces plausible-looking output rather
than an error.

Matches train_pneumonia.py exactly:
  Architecture   : torchvision efficientnet_b3 inside a PneumoniaEfficientNetB3
                   wrapper -> all checkpoint keys are prefixed "backbone."
  Head           : Dropout(p) -> Linear(1536, 1)
  Output         : ONE raw logit. No sigmoid inside the model.
  Loss (training): BCEWithLogitsLoss  -> activation at inference is sigmoid
  Class mapping  : 0 = NORMAL, 1 = PNEUMONIA
  Preprocessing  : convert("L") -> convert("RGB") -> letterbox resize to 300
                   -> ToTensor -> Normalize(ImageNet mean/std)
  Grad-CAM layer : backbone.features[-1]  (1536 ch, 10x10 at 300px input)

Usage (Windows):
    python test_pneumonia.py
    python test_pneumonia.py "C:\\path\\to\\xray.jpeg"
    python test_pneumonia.py --folder "C:\\path\\to\\folder"
    python test_pneumonia.py --evaluate "C:\\path\\to\\chest_xray\\test"

Research/educational use only. Outputs are model predictions, not diagnoses.
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b3

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_PATH = r"C:\Users\vemul\Downloads\MINI_Project_Aditya_pneumonia\pneumonia_checkpoints\pneumonia_efficientnet_b3_best.pth"

IMAGE_PATH = None          # or hard-code a path; CLI argument overrides this
OVERLAY_ALPHA = 0.42       # heatmap transparency
HEATMAP_CMAP = "jet"
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Fallbacks used ONLY if the checkpoint is missing the corresponding key.
# Each one is reported loudly when it fires.
FALLBACK_IMAGE_SIZE = 300
FALLBACK_MEAN = [0.485, 0.456, 0.406]
FALLBACK_STD = [0.229, 0.224, 0.225]
FALLBACK_CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


# ==============================================================================
# 1. DEVICE
# ==============================================================================

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("PNEUMONIA EfficientNet-B3 -- inference + Grad-CAM")
    print("=" * 76)
    print("PyTorch        :", torch.__version__)
    print("CUDA available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU            :", torch.cuda.get_device_name(0))
        print("GPU memory     : {:.2f} GB".format(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
    print("Device in use  :", device)
    return device


# ==============================================================================
# 2. PREPROCESSING -- identical to training
# ==============================================================================

class LetterboxResize:
    """Pad to square, then resize. Same class as in train_pneumonia.py.

    Must be reproduced exactly. If training used letterbox padding and inference
    used a plain squashing resize, every image arrives with a different aspect
    distortion than the model was fitted on -- nothing crashes, accuracy just
    quietly degrades.
    """

    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        side = max(w, h)
        canvas = Image.new(img.mode, (side, side), self.fill)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas.resize((self.size, self.size), Image.BILINEAR)

    def __repr__(self):
        return f"LetterboxResize(size={self.size}, fill={self.fill})"


def build_transform(cfg):
    resize = (LetterboxResize(cfg["image_size"]) if cfg["letterbox"]
              else transforms.Resize((cfg["image_size"], cfg["image_size"])))
    return transforms.Compose([
        resize,
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


def load_image(image_path, transform):
    """Returns (original grayscale PIL, model input tensor, mode, size)."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"\nImage not found:\n  {image_path}")
    try:
        with Image.open(image_path) as im:
            mode, size = im.mode, im.size
            gray = im.convert("L").copy()
    except Exception as e:
        raise RuntimeError(f"Could not decode the image file: {e}")

    # Training did convert("L") FIRST, collapsing any colour, then convert("RGB")
    # to replicate that single channel. Going straight to RGB is not equivalent
    # for the ~5% of files in this dataset stored as RGB.
    tensor = transform(gray.convert("RGB")).unsqueeze(0)
    return gray, tensor, mode, size


# ==============================================================================
# 3. MODEL -- must match the training wrapper byte for byte
# ==============================================================================

class PneumoniaEfficientNetB3(nn.Module):
    """Keeping the attribute name `backbone` is what makes the checkpoint keys
    line up. A bare torchvision efficientnet_b3 shares ZERO keys with this
    checkpoint, and load_state_dict(..., strict=False) would turn that total
    mismatch into a silent no-op -- leaving a randomly initialised network that
    returns confident nonsense on every image.
    """

    def __init__(self, dropout_p=0.4):
        super().__init__()
        self.backbone = efficientnet_b3(weights=None)
        in_features = self.backbone.classifier[1].in_features   # 1536
        self.in_features = in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)      # (B,) raw logits, NO sigmoid

    @property
    def gradcam_target_layer(self):
        return self.backbone.features[-1]


def extract_state_dict(ck):
    if not isinstance(ck, dict):
        return ck, "raw state_dict object"
    for key in ("model_state_dict", "state_dict", "model"):
        if key in ck and isinstance(ck[key], dict):
            return ck[key], f'checkpoint["{key}"]'
    if all(isinstance(v, torch.Tensor) for v in ck.values()):
        return ck, "checkpoint is itself a state_dict"
    raise RuntimeError("Could not locate weights inside the checkpoint.")


def align_prefixes(sd, model):
    """Reconcile key prefixes. Never silently accepts a mismatch -- the caller
    verifies the load and aborts if it did not actually happen."""
    model_keys = set(model.state_dict().keys())

    if all(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}
        print('  stripped "module." prefix (DataParallel checkpoint)')

    if set(sd) & model_keys:
        return sd, "keys already aligned"
    if all(("backbone." + k) in model_keys for k in list(sd)[:5]):
        return {"backbone." + k: v for k, v in sd.items()}, 'added "backbone." prefix'
    if all(k.startswith("backbone.") for k in sd):
        return ({k[len("backbone."):]: v for k, v in sd.items()},
                'stripped "backbone." prefix')
    return sd, "no prefix transformation applied"


def load_model(model_path, device):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n  {model_path}\n"
            "Edit MODEL_PATH at the top of this file, or pass --model <path>."
        )

    print("\n" + "=" * 76)
    print("CHECKPOINT INSPECTION")
    print("=" * 76)
    print("File :", model_path)
    print("Size : {:.1f} MB".format(os.path.getsize(model_path) / 1024 ** 2))

    ck = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ck, dict):
        print("\nMetadata stored in the checkpoint:")
        for k in sorted(k for k in ck if k != "model_state_dict"):
            v = ck[k]
            if isinstance(v, dict):
                short = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                         for kk, vv in list(v.items())[:9]}
                print(f"  {k:<22}: {short}")
            else:
                print(f"  {k:<22}: {v}")

    sd, where = extract_state_dict(ck)
    print(f"\nWeights from            : {where}")
    print(f"Tensors in state_dict   : {len(sd)}")

    head_keys = [k for k in sd if "classifier" in k and k.endswith(".weight")]
    if not head_keys:
        raise RuntimeError("No classifier weights found in the checkpoint.")
    head_key = head_keys[-1]
    num_outputs, head_in = sd[head_key].shape
    print(f"Classifier key          : {head_key}")
    print(f"Classifier shape        : [{num_outputs}, {head_in}]")

    if num_outputs != 1:
        raise RuntimeError(
            f"This checkpoint has {num_outputs} output neurons. train_pneumonia.py "
            "produces a single logit with BCEWithLogitsLoss, so this file did not "
            "come from that run. Do not interpret it with this script."
        )

    dropout_p = ck.get("dropout_p", 0.4) if isinstance(ck, dict) else 0.4
    model = PneumoniaEfficientNetB3(dropout_p=dropout_p)

    sd, note = align_prefixes(sd, model)
    print(f"Key alignment           : {note}")

    res = model.load_state_dict(sd, strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    total = len(model.state_dict())
    loaded = total - len(missing)

    print("\n" + "-" * 76)
    print("LOAD VERIFICATION")
    print("-" * 76)
    print(f"Missing keys    : {len(missing)}")
    print(f"Unexpected keys : {len(unexpected)}")
    for k in missing[:5]:
        print("   missing    ->", k)
    for k in unexpected[:5]:
        print("   unexpected ->", k)
    print(f"Loaded {loaded}/{total} tensors ({100 * loaded / total:.1f}%)")

    if loaded == 0:
        raise RuntimeError(
            "NO weights were loaded. The model is randomly initialised and every "
            "prediction from it would be meaningless. The checkpoint's key names "
            "do not match this architecture."
        )
    if missing:
        raise RuntimeError(
            f"{len(missing)} tensors did not load. Refusing to run inference on a "
            "partially initialised model -- the output would look plausible while "
            "being noise. First missing key: " + missing[0]
        )
    print("All tensors loaded. Weights verified.")

    model.to(device).eval()

    # ---- configuration read from the checkpoint, not assumed -----------------
    cfg, assumed = {}, []

    def get(key, fallback, label=None):
        if isinstance(ck, dict) and key in ck and ck[key] is not None:
            return ck[key]
        assumed.append(label or key)
        return fallback

    cfg["image_size"] = get("image_size", FALLBACK_IMAGE_SIZE)
    norm = ck.get("normalization") if isinstance(ck, dict) else None
    if norm:
        cfg["mean"], cfg["std"] = norm["mean"], norm["std"]
    else:
        cfg["mean"], cfg["std"] = FALLBACK_MEAN, FALLBACK_STD
        assumed.append("normalization")
    cfg["letterbox"] = get("letterbox", True)
    cn = get("class_names", FALLBACK_CLASS_NAMES)
    cfg["class_names"] = ({int(k): v for k, v in cn.items()}
                          if isinstance(cn, dict) else FALLBACK_CLASS_NAMES)
    cfg["grayscale_handling"] = get(
        "grayscale_handling", "convert('L') then replicate to 3 channels")
    cfg["loss"] = get("loss", "BCEWithLogitsLoss")
    cfg["model_name"] = get("model_name", "efficientnet_b3")
    cfg["stage"] = ck.get("stage") if isinstance(ck, dict) else None
    cfg["epoch"] = ck.get("epoch") if isinstance(ck, dict) else None
    cfg["val_metrics"] = ck.get("val_metrics") if isinstance(ck, dict) else None
    cfg["test_metrics"] = ck.get("test_metrics") if isinstance(ck, dict) else None
    cfg["num_outputs"] = num_outputs
    cfg["head_in_features"] = head_in

    # ---- threshold: use the trained one; never invent one --------------------
    thr = ck.get("threshold") if isinstance(ck, dict) else None
    if thr is None:
        cfg["threshold"] = 0.5
        cfg["threshold_source"] = ("NOT PRESENT in checkpoint -> falling back to "
                                   "0.5 (sigmoid default)")
        assumed.append("threshold")
    else:
        cfg["threshold"] = float(thr)
        crit = ck.get("threshold_criterion", "unknown")
        cfg["threshold_source"] = f"validation-selected during training (criterion: {crit})"
    cfg["threshold_rationale"] = ck.get("threshold_rationale") if isinstance(ck, dict) else None

    if assumed:
        print("\n  [warn] the checkpoint did not store: " + ", ".join(assumed))
        print("         Falling back to defaults for those. If training used "
              "different values, predictions will be wrong in a way that does "
              "not raise an error.")

    return model, cfg, ck


def report_provenance(cfg):
    print("\n" + "=" * 76)
    print("TRAINING PROVENANCE OF THIS CHECKPOINT")
    print("=" * 76)
    print(f"Stage : {cfg.get('stage')}    Epoch : {cfg.get('epoch')}")
    for name, key in (("Validation", "val_metrics"), ("Test", "test_metrics")):
        m = cfg.get(key)
        if m:
            print(f"{name} metrics recorded at save time:")
            for k in ("accuracy", "precision", "recall", "specificity",
                      "f1", "roc_auc", "pr_auc"):
                if k in m:
                    print(f"   {k:<12}: {float(m[k]):.4f}")
    if cfg.get("stage") == 1:
        print("\n  >>> This is a STAGE 1 checkpoint: the backbone was FROZEN and only")
        print("      the classification head was trained. The convolutional features")
        print("      are still plain ImageNet features, never adapted to radiographs.")
        print("      Stage 2 fine-tuning did not complete. Expect weak predictions --")
        print("      that is the model, not this script.")
    print(f"\nThreshold in use : {cfg['threshold']:.4f}")
    print(f"Threshold source : {cfg['threshold_source']}")
    if cfg.get("threshold_rationale"):
        print(f"Rationale        : {cfg['threshold_rationale']}")


# ==============================================================================
# 4. GRAD-CAM
# ==============================================================================

class GradCAM:
    """Grad-CAM for a single-logit model. float32, device-agnostic, hook-safe."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.layer = target_layer or model.gradcam_target_layer
        self.acts = None
        self.grads = None
        self.handles = []

    def __enter__(self):
        self.handles.append(self.layer.register_forward_hook(
            lambda m, i, o: setattr(self, "acts", o.detach())))
        self.handles.append(self.layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grads", go[0].detach())))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False

    def generate(self, x, device, signed=1.0, output_size=None):
        """signed=+1 explains evidence FOR pneumonia, -1 explains evidence FOR
        normal. Runs in float32 with autocast off: half-precision gradients
        through the hook can underflow to zero and produce a blank map."""
        self.model.eval()
        x = x.to(device).float().requires_grad_(True)

        with torch.enable_grad():
            logit = self.model(x)
            self.model.zero_grad(set_to_none=True)
            (signed * logit).sum().backward()

        if self.acts is None or self.grads is None:
            raise RuntimeError("Grad-CAM hooks captured nothing -- wrong target layer?")

        a, g = self.acts[0], self.grads[0]
        alpha = g.mean(dim=(1, 2), keepdim=True)       # channel importance
        cam = F.relu((alpha * a).sum(dim=0))           # keep supporting evidence only
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else torch.zeros_like(cam)

        size = output_size or (x.shape[-2], x.shape[-1])
        cam = F.interpolate(cam[None, None], size=size,
                            mode="bilinear", align_corners=False)[0, 0]
        return cam.detach().cpu().numpy(), float(logit.detach().item()), tuple(a.shape)


# ==============================================================================
# 5. SINGLE-IMAGE PREDICTION
# ==============================================================================

def predict(image_path, model, cfg, device, transform,
            show=True, save_dir=None, verbose=True):
    gray, x, orig_mode, orig_size = load_image(image_path, transform)
    gray_np = np.asarray(gray, dtype=np.float32) / 255.0

    model.eval()
    with torch.no_grad():
        logit = model(x.to(device)).float()
        p_pneu = torch.sigmoid(logit).item()
    p_norm = 1.0 - p_pneu

    threshold = cfg["threshold"]
    idx = int(p_pneu >= threshold)
    pred = cfg["class_names"][idx]
    conf = p_pneu if idx == 1 else p_norm

    result = {
        "image_path": os.path.abspath(str(image_path)),
        "prediction": pred,
        "predicted_class_index": idx,
        "normal_probability": round(p_norm * 100, 2),
        "pneumonia_probability": round(p_pneu * 100, 2),
        "confidence": round(conf * 100, 2),
        "logit": round(float(logit.item()), 6),
        "threshold": round(threshold, 6),
        "class_mapping": {str(k): v for k, v in cfg["class_names"].items()},
        "model_name": cfg["model_name"],
        "image_size": cfg["image_size"],
        "original_size": [orig_size[0], orig_size[1]],
        "disclaimer": ("Research/educational model output. Not a clinical "
                       "diagnosis. Grad-CAM shows model-attributed importance, "
                       "not lesion segmentation."),
    }

    # ---- Grad-CAM at the ORIGINAL resolution ---------------------------------
    signed = 1.0 if idx == 1 else -1.0
    with GradCAM(model) as engine:
        cam, _, feat_shape = engine.generate(x, device, signed,
                                             output_size=gray_np.shape)

    if verbose:
        print("\n" + "=" * 62)
        print(f"Image      : {os.path.basename(str(image_path))} "
              f"({orig_size[0]}x{orig_size[1]}, mode {orig_mode})")
        print(f"Prediction : {pred}")
        print(f"NORMAL     : {result['normal_probability']:.2f}%")
        print(f"PNEUMONIA  : {result['pneumonia_probability']:.2f}%")
        print(f"Confidence : {result['confidence']:.2f}%")
        print("=" * 62)
        print("\nDIAGNOSTICS")
        print(f" 1. Raw logit                 : {logit.item():+.6f}")
        print(f" 2. sigmoid(logit) = P(class1): {p_pneu:.6f}")
        print(f"    1 - sigmoid    = P(class0): {p_norm:.6f}")
        print(f" 3. Class mapping             : "
              f"{{0: '{cfg['class_names'][0]}', 1: '{cfg['class_names'][1]}'}}")
        print(f" 4. Threshold                 : {threshold:.6f}")
        print(f"    source                    : {cfg['threshold_source']}")
        print(f" 5. Input size                : {cfg['image_size']}x{cfg['image_size']} "
              f"(letterbox={cfg['letterbox']})")
        print(f" 6. Tensor shape              : {tuple(x.shape)} {x.dtype}")
        print(f"    value range               : [{x.min():.4f}, {x.max():.4f}]")
        print(f" 7. Normalisation mean/std    : {cfg['mean']} / {cfg['std']}")
        print(f"    grayscale handling        : {cfg['grayscale_handling']}")
        print(f" 8. Classifier output dim     : {cfg['num_outputs']} "
              f"(in_features {cfg['head_in_features']}); loss {cfg['loss']} "
              f"-> sigmoid")
        print(f" 9. Grad-CAM target layer     : backbone.features[-1], "
              f"feature map {feat_shape}")
        print(f"    explaining                : {pred} (signed target {signed:+.0f})")

        margin = abs(p_pneu - threshold)
        if margin < 0.05:
            print("\n  >>> The probability sits within 0.05 of the threshold. This is")
            print("      an essentially undecided output; do not read the label as a")
            print("      finding.")

    # ---- two-panel figure ----------------------------------------------------
    if show or save_dir:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))

        axes[0].imshow(gray_np, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original X-ray", fontsize=19, pad=12)
        axes[0].axis("off")

        axes[1].imshow(gray_np, cmap="gray", vmin=0, vmax=1)
        heat = axes[1].imshow(cam, cmap=HEATMAP_CMAP, alpha=OVERLAY_ALPHA,
                              vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM Overlay", fontsize=19, pad=12)
        axes[1].axis("off")
        cb = fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
        cb.set_label("Model-attributed importance", fontsize=12)

        colour = "#C1445A" if idx == 1 else "#2E7D4F"
        fig.suptitle(f"Model Prediction: {pred}    |    Confidence: {conf * 100:.2f}%",
                     fontsize=21, fontweight="bold", y=0.97, color=colour)
        fig.text(0.5, 0.05,
                 f"NORMAL: {result['normal_probability']:.2f}%   |   "
                 f"PNEUMONIA: {result['pneumonia_probability']:.2f}%   |   "
                 f"threshold {threshold:.4f}   |   logit {logit.item():+.4f}",
                 ha="center", fontsize=13, family="monospace")
        fig.text(0.5, 0.008,
                 "Research visualisation. Model attention, not lesion "
                 "segmentation, and not a clinical diagnosis.",
                 ha="center", fontsize=9.5, style="italic", color="#555555")
        plt.tight_layout(rect=[0, 0.07, 1, 0.93])

        out_dir = Path(save_dir) if save_dir else (
            Path(os.path.dirname(os.path.abspath(image_path))) / "gradcam_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        fig_path = out_dir / f"{stem}_gradcam.png"
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        with open(out_dir / f"{stem}_prediction.json", "w") as f:
            json.dump(result, f, indent=2)
        result["saved_figure"] = str(fig_path)
        result["saved_json"] = str(out_dir / f"{stem}_prediction.json")

        if show:
            plt.show()
        plt.close(fig)

        if verbose:
            print(f"\nFigure saved   : {fig_path}")
            print(f"Prediction JSON: {result['saved_json']}")

    return result


# ==============================================================================
# 6. BATCH PREDICTION
# ==============================================================================

def predict_folder(folder, model, cfg, device, transform, output_csv=None):
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader

    folder = Path(folder)
    paths = sorted(str(p) for p in folder.rglob("*")
                   if p.is_file() and not p.name.startswith("._")
                   and p.suffix.lower() in VALID_EXT
                   and "__MACOSX" not in str(p))
    if not paths:
        print(f"No images found under {folder}")
        return None

    print(f"\nFound {len(paths):,} image(s) under {folder}")

    class _DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            try:
                with Image.open(paths[i]) as im:
                    img = im.convert("L").convert("RGB")
            except Exception:
                img = Image.new("RGB", (cfg["image_size"], cfg["image_size"]))
            return transform(img), i

    # num_workers=0 keeps this safe on Windows without a spawn guard around _DS.
    loader = DataLoader(_DS(), batch_size=16, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")

    probs = np.zeros(len(paths), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for n, (xb, idx) in enumerate(loader, 1):
            xb = xb.to(device, non_blocking=device.type == "cuda")
            probs[idx.numpy()] = torch.sigmoid(model(xb).float()).cpu().numpy()
            print(f"\r  batch {n}/{len(loader)}", end="", flush=True)
    print()

    threshold = cfg["threshold"]
    preds = (probs >= threshold).astype(int)
    df = pd.DataFrame({
        "image_path": paths,
        "prediction": [cfg["class_names"][p] for p in preds],
        "normal_probability": np.round((1 - probs) * 100, 2),
        "pneumonia_probability": np.round(probs * 100, 2),
        "confidence": np.round(np.where(preds == 1, probs, 1 - probs) * 100, 2),
    })
    out = Path(output_csv or (folder / "batch_predictions.csv"))
    df.to_csv(out, index=False)

    print(f"\nResults -> {out.resolve()}")
    print(df.prediction.value_counts().to_string())
    print(f"Mean P(pneumonia): {df.pneumonia_probability.mean():.2f}%")
    return df


# ==============================================================================
# 7. FULL TEST-SET EVALUATION
# ==============================================================================

def evaluate_test_folder(test_root, model, cfg, device, transform):
    """Point this at chest_xray/test (containing NORMAL/ and PNEUMONIA/) to get
    a full metric report using the locked threshold."""
    import pandas as pd
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, average_precision_score,
                                 confusion_matrix, classification_report)

    test_root = Path(test_root)
    rows = []
    for cls_dir, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
        d = test_root / cls_dir
        if not d.is_dir():
            raise FileNotFoundError(
                f"Expected {d} to exist. Point --evaluate at the folder that "
                "CONTAINS NORMAL/ and PNEUMONIA/."
            )
        for p in sorted(d.rglob("*")):
            if (p.is_file() and not p.name.startswith("._")
                    and p.suffix.lower() in VALID_EXT and "__MACOSX" not in str(p)):
                rows.append({"image_path": str(p), "label": label})

    df = pd.DataFrame(rows)
    print(f"\nEvaluating {len(df):,} images "
          f"({int((df.label == 0).sum()):,} NORMAL, "
          f"{int((df.label == 1).sum()):,} PNEUMONIA)")

    probs = []
    model.eval()
    with torch.no_grad():
        for i, r in df.iterrows():
            _, x, _, _ = load_image(r.image_path, transform)
            probs.append(torch.sigmoid(model(x.to(device)).float()).item())
            if (i + 1) % 50 == 0 or i + 1 == len(df):
                print(f"\r  {i + 1}/{len(df)}", end="", flush=True)
    print()

    y = df.label.values
    p = np.array(probs)
    t = cfg["threshold"]
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()

    print("\n" + "=" * 76)
    print(f"TEST EVALUATION | n = {len(y):,} | threshold {t:.4f} (from checkpoint)")
    print("=" * 76)
    baseline = max(y.mean(), 1 - y.mean())
    print(f"Accuracy             : {accuracy_score(y, yhat):.4f}   "
          f"(majority baseline {baseline:.4f})")
    print(f"Precision (PPV)      : {precision_score(y, yhat, zero_division=0):.4f}")
    print(f"Recall / Sensitivity : {recall_score(y, yhat, zero_division=0):.4f}"
          f"   <- the safety-critical one")
    print(f"Specificity (TNR)    : {tn / max(tn + fp, 1):.4f}")
    print(f"F1                   : {f1_score(y, yhat, zero_division=0):.4f}")
    print(f"ROC-AUC              : {roc_auc_score(y, p):.4f}")
    print(f"PR-AUC               : {average_precision_score(y, p):.4f}")
    print(f"\nTP {tp:,}  TN {tn:,}  FP {fp:,}  FN {fn:,}")
    print(f"Missed pneumonia (FN): {fn:,} of {tp + fn:,} "
          f"({100 * fn / max(tp + fn, 1):.2f}%)")
    print("\n" + classification_report(y, yhat,
                                       target_names=["NORMAL", "PNEUMONIA"],
                                       digits=4))
    print("Accuracy alone is not a sufficient result here: a constant predictor")
    print(f"scores {100 * baseline:.2f}% on this split.")

    df["probability"] = p
    df["prediction"] = [cfg["class_names"][v] for v in yhat]
    out = test_root / "test_evaluation_predictions.csv"
    df.to_csv(out, index=False)
    print(f"\nPer-image predictions -> {out.resolve()}")
    return df


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="Test the pneumonia EfficientNet-B3 model")
    ap.add_argument("image", nargs="?", default=None, help="path to one X-ray")
    ap.add_argument("--model", default=MODEL_PATH, help="path to the .pth checkpoint")
    ap.add_argument("--folder", default=None, help="run batch inference on a folder")
    ap.add_argument("--evaluate", default=None,
                    help="path to chest_xray/test for a full metric report")
    ap.add_argument("--no-show", action="store_true",
                    help="save figures without opening a window")
    ap.add_argument("--save-dir", default=None, help="where to write figures/JSON")
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    device = setup_device()
    model, cfg, _ = load_model(args.model, device)
    report_provenance(cfg)
    transform = build_transform(cfg)

    print("\nPreprocessing reproduced from the checkpoint:")
    print(" ", transform)

    if args.evaluate:
        evaluate_test_folder(args.evaluate, model, cfg, device, transform)
        return

    if args.folder:
        predict_folder(args.folder, model, cfg, device, transform)
        return

    image_path = args.image or IMAGE_PATH
    if not image_path:
        image_path = input("\nEnter the full path of the chest X-ray image: ").strip().strip('"')

    predict(image_path, model, cfg, device, transform,
            show=not args.no_show, save_dir=args.save_dir)

    print("\n" + "=" * 76)
    print("Research/educational output. Not a clinical diagnosis. This model was")
    print("trained on paediatric (~1-5y) single-centre chest radiographs; it does")
    print("not transfer to adults, other hospitals, or any other pathology.")
    print("=" * 76)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print("\n[ERROR]", e)
        sys.exit(1)
    except RuntimeError as e:
        print("\n[ERROR]", e)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception:
        print("\n[UNEXPECTED ERROR]")
        traceback.print_exc()
        sys.exit(3)
