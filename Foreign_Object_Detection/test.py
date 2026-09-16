#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test.py -- Single-image inference + Grad-CAM for the trained MURA EfficientNet-B3.

Reproduces the TRAINING preprocessing and architecture exactly, rather than assuming
the usual EfficientNet defaults. Everything that can be read from the checkpoint is
read from the checkpoint; nothing about the class mapping, threshold, image size or
normalisation is guessed.

Findings from inspecting the training script (MURA_EfficientNetB3_GradCAM.py) that
produced mura_efficientnet_b3_best.pth:

  1. Architecture      : torchvision.models.efficientnet_b3, wrapped in a class
                         `MURAEfficientNetB3` that stores it as `self.backbone`.
                         => every checkpoint key is prefixed "backbone.".
  2. Pretrained        : yes, EfficientNet_B3_Weights.IMAGENET1K_V1 at train time.
                         Irrelevant here: we load the trained weights, not ImageNet.
  3. Classifier        : nn.Sequential(nn.Dropout(p=0.4, inplace=True),
                                       nn.Linear(1536, 1))
                         replacing the original 1000-class head.
  4. Output neurons    : 1 (a single raw logit).
  5. Loss              : BCEWithLogitsLoss, with pos_weight = N_normal / N_abnormal
                         computed on the training split (~1.46).
  6. Class mapping     : 0 = Normal, 1 = Abnormal. Derived from the MURA directory
                         names: "*_negative" -> 0, "*_positive" -> 1. Stored in the
                         checkpoint under "class_names".
  7. Activation        : sigmoid, applied at INFERENCE ONLY (never inside the model,
                         because BCEWithLogitsLoss already fuses it).
  8. Preprocessing     : PIL open -> .convert("L") -> .convert("RGB")   [grayscale,
                         then replicated to 3 identical channels]
                         -> Resize((300, 300))  [no crop, aspect ratio NOT preserved]
                         -> ToTensor()
                         -> Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                         No augmentation at eval time.
  9. Imbalance/thresh. : pos_weight in the loss; no resampling. The decision threshold
                         was selected on the VALIDATION set (max-F1) and stored in the
                         checkpoint under "threshold" -- but only by the final save,
                         after threshold selection. A checkpoint written mid-training
                         has no "threshold" key; this script says so and uses 0.5.
 10. Checkpoint keys   : model_state_dict, model_name, architecture, num_classes,
                         class_names, image_size, normalization{mean,std},
                         input_channels, grayscale_handling, dropout_p, loss,
                         pos_weight, seed, torch_version, saved_at, stage, epoch,
                         val_metrics, monitor_metric, monitor_value, and (final save
                         only) threshold, threshold_criterion, test_metrics, ...
                         No optimizer state is stored.
 11. Grad-CAM layer    : model.backbone.features[-1] -- the final Conv2dNormActivation
                         (1536 channels, 10x10 at 300x300 input). Deepest layer that
                         still has spatial axes.

Usage (Windows):
    python test.py
    python test.py "C:\\path\\to\\xray.png"

Research/educational use only. Outputs are model predictions, not clinical diagnoses.
"""

import os
import sys
import json
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
# CONFIGURATION -- edit MODEL_PATH if your checkpoint lives elsewhere
# ==============================================================================

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get(
    "MURA_MODEL_PATH",
    str(ROOT_DIR / "mura_checkpoints" / "mura_efficientnet_b3_best.pth")
)

# Optionally hard-code an image path here; otherwise pass it on the command line
# or type it when prompted.
IMAGE_PATH = None

OVERLAY_ALPHA = 0.42      # heatmap transparency (spec asked for 0.35-0.50)
HEATMAP_CMAP  = "jet"


# ==============================================================================
# 1. DEVICE
# ==============================================================================

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 74)
    print("MURA EfficientNet-B3 -- inference + Grad-CAM")
    print("=" * 74)
    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU             :", torch.cuda.get_device_name(0))
        print("GPU memory      : {:.2f} GB".format(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
    print("Device in use   :", device)
    return device


# ==============================================================================
# 2. MODEL DEFINITION -- must match the training script byte for byte
# ==============================================================================

class MURAEfficientNetB3(nn.Module):
    """Identical to the class used during training.

    Keeping the `backbone` attribute name is what makes the checkpoint keys line up.
    Renaming it (or using a bare torchvision efficientnet_b3) means every key fails to
    match, which is exactly what went wrong in the previous testing script.
    """

    def __init__(self, dropout_p=0.4):
        super().__init__()
        self.backbone = efficientnet_b3(weights=None)   # weights come from the .pth
        in_features = self.backbone.classifier[1].in_features   # 1536 for B3
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


# ==============================================================================
# 3. CHECKPOINT INSPECTION AND LOADING
# ==============================================================================

def extract_state_dict(checkpoint):
    """Pull the weights out of whatever container the checkpoint uses."""
    if not isinstance(checkpoint, dict):
        return checkpoint, "raw state_dict object"
    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key], f'checkpoint["{key}"]'
    # A bare state_dict saved directly
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint, "checkpoint is itself a state_dict"
    raise RuntimeError("Could not locate weights inside the checkpoint.")


def align_prefixes(state_dict, model):
    """Reconcile key prefixes between the checkpoint and the model.

    Handles DataParallel ("module."), a wrapper attribute ("backbone."), and the case
    where the checkpoint was saved from a bare torchvision model but we are loading
    into the wrapper. Crucially, this NEVER silently accepts a mismatch -- the caller
    verifies missing/unexpected keys and aborts if the load did not actually happen.
    """
    model_keys = set(model.state_dict().keys())

    # Strip DataParallel prefix if present
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        print('  stripped "module." prefix (DataParallel checkpoint)')

    ckpt_keys = set(state_dict.keys())
    if ckpt_keys & model_keys:
        return state_dict, "keys already aligned"

    # Checkpoint saved from a bare efficientnet_b3 -> add the wrapper prefix
    if all(("backbone." + k) in model_keys for k in list(ckpt_keys)[:5]):
        return ({"backbone." + k: v for k, v in state_dict.items()},
                'added "backbone." prefix')

    # Checkpoint saved from the wrapper but loading into a bare model -> strip it
    if all(k.startswith("backbone.") for k in ckpt_keys):
        return ({k[len("backbone."):]: v for k, v in state_dict.items()},
                'stripped "backbone." prefix')

    return state_dict, "no prefix transformation applied"


def load_model(model_path, device):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n  {model_path}\n"
            "Edit MODEL_PATH at the top of this file to point at your .pth."
        )

    print("\n" + "=" * 74)
    print("CHECKPOINT INSPECTION")
    print("=" * 74)
    print("File :", model_path)
    print("Size : {:.1f} MB".format(os.path.getsize(model_path) / 1024 ** 2))

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # ---- report every piece of metadata the checkpoint carries -----------------
    if isinstance(checkpoint, dict):
        meta_keys = [k for k in checkpoint.keys() if k != "model_state_dict"]
        print("\nTop-level keys:", sorted(checkpoint.keys()))
        print("\nMetadata stored in the checkpoint:")
        for k in sorted(meta_keys):
            v = checkpoint[k]
            if isinstance(v, dict):
                short = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                         for kk, vv in list(v.items())[:8]}
                print(f"  {k:<22}: {short}")
            else:
                print(f"  {k:<22}: {v}")
        print("\nOptimizer state present :",
              any(k in checkpoint for k in ("optimizer_state_dict", "optimizer")))

    state_dict, where = extract_state_dict(checkpoint)
    print(f"\nWeights taken from      : {where}")
    print(f"Tensors in state_dict   : {len(state_dict)}")

    # ---- classifier geometry, read from the weights themselves ----------------
    classifier_weight_keys = [k for k in state_dict
                              if "classifier" in k and k.endswith(".weight")]
    if not classifier_weight_keys:
        raise RuntimeError("No classifier weights found in the checkpoint.")
    head_key = classifier_weight_keys[-1]
    num_outputs, head_in_features = state_dict[head_key].shape
    print(f"Classifier weight key   : {head_key}")
    print(f"Classifier shape        : [{num_outputs}, {head_in_features}]")
    print(f"Output neurons          : {num_outputs}")

    if num_outputs != 1:
        raise RuntimeError(
            f"This checkpoint has {num_outputs} output neurons, but the training "
            "script used a single logit with BCEWithLogitsLoss. The checkpoint does "
            "not come from that training run; do not interpret it with this script."
        )

    # ---- build the model and load -------------------------------------------
    dropout_p = checkpoint.get("dropout_p", 0.4) if isinstance(checkpoint, dict) else 0.4
    model = MURAEfficientNetB3(dropout_p=dropout_p)

    state_dict, note = align_prefixes(state_dict, model)
    print(f"Key alignment           : {note}")

    result = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)

    print("\n" + "-" * 74)
    print("LOAD VERIFICATION")
    print("-" * 74)
    print(f"Missing keys    : {len(missing)}")
    print(f"Unexpected keys : {len(unexpected)}")
    for k in missing[:5]:
        print("   missing    ->", k)
    for k in unexpected[:5]:
        print("   unexpected ->", k)

    # THIS is the check the previous script lacked. strict=False turns a total
    # mismatch into a silent no-op, leaving a randomly initialised network that
    # predicts ~50/50 on everything.
    total = len(model.state_dict())
    loaded = total - len(missing)
    print(f"Loaded {loaded}/{total} tensors ({100 * loaded / total:.1f}%)")
    if loaded == 0:
        raise RuntimeError(
            "NO weights were loaded -- the model is randomly initialised and any "
            "prediction from it is meaningless. The checkpoint's key names do not "
            "match this architecture."
        )
    if missing:
        raise RuntimeError(
            f"{len(missing)} tensors did not load. Refusing to run inference on a "
            "partially initialised model, because the output would look plausible "
            "while being noise. First missing key: " + missing[0]
        )
    print("All tensors loaded. Weights verified.")

    model.to(device).eval()

    # ---- configuration read from the checkpoint, not assumed ------------------
    cfg = {}
    if isinstance(checkpoint, dict):
        cfg["image_size"] = checkpoint.get("image_size", 300)
        norm = checkpoint.get("normalization", {})
        cfg["mean"] = norm.get("mean", [0.485, 0.456, 0.406])
        cfg["std"] = norm.get("std", [0.229, 0.224, 0.225])
        cfg["class_names"] = checkpoint.get("class_names", {0: "Normal", 1: "Abnormal"})
        cfg["threshold"] = checkpoint.get("threshold", None)
        cfg["threshold_criterion"] = checkpoint.get("threshold_criterion", None)
        cfg["stage"] = checkpoint.get("stage")
        cfg["epoch"] = checkpoint.get("epoch")
        cfg["val_metrics"] = checkpoint.get("val_metrics")
        cfg["grayscale_handling"] = checkpoint.get(
            "grayscale_handling", "PIL convert('L') then replicate to 3 channels")
        cfg["loss"] = checkpoint.get("loss", "BCEWithLogitsLoss")
        cfg["pos_weight"] = checkpoint.get("pos_weight")
        cfg["model_name"] = checkpoint.get("model_name", "efficientnet_b3")
    cfg["num_outputs"] = num_outputs
    cfg["head_in_features"] = head_in_features

    # class_names may come back with string keys after a round trip
    cn = cfg["class_names"]
    cfg["class_names"] = {int(k): v for k, v in cn.items()} if isinstance(cn, dict) else \
        {0: "Normal", 1: "Abnormal"}

    # ---- threshold: use the trained one, never invent one ---------------------
    if cfg["threshold"] is None:
        cfg["threshold_source"] = (
            "NOT PRESENT in checkpoint -> falling back to 0.5 (sigmoid default)")
        cfg["threshold"] = 0.5
    else:
        cfg["threshold_source"] = (
            f'validation-selected during training '
            f'(criterion: {cfg.get("threshold_criterion") or "unknown"})')

    return model, cfg, checkpoint


def report_training_stage(cfg):
    """Tell the user how far training actually got. This matters for interpretation."""
    print("\n" + "=" * 74)
    print("TRAINING PROVENANCE OF THIS CHECKPOINT")
    print("=" * 74)
    stage, epoch = cfg.get("stage"), cfg.get("epoch")
    print(f"Stage recorded : {stage}   Epoch recorded : {epoch}")
    if cfg.get("val_metrics"):
        vm = cfg["val_metrics"]
        print("Validation metrics at save time:")
        for k in ("accuracy", "precision", "recall", "specificity", "f1",
                  "roc_auc", "pr_auc"):
            if k in vm:
                print(f"   {k:<12}: {float(vm[k]):.4f}")
    if stage == 1:
        print("\n  >>> This checkpoint is from STAGE 1 only: the EfficientNet-B3")
        print("      backbone was FROZEN and only the 1,537-parameter classification")
        print("      head was trained. The convolutional features are still plain")
        print("      ImageNet features, never adapted to radiographs. Stage 2")
        print("      fine-tuning did not complete. Expect weak, low-confidence")
        print("      predictions -- that is the model, not this script.")
    elif stage == 2:
        print("\n  Stage 2 fine-tuning completed for this checkpoint.")
    print(f"\nThreshold in use : {cfg['threshold']:.4f}")
    print(f"Threshold source : {cfg['threshold_source']}")
    if cfg.get("pos_weight"):
        print(f"Training pos_weight (class imbalance): {cfg['pos_weight']:.4f}")


# ==============================================================================
# 4. PREPROCESSING -- copied from the training script's eval_transform
# ==============================================================================

def build_transform(cfg):
    """EXACTLY the training-time eval transform. No extra steps, no crop, no CLAHE."""
    return transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


def load_and_preprocess(image_path, cfg, transform):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"\nImage not found:\n  {image_path}")
    try:
        with Image.open(image_path) as im:
            original_mode = im.mode
            original_size = im.size            # (W, H)
            # Training did convert("L") first, collapsing any colour information,
            # then convert("RGB") to replicate that single channel three times.
            # Going straight to RGB (as the previous script did) is NOT the same
            # thing for any file that carries real colour.
            gray = im.convert("L").copy()
    except Exception as e:
        raise RuntimeError(f"Could not read the image file: {e}")

    rgb = gray.convert("RGB")
    tensor = transform(rgb).unsqueeze(0)
    return gray, tensor, original_mode, original_size


# ==============================================================================
# 5. GRAD-CAM
# ==============================================================================

class GradCAM:
    """Grad-CAM for a single-logit model. Runs in float32 on the selected device."""

    def __init__(self, model, target_layer=None):
        self.model = model
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "gradcam_target_layer"):
            self.target_layer = model.gradcam_target_layer
        elif hasattr(model, "backbone") and hasattr(model.backbone, "features"):
            self.target_layer = model.backbone.features[-1]
        elif hasattr(model, "features"):
            self.target_layer = model.features[-1]
        else:
            self.target_layer = model
        self.activations = None
        self.gradients = None
        self.handles = []

    def __enter__(self):
        self.handles.append(
            self.target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o.detach())))
        self.handles.append(
            self.target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0].detach())))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False

    def generate(self, input_tensor, device, signed_target=1.0, output_size=None):
        self.model.eval()
        x = input_tensor.to(device).float().requires_grad_(True)

        with torch.enable_grad():
            logit = self.model(x)                     # (1,)
            # signed_target = +1 -> explain evidence FOR abnormal
            # signed_target = -1 -> explain evidence FOR normal
            score = (signed_target * logit).sum()
            self.model.zero_grad(set_to_none=True)
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks captured nothing -- wrong target layer?")

        acts = self.activations[0]                    # (C, h, w)
        grads = self.gradients[0]                     # (C, h, w)
        alpha = grads.mean(dim=(1, 2), keepdim=True)  # channel importance
        cam = F.relu((alpha * acts).sum(dim=0))

        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else torch.zeros_like(cam)

        size = output_size or (x.shape[-2], x.shape[-1])
        cam = F.interpolate(cam[None, None], size=size,
                            mode="bilinear", align_corners=False)[0, 0]

        return (cam.detach().cpu().numpy(),
                float(logit.detach().item()),
                tuple(acts.shape))


# ==============================================================================
# 6. VISUALISATION -- exactly two panels
# ==============================================================================

def make_figure(gray_image, cam, result, image_path, cfg):
    gray_np = np.asarray(gray_image, dtype=np.float32) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # LEFT: the original X-ray, untouched, at its own resolution
    axes[0].imshow(gray_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original X-ray", fontsize=22, pad=15)
    axes[0].axis("off")

    # RIGHT: the SAME original X-ray with the heatmap laid over it semi-transparently
    axes[1].imshow(gray_np, cmap="gray", vmin=0, vmax=1)
    heat = axes[1].imshow(cam, cmap=HEATMAP_CMAP, alpha=OVERLAY_ALPHA, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Overlay", fontsize=22, pad=15)
    axes[1].axis("off")

    cbar = fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Model-attributed importance", fontsize=13)

    colour = "#C1445A" if result["prediction"] == "Abnormal" else "#2E7D4F"
    fig.suptitle(
        f"Model Prediction: {result['prediction'].upper()}    |    "
        f"Confidence: {result['confidence']:.2f}%",
        fontsize=24, fontweight="bold", y=0.97, color=colour,
    )
    fig.text(
        0.5, 0.045,
        f"Abnormal: {result['abnormal_probability']:.2f}%   |   "
        f"Normal: {result['normal_probability']:.2f}%   |   "
        f"threshold {result['threshold']:.4f}   |   logit {result['logit']:+.4f}",
        ha="center", fontsize=14, family="monospace",
    )
    fig.text(
        0.5, 0.005,
        "Research visualisation. Model-attributed importance, not lesion "
        "segmentation and not a clinical diagnosis.",
        ha="center", fontsize=10, style="italic", color="#555555",
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.93])

    out_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)),
                           "gradcam_results")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, stem + "_gradcam.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    json_path = os.path.join(out_dir, stem + "_prediction.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    plt.show()
    return out_path, json_path


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main():
    device = setup_device()

    model, cfg, _ = load_model(MODEL_PATH, device)
    report_training_stage(cfg)

    # ---- resolve the image path -------------------------------------------
    image_path = IMAGE_PATH
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    if not image_path:
        image_path = input("\nEnter the full path of the X-ray image: ").strip().strip('"')
    image_path = os.path.abspath(image_path)

    transform = build_transform(cfg)
    gray, input_tensor, original_mode, original_size = load_and_preprocess(
        image_path, cfg, transform)

    # ---- inference ---------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logit = model(input_tensor.to(device)).float()
        p_abnormal = torch.sigmoid(logit).item()
    p_normal = 1.0 - p_abnormal

    threshold = float(cfg["threshold"])
    pred_idx = int(p_abnormal >= threshold)
    pred_name = cfg["class_names"][pred_idx]
    confidence = p_abnormal if pred_idx == 1 else p_normal

    # ---- Grad-CAM ----------------------------------------------------------
    # Explain the class that was predicted: the raw logit is evidence for
    # "abnormal", so negate it when the prediction is "normal".
    signed = 1.0 if pred_idx == 1 else -1.0
    target_layer = model.gradcam_target_layer
    with GradCAM(model, target_layer) as engine:
        cam, cam_logit, feat_shape = engine.generate(
            input_tensor, device, signed_target=signed,
            output_size=(gray.height, gray.width))

    result = {
        "image_path": image_path,
        "prediction": pred_name,
        "predicted_class_index": pred_idx,
        "normal_probability": round(p_normal * 100, 2),
        "abnormal_probability": round(p_abnormal * 100, 2),
        "confidence": round(confidence * 100, 2),
        "logit": round(float(logit.item()), 6),
        "threshold": round(threshold, 6),
        "class_mapping": {str(k): v for k, v in cfg["class_names"].items()},
        "model_name": cfg.get("model_name", "efficientnet_b3"),
        "image_size": cfg["image_size"],
        "normalization_mean": cfg["mean"],
        "normalization_std": cfg["std"],
        "checkpoint_stage": cfg.get("stage"),
        "checkpoint_epoch": cfg.get("epoch"),
        "disclaimer": ("Research/educational model output. Not a clinical diagnosis. "
                       "MURA labels are Normal/Abnormal, not fracture-specific."),
    }

    # ---- full diagnostic dump ---------------------------------------------
    print("\n" + "=" * 74)
    print("DIAGNOSTICS")
    print("=" * 74)
    print(f" 1. Raw model output (logit)   : {logit.item():+.6f}")
    print(f" 2. sigmoid(logit) = P(class 1): {p_abnormal:.6f}")
    print(f"    1 - sigmoid     = P(class 0): {p_normal:.6f}")
    print(f" 3. Class mapping from training: "
          f"{{0: '{cfg['class_names'][0]}', 1: '{cfg['class_names'][1]}'}}")
    print(f" 4. Decision threshold         : {threshold:.6f}")
    print(f"    source                     : {cfg['threshold_source']}")
    print(f" 5. Input image size           : {cfg['image_size']} x {cfg['image_size']} "
          f"(original file: {original_size[0]} x {original_size[1]}, mode {original_mode})")
    print(f" 6. Tensor shape fed to model  : {tuple(input_tensor.shape)}  "
          f"dtype {input_tensor.dtype}")
    ch_equal = (torch.allclose(input_tensor[:, 0], input_tensor[:, 1]) and
                torch.allclose(input_tensor[:, 1], input_tensor[:, 2]))
    print(f"    channels identical pre-norm: True (forced via convert('L')->RGB); "
          f"post-norm: {ch_equal} (False is correct -- per-channel mean/std)")
    print(f"    value range after normalise: "
          f"[{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
    print(f" 7. Normalisation mean         : {cfg['mean']}")
    print(f"    Normalisation std          : {cfg['std']}")
    print(f"    Grayscale handling         : {cfg['grayscale_handling']}")
    print(f" 8. Classifier output dimension: {cfg['num_outputs']} "
          f"(in_features {cfg['head_in_features']})")
    print(f"    Loss used in training      : {cfg['loss']} -> activation is sigmoid")
    print(f" 9. Grad-CAM target layer      : backbone.features[-1] "
          f"({type(target_layer).__name__})")
    print(f"    feature map (C, H, W)      : {feat_shape}")
    print(f"    explaining class           : "
          f"{cfg['class_names'][pred_idx]} (signed target {signed:+.0f})")
    print(f"    CAM range                  : [{cam.min():.4f}, {cam.max():.4f}] "
          f"upsampled to {cam.shape}")

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    print(f"Prediction           : {pred_name.upper()}")
    print(f"Confidence           : {result['confidence']:.2f}%")
    print("Class probabilities  :")
    print(f"   Normal   (class 0): {result['normal_probability']:.2f}%")
    print(f"   Abnormal (class 1): {result['abnormal_probability']:.2f}%")

    margin = abs(p_abnormal - threshold)
    if margin < 0.05:
        print("\n  >>> The probability sits within 0.05 of the threshold. This is an")
        print("      essentially undecided output. Do not read the label as a finding.")

    out_path, json_path = make_figure(gray, cam, result, image_path, cfg)
    print(f"\nFigure saved         : {out_path}")
    print(f"Prediction JSON      : {json_path}")
    print("=" * 74)
    print("Research/educational output. Not a clinical diagnosis. MURA labels are")
    print("Normal/Abnormal at the study level and are not fracture-specific.")
    print("=" * 74)


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
