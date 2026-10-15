#!/usr/bin/env python3
"""
Multi-Layer Gradient Projection & Grad-CAM Engine for COVIDx CT-3A

This module implements multi-layer gradient extraction across all Conv2d layers
in the backbone, capturing feature activations and class gradients:
- Every Conv2d layer: captures A_l and d(score)/dA_l
- Layer-CAM: Element-wise positive gradient projection (ReLU(G) * ReLU(A))
- Multi-layer Fusion: Gradient-activation energy weighted fusion (sqrt(grad_energy * act_energy))
- Original Grad-CAM Colormap (JET) Overlay: Classic Grad-CAM red/yellow/green/cyan/blue heatmap with soft alpha blending
- Coordinate Projection: Exact model crop -> original CT image coordinate mapping
"""

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================
# MULTI-LAYER GRADIENT CAPTURE ENGINE
# ============================================================
class MultiLayerGradCAM:
    """
    Captures activation maps + class gradients from every Conv2d layer
    in the EfficientNet backbone and builds gradient-weighted fused CAMs.
    """

    def __init__(self, model):
        self.model = model
        self.records = {}
        self.handles = []

        # Disable in-place operations across model to allow clean backward hooks
        for mod in self.model.modules():
            if hasattr(mod, "inplace"):
                mod.inplace = False

        # Hook all convolutional feature layers in the backbone
        for name, module in model.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                full_name = f"backbone.{name}"
                self.handles.append(module.register_forward_hook(self._fwd(full_name)))
                self.handles.append(module.register_full_backward_hook(self._bwd(full_name)))

    def _fwd(self, name):
        def hook(module, inputs, output):
            if torch.is_tensor(output) and output.ndim == 4:
                self.records.setdefault(name, {})["activation"] = output.detach().clone()
        return hook

    def _bwd(self, name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0]
            if torch.is_tensor(g) and g.ndim == 4:
                self.records.setdefault(name, {})["gradient"] = g.detach().clone()
        return hook

    def remove(self):
        """Remove all forward and backward hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @staticmethod
    def normalize_map(cam):
        """Robust percentile normalization [0, 1]."""
        cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
        cam = np.maximum(cam, 0)
        if not np.any(cam > 0):
            return np.zeros_like(cam, dtype=np.float32)
        lo, hi = np.percentile(cam[cam > 0], [1, 99])
        if hi <= lo:
            lo, hi = float(cam.min()), float(cam.max())
        if hi <= lo:
            return np.zeros_like(cam, dtype=np.float32)
        return np.clip((cam - lo) / (hi - lo), 0, 1).astype(np.float32)

    def build_maps(self, input_hw, mode="layercam"):
        """
        Extracts gradients from each layer and projects the Grad-CAM.

        Args:
            input_hw: (Height, Width) of the input tensor
            mode: 'layercam' (elementwise positive-gradient projection)
                  or 'gradcam' (global average pooling gradient projection)

        Returns:
            fused: 2D numpy array [0, 1] fused CAM
            diagnostics: List of layer diagnostics with fusion weights & shapes
            individual_maps: Dict[layer_name -> 2D numpy array [0, 1]]
        """
        H, W = input_hw
        maps = []
        diagnostics = []
        individual_maps = {}

        for name, rec in self.records.items():
            a, g = rec.get("activation"), rec.get("gradient")
            if a is None or g is None or a.ndim != 4 or g.ndim != 4:
                continue
            if a.shape[-1] < 2 or a.shape[-2] < 2:
                continue
            if a.shape != g.shape:
                continue

            if mode == "layercam":
                # Element-wise positive gradient projection (Layer-CAM)
                cam_t = (F.relu(g) * F.relu(a)).sum(dim=1, keepdim=True)
            else:
                # Standard Global Average Pooled Grad-CAM
                weights = g.mean(dim=(2, 3), keepdim=True)
                cam_t = F.relu((weights * a).sum(dim=1, keepdim=True))

            cam_t = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)
            cam = cam_t[0, 0].float().cpu().numpy()
            cam = self.normalize_map(cam)
            individual_maps[name] = cam

            # Gradient energy is the layer's class-specific signal strength
            grad_energy = float(g.abs().mean().cpu())
            activation_energy = float(a.abs().mean().cpu())
            score = grad_energy * activation_energy

            maps.append((name, cam, score))
            diagnostics.append({
                "layer": name,
                "activation_shape": list(a.shape),
                "gradient_shape": list(g.shape),
                "mean_abs_gradient": grad_energy,
                "mean_abs_activation": activation_energy,
                "fusion_score": score,
                "cam_mean": float(cam.mean()),
                "cam_max": float(cam.max()),
            })

        if not maps:
            raise RuntimeError("No convolutional layers produced valid activation/gradient maps")

        # Stable fusion weights based on gradient-activation energy
        scores = np.array([m[2] for m in maps], dtype=np.float64)
        weights = np.sqrt(scores + 1e-20)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights /= weights.sum()

        fused = np.zeros((H, W), dtype=np.float32)
        for w, (_, cam, _) in zip(weights, maps):
            fused += float(w) * cam

        fused = self.normalize_map(fused)

        for d, w in zip(diagnostics, weights):
            d["fusion_weight"] = float(w)

        return fused, diagnostics, individual_maps


# ============================================================
# COORDINATE PROJECTION & OVERLAY
# ============================================================
def project_to_original(cam_model, crop_box, original_size):
    """
    Projects Grad-CAM from cropped model coordinates back to original CT coordinates.
    """
    W, H = original_size
    x1, y1, x2, y2 = map(int, crop_box)
    cw, ch = max(1, x2 - x1), max(1, y2 - y1)

    cam_crop = cv2.resize(cam_model, (cw, ch), interpolation=cv2.INTER_CUBIC)
    full = np.zeros((H, W), dtype=np.float32)

    y1_cl = max(0, min(y1, H))
    y2_cl = max(0, min(y2, H))
    x1_cl = max(0, min(x1, W))
    x2_cl = max(0, min(x2, W))

    crop_h = y2_cl - y1_cl
    crop_w = x2_cl - x1_cl

    if crop_h > 0 and crop_w > 0:
        full[y1_cl:y2_cl, x1_cl:x2_cl] = cam_crop[:crop_h, :crop_w]

    return np.clip(full, 0, 1)


def overlay(original, cam, alpha=0.48, threshold=0.18):
    """
    Original Grad-CAM JET colormap overlay (blue -> cyan -> green -> yellow -> red).
    Focal hotspots appear in classic vibrant red/yellow, while weak activations below threshold
    are gracefully attenuated so normal anatomy isn't flooded with blue.
    """
    rgb = np.asarray(original.convert("RGB")).copy()
    u8 = np.uint8(np.clip(cam, 0, 1) * 255)
    heat = cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    mask = cam >= threshold
    alpha_map = (np.clip((cam - threshold) / (1.0 - threshold + 1e-8), 0, 1) * alpha)[..., None]
    blended = (rgb.astype(np.float32) * (1.0 - alpha_map) + heat.astype(np.float32) * alpha_map).astype(np.uint8)

    out = rgb.copy()
    out[mask] = blended[mask]
    return out, heat


# ============================================================
# BOUNDING BOX & FOREGROUND EXTRACTION
# ============================================================
def clamp_bbox(bbox, w, h):
    x1, y1, x2, y2 = map(float, bbox)
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def add_margin(box, w, h, margin=0.04):
    x1, y1, x2, y2 = box
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    return max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy)


def safe_foreground_bbox(img_rgb, threshold=12):
    """
    Extracts the patient body contour using connected components analysis,
    effectively excluding the scanner couch, peripheral markings, and empty air margins.
    """
    gray = np.asarray(img_rgb.convert("L"))
    h, w = gray.shape

    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, w, h)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (w * h * 0.05):
        return (0, 0, w, h)

    x, y, cw, ch = cv2.boundingRect(largest)
    box = (x, y, x + cw, y + ch)
    return add_margin(box, w, h, margin=0.03)


def read_official_bbox(label_file, filename):
    target = Path(filename).name
    p_file = Path(label_file)
    if not p_file.is_file():
        return None, None
    with open(p_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 6 and Path(p[0]).name == target:
                return tuple(map(float, p[2:6])), int(p[1])
    return None, None


def crop_image(img, bbox, margin=0.03):
    w, h = img.size
    box = clamp_bbox(bbox, w, h)
    if box is None:
        box = (0, 0, w, h)
    box = add_margin(box, w, h, margin)
    return img.crop(box), box


def make_tensor(crop, size=256):
    tfm = transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return tfm(crop).unsqueeze(0)


def choose_crop(original, label_file=None, filename=None, bbox=None, no_bbox=False, bbox_margin=0.03):
    """
    Selects and crops the CT region based on priority:
    1. no_bbox -> full original image
    2. official label file -> official COVIDx annotated bbox
    3. manual bbox -> user supplied [x1, y1, x2, y2]
    4. automatic foreground -> patient body contour (scanner couch excluded)
    5. fallback -> full image
    """
    if no_bbox:
        return original, (0, 0, original.width, original.height), "disabled (full image)"

    box = None
    source = "none"

    if label_file:
        box, _ = read_official_bbox(label_file, filename or "image")
        if box is not None:
            source = "official COVIDx bbox"

    if box is None and bbox is not None:
        box = tuple(bbox)
        source = "manual bbox"

    if box is None:
        box = safe_foreground_bbox(original)
        source = "patient body contour (couch excluded)"

    clamped = clamp_bbox(box, original.width, original.height)
    if clamped is None:
        clamped = (0, 0, original.width, original.height)
        source = "full image fallback"

    final_box = add_margin(clamped, original.width, original.height, bbox_margin)
    return original.crop(final_box), final_box, source
