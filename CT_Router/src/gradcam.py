"""Multi-Layer Gradient Projection & Grad-CAM Explainability Engine.

Collects class-specific gradients and activations across multiple hierarchical
layers of the model, projects their gradient-activation energy, and fuses them
into a high-resolution, multi-scale anatomical heatmap.
"""

from typing import Tuple, Optional, Union, List, Dict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAMOutput:
    """Encapsulates Multi-Layer Gradient Projection results."""
    def __init__(
        self,
        fused_norm: np.ndarray,
        overlay_pil: Image.Image,
        heatmap_pil: Image.Image,
        weights_dict: Dict[str, float],
        layer_overlays: Dict[str, Image.Image],
        grid_pil: Optional[Image.Image] = None,
    ):
        self.fused_norm = fused_norm
        self.overlay_pil = overlay_pil
        self.heatmap_pil = heatmap_pil
        self.weights_dict = weights_dict
        self.layer_overlays = layer_overlays
        self.grid_pil = grid_pil

    # Allow tuple-like indexing and unpacking for seamless backward compatibility
    def __iter__(self):
        return iter((self.fused_norm, self.overlay_pil, self.heatmap_pil, self.weights_dict))

    def __getitem__(self, idx):
        return (self.fused_norm, self.overlay_pil, self.heatmap_pil, self.weights_dict)[idx]


def extract_anatomical_mask(orig_img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Generates an anatomical tissue mask to suppress background air and couch artifacts."""
    if isinstance(orig_img, Image.Image):
        arr = np.array(orig_img.convert("RGB"))
    else:
        arr = orig_img
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # Threshold body region with morphological closing
    _, thresh = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Extract largest contour (patient anatomy)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask.astype(np.float32) / 255.0
    return np.ones_like(gray, dtype=np.float32)


def build_projection_grid(
    orig_pil: Image.Image,
    layer_maps: List[Tuple[str, float, np.ndarray]],
    fused_overlay: Image.Image,
    target_dim: int = 224,
) -> Image.Image:
    """Builds a horizontal clinical composite grid showing multi-layer gradient projections."""
    orig_resized = orig_pil.resize((target_dim, target_dim), Image.Resampling.LANCZOS)
    fused_resized = fused_overlay.resize((target_dim, target_dim), Image.Resampling.LANCZOS)

    panels = []
    # 1. Original scan
    panels.append(("Input CT Scan", np.array(orig_resized)))

    # 2. Individual layer gradient projections
    for name, weight, l_cam in layer_maps:
        cam_u8 = np.uint8(np.clip(l_cam, 0, 1) * 255)
        heat_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        heat_pil = Image.fromarray(heat_rgb).resize((target_dim, target_dim), Image.Resampling.BILINEAR)

        # Soft blend with original
        orig_arr = np.array(orig_resized).astype(np.float32)
        heat_arr = np.array(heat_pil).astype(np.float32)
        blended = (orig_arr * 0.45 + heat_arr * 0.55).astype(np.uint8)

        # Label with contribution percentage
        lbl = f"{name} ({weight:.1f}%)"
        panels.append((lbl, blended))

    # 3. Final fused CAM
    panels.append(("Fused Grad-CAM", np.array(fused_resized)))

    # Assemble horizontally with top label banners
    header_h = 28
    tile_h = target_dim + header_h
    tile_w = target_dim
    n_panels = len(panels)
    grid_img = np.zeros((tile_h, n_panels * tile_w, 3), dtype=np.uint8)
    grid_img[:] = (15, 23, 42)  # Dark slate background #0f172a

    for i, (title, img_arr) in enumerate(panels):
        x_start = i * tile_w
        # Header banner
        header_strip = np.zeros((header_h, tile_w, 3), dtype=np.uint8)
        header_strip[:] = (20, 31, 55) if i not in (0, n_panels - 1) else (30, 41, 59)
        cv2.putText(
            header_strip,
            title,
            (6, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 230, 242),
            1,
            cv2.LINE_AA,
        )
        grid_img[0:header_h, x_start:x_start + tile_w] = header_strip
        grid_img[header_h:tile_h, x_start:x_start + tile_w] = img_arr

        # Separator border
        if i > 0:
            grid_img[:, x_start:x_start + 1] = (51, 65, 85)

    return Image.fromarray(grid_img)


def compute_multilayer_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_image: Union[Image.Image, np.ndarray],
    hook_layers: List[Tuple[str, nn.Module]],
    class_idx: Optional[int] = None,
    alpha: float = 0.48,
    mode: str = "hybrid",
) -> GradCAMOutput:
    """Collects gradients across multiple layers, projects them, and fuses into a single CAM.

    Args:
        model: PyTorch neural network
        input_tensor: 4D preprocessed input tensor (1, C, H, W)
        original_image: Original scan (PIL Image or numpy array)
        hook_layers: List of (layer_name, module) pairs to hook across the network hierarchy
        class_idx: Target class index to explain (None = highest predicted class)
        alpha: Heatmap blend transparency [0.0, 1.0]
        mode: 'hybrid' (blended spatial + channel projection), 'layercam', or 'gradcam'

    Returns:
        GradCAMOutput containing:
            - fused_norm (np.ndarray): Fused 2D normalized attention map
            - overlay_pil (PIL.Image): Blended JET overlay on original scan
            - heatmap_pil (PIL.Image): Raw JET heatmap
            - weights_dict (Dict[str, float]): Layer contribution percentages
            - layer_overlays (Dict[str, PIL.Image]): Individual projected layer overlays
            - grid_pil (PIL.Image): Multi-panel progression grid
    """
    if isinstance(original_image, np.ndarray):
        orig_pil = Image.fromarray(original_image).convert("RGB")
    else:
        orig_pil = original_image.convert("RGB")

    w_orig, h_orig = orig_pil.size
    tissue_mask = extract_anatomical_mask(orig_pil)

    records = {}
    handles = []

    # Register forward and backward hooks on all selected hierarchical layers
    for name, module in hook_layers:
        def make_fwd(layer_name):
            def fwd_hook(mod, inp, out):
                if torch.is_tensor(out) and out.ndim == 4:
                    records.setdefault(layer_name, {})["act"] = out.detach()
            return fwd_hook

        def make_bwd(layer_name):
            def bwd_hook(mod, grad_in, grad_out):
                if grad_out and grad_out[0] is not None and grad_out[0].ndim == 4:
                    records.setdefault(layer_name, {})["grad"] = grad_out[0].detach()
            return bwd_hook

        handles.append(module.register_forward_hook(make_fwd(name)))
        handles.append(module.register_full_backward_hook(make_bwd(name)))

    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        # Single backward pass: class gradients propagate to all hooked layers simultaneously
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        layer_maps = []
        layer_scores = []
        layer_names = []

        for name, _ in hook_layers:
            rec = records.get(name, {})
            a = rec.get("act")
            g = rec.get("grad")

            if a is None or g is None or a.shape != g.shape:
                continue

            # 1. Project gradient and activation
            if mode == "layercam":
                # Element-wise positive gradient projection (Layer-CAM)
                cam_t = F.relu((F.relu(g) * a).sum(dim=1, keepdim=True))
            elif mode == "gradcam":
                # Global average pooled Grad-CAM
                weights = g.mean(dim=(2, 3), keepdim=True)
                cam_t = F.relu((weights * a).sum(dim=1, keepdim=True))
            else:
                # Hybrid projection: combines channel-wise positive GAP with spatial element-wise projection
                w_pos = g.clamp(min=0).mean(dim=(2, 3), keepdim=True)
                cam_gap = F.relu((w_pos * a).sum(dim=1, keepdim=True))
                cam_spatial = F.relu((F.relu(g) * a).sum(dim=1, keepdim=True))
                cam_t = 0.5 * cam_gap + 0.5 * cam_spatial

            # 2. Interpolate layer map to target image resolution
            cam_t = F.interpolate(cam_t, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
            cam_np = cam_t[0, 0].float().cpu().numpy()
            cam_np = np.maximum(0, cam_np)

            # Apply anatomical tissue mask to suppress couch/air artifacts
            cam_np = cam_np * tissue_mask

            diff = cam_np.max() - cam_np.min()
            if diff > 1e-12:
                cam_np = (cam_np - cam_np.min()) / diff
            else:
                cam_np = np.zeros_like(cam_np)

            # 3. Layer gradient energy weighting: mean(|Grad|) * mean(|Act|)
            grad_energy = float(g.abs().mean().cpu())
            act_energy = float(a.abs().mean().cpu())
            energy_score = grad_energy * act_energy

            layer_maps.append(cam_np)
            layer_scores.append(energy_score)
            layer_names.append(name)

        if not layer_maps:
            blank = np.zeros((h_orig, w_orig), dtype=np.float32)
            return GradCAMOutput(blank, orig_pil, orig_pil, {}, {}, None)

        # 4. Normalize projection weights across multi-layer hierarchy
        scores_arr = np.array(layer_scores, dtype=np.float64)
        weights = np.sqrt(scores_arr + 1e-15)
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)

        # 5. Fuse layer-projected maps and generate individual overlays
        fused = np.zeros((h_orig, w_orig), dtype=np.float32)
        weights_dict = {}
        layer_overlays = {}
        grid_layer_info = []

        orig_np = np.array(orig_pil).astype(np.float32)

        for w_val, name, l_map in zip(weights, layer_names, layer_maps):
            fused += float(w_val) * l_map
            pct = round(float(w_val) * 100.0, 2)
            weights_dict[name] = pct
            grid_layer_info.append((name, pct, l_map))

            # Generate individual layer overlay
            l_u8 = np.uint8(np.clip(l_map, 0, 1) * 255)
            l_bgr = cv2.applyColorMap(l_u8, cv2.COLORMAP_JET)
            l_rgb = cv2.cvtColor(l_bgr, cv2.COLOR_BGR2RGB)
            l_mask = (l_map >= 0.12)[..., None]
            l_blended = (orig_np * (1.0 - alpha) + l_rgb.astype(np.float32) * alpha).astype(np.uint8)
            l_final = np.where(l_mask, l_blended, np.array(orig_pil))
            layer_overlays[name] = Image.fromarray(l_final)

        # 6. Percentile normalization for high visual contrast
        fused = np.nan_to_num(fused, nan=0.0)
        fused = np.maximum(fused, 0)
        diff = fused.max() - fused.min()
        if diff > 1e-12:
            fused_norm = (fused - fused.min()) / diff
        else:
            fused_norm = np.zeros_like(fused)

        # 7. Apply JET colormap to fused CAM
        heat_u8 = np.uint8(np.clip(fused_norm, 0, 1) * 255)
        heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

        # 8. Soft threshold overlay blending (attenuates noise below 0.12)
        mask = (fused_norm >= 0.12)[..., None]
        blended = (orig_np * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)

        final_overlay = np.array(orig_pil).copy()
        final_overlay = np.where(mask, blended, final_overlay)

        overlay_pil = Image.fromarray(final_overlay)
        heatmap_pil = Image.fromarray(heat_rgb)

        # 9. Build multi-layer projection composite grid
        grid_pil = build_projection_grid(orig_pil, grid_layer_info, overlay_pil)

        return GradCAMOutput(
            fused_norm=fused_norm,
            overlay_pil=overlay_pil,
            heatmap_pil=heatmap_pil,
            weights_dict=weights_dict,
            layer_overlays=layer_overlays,
            grid_pil=grid_pil,
        )

    finally:
        for h in handles:
            h.remove()

