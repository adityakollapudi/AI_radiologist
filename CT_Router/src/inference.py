"""Medical CT Multi-Model Unified Inference Engine & Report Generator.

Exposes:
  - MedicalCTPredictor: End-to-end multi-model inference with Multi-Layer Gradient Projection
  - format_report(r): String report formatter matching the clinical prototype style
  - save_prediction_json(r, path): JSON serialization helper
"""

import os
import re
import cv2
import json
import time
import hashlib
from pathlib import Path
from typing import Union, Dict, Any, Optional

import numpy as np
import torch
from PIL import Image

from .pipeline import MedicalCTPipeline
from .gradcam import compute_multilayer_gradcam

BASE_DIR = Path(__file__).resolve().parent.parent


def format_report(r: Dict[str, Any]) -> str:
    """Formats the multi-model prediction result into a clean console report."""
    lines = []
    lines.append("=" * 70)
    lines.append("                 MEDICAL CT AI DIAGNOSTIC REPORT")
    lines.append("=" * 70)
    lines.append(f"Image File       : {r.get('image_path', 'unknown')}")
    lines.append(f"Processing Time  : {r.get('latency_seconds', 0.0):.2f}s")
    lines.append("-" * 70)

    # Stage 1: Router
    router_res = r.get("router", {})
    anatomy = router_res.get("prediction", "UNKNOWN")
    lines.append("[STAGE 1: ANATOMY ROUTER]")
    lines.append(f"  • Identified Anatomy : {anatomy} ({router_res.get('confidence_pct', 'N/A')} confidence)")
    lines.append(f"  • Routing Status     : {router_res.get('status', 'ROUTED')}")
    lines.append(f"  • Pipeline Route     : {r.get('routing_action', 'N/A')}")
    probs_router = router_res.get("probabilities", {})
    if probs_router:
        lines.append(f"  • Router Probabilities:")
        lines.append(f"      - CHEST_CT : {probs_router.get('CHEST_CT', 0.0) * 100:.2f}%")
        lines.append(f"      - BRAIN_CT : {probs_router.get('BRAIN_CT', 0.0) * 100:.2f}%")
    lines.append("-" * 70)

    # Uncertainty check
    if router_res.get("uncertain"):
        lines.append("[ALERT] Scan falls in the uncertainty band (|P(Brain) - 0.5| < 0.45).")
        lines.append("        Automated routing halted. Slices must be manually reviewed by a radiologist.")
        lines.append("=" * 70)
        return "\n".join(lines)

    expert_res = r.get("expert_result", {})
    if not expert_res or expert_res.get("status") != "SUCCESS":
        lines.append(f"[Notice] {expert_res.get('message', 'Downstream expert unavailable.')}")
        lines.append("=" * 70)
        return "\n".join(lines)

    # Stage 2: Individual Model Native Formatting
    lines.append("[STAGE 2: DOWNSTREAM EXPERT ANALYSIS]")

    if anatomy == "CHEST_CT":
        lines.append(f"  • Target Expert      : {expert_res.get('expert', 'Chest CT Expert')}")
        lines.append(f"  • Predicted Class    : {expert_res.get('diagnosis', 'N/A')}")
        lines.append(f"  • Confidence         : {expert_res.get('confidence_pct', 'N/A')}")
        if "top1_margin_pct" in expert_res:
            lines.append(f"  • Top-1 Margin       : {expert_res['top1_margin_pct']}")
        if "entropy" in expert_res:
            lines.append(f"  • Entropy            : {expert_res['entropy']:.4f}")
        lines.append(f"  • Crop Source        : {expert_res.get('crop_source', 'patient body contour')}")
        lines.append("")
        lines.append("  CLASS PROBABILITIES")
        lines.append("  " + "-" * 50)
        probs = expert_res.get("probabilities", {})
        for name, p in probs.items():
            bar_len = int(p * 26)
            bar = "=" * bar_len
            lines.append(f"    {name:<12s}: {p * 100:6.2f}%  [{bar:<26}]")
        lines.append("  " + "-" * 50)
        lines.append(f"  CLINICAL SUMMARY: {expert_res.get('clinical_summary', 'N/A')}")

    elif anatomy == "BRAIN_CT":
        arch = expert_res.get("model_name", "se_resnext101_32x4d")
        diag_title = expert_res.get("diag_title", expert_res.get("diagnosis", "N/A"))
        lines.append(f"  • Model Architecture : {arch}")
        lines.append(f"  • Diagnosis Badge    : {diag_title}")
        lines.append("")
        lines.append(f"  {'CLASS':<20s} {'PROBABILITY':<12s} {'VISUAL BAR (0-100%)':<24s}")
        lines.append("  " + "-" * 60)
        probs = expert_res.get("probabilities", {})
        for lab, p in probs.items():
            pct = p * 100.0
            filled = int(round((pct / 100.0) * 20))
            filled = max(0, min(20, filled))
            bar = "#" * filled + "-" * (20 - filled)
            lines.append(f"  {lab:<20s}: {pct:6.2f}%    [{bar}]")
        lines.append("  " + "-" * 60)
        lines.append(f"  CLINICAL SUMMARY: {expert_res.get('clinical_summary', 'N/A')}")

    # Multi-Layer Gradient Projection Contributions
    layer_weights = r.get("gradcam_layer_weights", {})
    if layer_weights:
        lines.append("-" * 70)
        lines.append("MULTI-LAYER GRADIENT PROJECTIONS:")
        for l_name, l_wt in layer_weights.items():
            bar_len = int(l_wt / 100.0 * 20)
            bar_str = "=" * bar_len
            lines.append(f"  • {l_name:<16s}: {l_wt:5.1f}%  [{bar_str:<20}]")

    # Visual Artifacts section
    gradcam_files = r.get("gradcam_files", {})
    if gradcam_files:
        lines.append("-" * 70)
        lines.append("EXPLAINABILITY ARTIFACTS (Grad-CAM):")
        for k, v in gradcam_files.items():
            lines.append(f"  • {k:<15s}: {v}")

    lines.append("=" * 70)
    return "\n".join(lines)


def save_prediction_json(r: Dict[str, Any], output_path: Union[str, Path]) -> str:
    """Serializes the prediction dictionary into a clean JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def sanitize(obj):
        if isinstance(obj, dict):
            return {
                k: sanitize(v)
                for k, v in obj.items()
                if not k.startswith("_") and k not in (
                    "cam_overlay_pil",
                    "cam_heatmap_pil",
                    "cam_grid_pil",
                    "cam_layer_overlays",
                )
            }
        elif isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist() if obj.size < 100 else "<array>"
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist() if obj.numel() < 100 else "<tensor>"
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, Image.Image):
            return "<PIL.Image>"
        return obj

    clean_dict = sanitize(r)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(clean_dict, f, indent=2)

    return str(p.resolve())


class MedicalCTPredictor:
    """Unified predictor with Multi-Layer Gradient Projection explainability."""

    def __init__(self, device: Optional[str] = None):
        self.pipeline = MedicalCTPipeline(device=device)

    def predict(
        self,
        image_path: Union[str, Path],
        gradcam_dir: Optional[Union[str, Path]] = None,
        target_class: Optional[str] = None,
        multi_layer: bool = True,
    ) -> Dict[str, Any]:
        """Runs multi-model inference and multi-layer gradient projection."""
        img_p = Path(image_path).resolve()
        result = self.pipeline.process(img_p)
        result["gradcam_files"] = {}
        result["gradcam_layer_weights"] = {}
        result["is_multi_layer"] = multi_layer

        anatomy = result["router"]["prediction"]
        orig_img = Image.open(str(img_p)).convert("RGB")

        try:
            overlay_pil = None
            heatmap_pil = None
            grid_pil = None
            layer_overlays = {}
            layer_weights = {}

            # ---------------- Multi-Layer Grad-CAM for Chest CT ----------------
            if anatomy == "CHEST_CT" and self.pipeline.chest_expert.is_loaded:
                chest_model = self.pipeline.chest_expert.model
                from grad_cam import (
                    MultiLayerGradCAM as ChestGradCAM,
                    project_to_original,
                    overlay as chest_overlay,
                    choose_crop,
                    make_tensor,
                )

                crop, crop_box, crop_source = choose_crop(orig_img)
                size = 256
                x_tensor = make_tensor(crop, size=size).to(self.pipeline.chest_expert.device)

                extractor = ChestGradCAM(chest_model)
                try:
                    chest_model.zero_grad(set_to_none=True)
                    logits = chest_model(x_tensor)
                    probs_t = torch.softmax(logits, dim=1)[0]
                    if target_class and target_class in self.pipeline.chest_expert.class_names:
                        c_idx = self.pipeline.chest_expert.class_names.index(target_class)
                    else:
                        c_idx = int(torch.argmax(probs_t).item())

                    logits[0, c_idx].backward()
                    fused_model_cam, diagnostics, individual_maps = extractor.build_maps((size, size), mode="layercam")
                finally:
                    extractor.remove()

                # Re-project to exact given image coordinates
                full_cam = project_to_original(fused_model_cam, crop_box, orig_img.size)
                overlay_np, heat_np = chest_overlay(orig_img, full_cam, alpha=0.48, threshold=0.18)
                overlay_pil = Image.fromarray(overlay_np)
                heatmap_pil = Image.fromarray(heat_np)

                # Layer weights and overlays
                grid_layer_info = []
                for d in diagnostics:
                    short_name = d["layer"].replace("backbone.", "")
                    pct = round(d.get("fusion_weight", 0.0) * 100.0, 2)
                    layer_weights[short_name] = pct
                    l_map = individual_maps.get(d["layer"])
                    if l_map is not None:
                        l_full = project_to_original(l_map, crop_box, orig_img.size)
                        l_ov, _ = chest_overlay(orig_img, l_full, alpha=0.48, threshold=0.18)
                        layer_overlays[short_name] = Image.fromarray(l_ov)
                        grid_layer_info.append((short_name, pct, l_full))

                from .gradcam import build_projection_grid
                grid_pil = build_projection_grid(orig_img, grid_layer_info[:4], overlay_pil)

                if gradcam_dir:
                    out_d = Path(gradcam_dir)
                    out_d.mkdir(parents=True, exist_ok=True)
                    ov_path = out_d / "chest_gradcam_overlay.png"
                    heat_path = out_d / "chest_gradcam_heatmap.png"
                    overlay_pil.save(ov_path)
                    heatmap_pil.save(heat_path)
                    result["gradcam_files"]["overlay"] = str(ov_path)
                    result["gradcam_files"]["heatmap"] = str(heat_path)

                    if grid_pil is not None:
                        grid_path = out_d / "chest_gradcam_multilayer_grid.png"
                        grid_pil.save(grid_path)
                        result["gradcam_files"]["multilayer_grid"] = str(grid_path)

                    for l_name, l_img in layer_overlays.items():
                        c_name = re.sub(r"[^a-zA-Z0-9_]", "_", l_name).strip("_")
                        l_path = out_d / f"chest_layer_{c_name}.png"
                        l_img.save(l_path)
                        result["gradcam_files"][f"layer_{c_name}"] = str(l_path)

            # ---------------- Multi-Layer Grad-CAM for Brain CT ----------------
            elif anatomy == "BRAIN_CT" and self.pipeline.brain_expert.is_loaded:
                predictor = self.pipeline.brain_expert.predictor

                # Target class resolution
                brain_labels = getattr(self.pipeline.brain_expert, "labels", ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"])
                if target_class and target_class in brain_labels:
                    target_name = target_class
                else:
                    target_name = "auto"

                from rsna_ich_predict_gradcam import overlay_heatmap
                brain_res = predictor.predict(
                    str(img_p),
                    target_class=target_name,
                    mode="layercam" if multi_layer else "gradcam",
                    threshold=0.22,
                    apply_crop=True,
                )

                cam_full = brain_res["fused_cam"]
                overlay_bgr = brain_res["overlay_bgr"]
                central_slice = brain_res["central_slice"]
                crop_box = brain_res["crop_box_xywh"]
                crop_x, crop_y, crop_w, crop_h = crop_box

                # Convert overlay and heatmap to PIL Images
                overlay_pil = Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
                heat_u8 = np.uint8(np.clip(cam_full, 0, 1) * 255)
                heat_rgb = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                heatmap_pil = Image.fromarray(heat_rgb)

                # Individual layer projections and weights
                diagnostics = brain_res.get("layers", [])
                total_energy = sum(L.get("gradient_energy", 0) * L.get("activation_energy", 0) for L in diagnostics) + 1e-15
                layer_weights = {
                    L["layer"]: round((L.get("gradient_energy", 0) * L.get("activation_energy", 0)) / total_energy * 100, 2)
                    for L in diagnostics
                }

                layer_maps = brain_res.get("layer_maps", [])
                grid_layer_info = []
                for name, m in layer_maps:
                    l_crop = cv2.resize(m, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                    l_full = np.zeros_like(cam_full)
                    l_full[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w] = l_crop
                    l_bgr = overlay_heatmap(central_slice, l_full, alpha=0.5, threshold=0.22)
                    layer_overlays[name] = Image.fromarray(cv2.cvtColor(l_bgr, cv2.COLOR_BGR2RGB))
                    grid_layer_info.append((name, layer_weights.get(name, 25.0), l_full))

                from .gradcam import build_projection_grid
                grid_pil = build_projection_grid(orig_img, grid_layer_info, overlay_pil)

                if gradcam_dir:
                    out_d = Path(gradcam_dir)
                    out_d.mkdir(parents=True, exist_ok=True)
                    ov_path = out_d / "brain_gradcam_overlay.png"
                    heat_path = out_d / "brain_gradcam_heatmap.png"
                    overlay_pil.save(ov_path)
                    heatmap_pil.save(heat_path)
                    result["gradcam_files"]["overlay"] = str(ov_path)
                    result["gradcam_files"]["heatmap"] = str(heat_path)

                    if grid_pil is not None:
                        grid_path = out_d / "brain_gradcam_multilayer_grid.png"
                        grid_pil.save(grid_path)
                        result["gradcam_files"]["multilayer_grid"] = str(grid_path)

                    for l_name, l_img in layer_overlays.items():
                        c_name = re.sub(r"[^a-zA-Z0-9_]", "_", l_name).strip("_")
                        l_path = out_d / f"brain_layer_{c_name}.png"
                        l_img.save(l_path)
                        result["gradcam_files"][f"layer_{c_name}"] = str(l_path)
                        l_img.save(l_path)
                        result["gradcam_files"][f"layer_{c_name}"] = str(l_path)

            result["cam_overlay_pil"] = overlay_pil
            result["cam_heatmap_pil"] = heatmap_pil
            result["cam_grid_pil"] = grid_pil
            result["cam_layer_overlays"] = layer_overlays
            result["gradcam_layer_weights"] = layer_weights

        except Exception as e:
            result["gradcam_error"] = str(e)

        return result

