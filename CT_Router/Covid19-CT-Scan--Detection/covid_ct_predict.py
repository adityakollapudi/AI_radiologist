#!/usr/bin/env python3
"""
COVIDx CT-3A - Multi-Layer Gradient Projection & Grad-CAM Prediction Dashboard

Key Capabilities:
- Multi-Layer Gradient Capture: Hooks every Conv2d layer in the EfficientNet backbone,
  capturing activations and exact class gradients.
- Multi-Layer Gradient Projection: Fuses layer maps weighted by each layer's gradient
  signal energy, eliminating single-layer edge/border noise and resolving true lung lesion attribution.
- Original Grad-CAM JET Colormap: Classic red-hot focal centers, yellow/green mid-tones, and cyan/blue gradients.
- Coordinate Alignment: Accurately maps the model crop CAM back into the original CT slice dimensions.
- Patient Body Isolation: Uses connected-component contour detection to exclude the scanner couch.
- Interactive Terminal Loop: Prompts for CT image path, supports drag-and-drop, and provides fast 'kill' commands.
- Interactive GUI Pop-up: Opens full diagnostic dashboard (press 'q' or 'Esc' to close).
- Zero Disk Footprint: No output files saved by default (optional saving with --output).

Expected Checkpoint:
    covidx_ct_results/checkpoints/best.pt (auto-detected by default)

Usage:
    python covid_ct_predict.py
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import matplotlib
# Headless backend only if explicitly requested
if "--no-gui" in sys.argv or "--headless" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Import multi-layer gradient extraction engine and utilities
from grad_cam import (
    MultiLayerGradCAM,
    project_to_original,
    overlay,
    safe_foreground_bbox,
    crop_image,
    make_tensor,
    read_official_bbox,
    choose_crop,
)

CLASS_NAMES = ["Normal", "Pneumonia", "COVID-19"]


# -----------------------------
# Input & Path Utilities
# -----------------------------
def clean_image_path(raw_path):
    """Cleans up paths entered in terminal, stripping quotes and shell prefixes."""
    if not raw_path:
        return ""
    p = raw_path.strip()
    if p.startswith("& "):
        p = p[2:].strip()
    p = p.strip('"').strip("'")
    return p


# -----------------------------
# Model — Matches B0/B3 Architecture
# -----------------------------
class EfficientNetClassifier(nn.Module):
    def __init__(self, arch="b0", num_classes=3):
        super().__init__()
        if arch == "b0":
            self.backbone = models.efficientnet_b0(weights=None)
        elif arch == "b3":
            self.backbone = models.efficientnet_b3(weights=None)
        else:
            raise ValueError("arch must be b0 or b3")

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


# -----------------------------
# Visualization & Dashboard
# -----------------------------
def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def save_dashboard(
    original,
    overlay_img,
    probs,
    pred_idx,
    mode,
    diagnostics,
    metrics,
    crop_box,
    crop_source,
    image_name="input",
    output_path=None,
    show_window=True,
):
    pred = CLASS_NAMES[pred_idx]
    conf = float(probs[pred_idx])
    order = np.argsort(probs)[::-1]
    margin = float(probs[order[0]] - probs[order[1]])

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[0.8, 4.5, 2.4, 2.4])

    # ---------------- Top Banner ----------------
    top = fig.add_subplot(gs[0, :])
    top.axis("off")
    top.text(0.5, 0.68, f"PREDICTION: {pred}", ha="center", fontsize=27, fontweight="bold")
    top.text(
        0.5, 0.18,
        f"Confidence: {conf*100:.2f}%   |   Top-1 margin: {margin*100:.2f}%   |   Entropy: {entropy(probs):.3f}",
        ha="center",
        fontsize=12,
    )

    # ---------------- Row 1: Original CT ----------------
    a1 = fig.add_subplot(gs[1, :2])
    a1.imshow(original, cmap="gray")
    a1.axis("off")
    a1.set_title("Uploaded CT Image", fontsize=18, fontweight="bold")

    # ---------------- Row 1: Multi-Layer Grad-CAM ----------------
    a2 = fig.add_subplot(gs[1, 2:])
    a2.imshow(overlay_img)
    a2.axis("off")
    a2.set_title(
        f"Multi-Layer {mode.upper()} — {pred}\n"
        "Projected from all Conv2d layer gradients & aligned to CT coordinates",
        fontsize=17,
        fontweight="bold",
    )

    # ---------------- Row 2: Probabilities ----------------
    a3 = fig.add_subplot(gs[2, 0])
    y = np.arange(len(CLASS_NAMES))
    a3.barh(y, probs * 100)
    a3.set_yticks(y)
    a3.set_yticklabels(CLASS_NAMES)
    a3.invert_yaxis()
    a3.set_xlim(0, 100)
    a3.set_xlabel("Probability (%)")
    a3.set_title("Class Probabilities", fontweight="bold")
    for i, p in enumerate(probs):
        a3.text(min(float(p * 100) + 1, 97), i, f"{p*100:.2f}%", va="center")

    # ---------------- Row 2: Diagnostics & Metrics ----------------
    a4 = fig.add_subplot(gs[2, 1])
    a4.axis("off")
    a4.set_title("Test Metrics / Projection", fontweight="bold")
    lines = [
        f"Test Accuracy : {metrics.get('accuracy', np.nan):.4f}",
        f"Macro F1      : {metrics.get('macro_f1', np.nan):.4f}",
        f"Macro Recall  : {metrics.get('macro_recall', np.nan):.4f}",
        f"Macro AUC     : {metrics.get('macro_auc_ovr', np.nan):.4f}",
        "",
        f"CAM Method    : Multi-Layer {mode.upper()}",
        f"Layers Hooked : {len(diagnostics)} Conv2d layers",
        f"Crop Source   : {crop_source}",
        f"Crop Box      : {tuple(map(int, crop_box))}",
    ]
    a4.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)

    # ---------------- Row 2-3: Layer Gradient Weights ----------------
    a5 = fig.add_subplot(gs[2:, 2:])
    top_layers = sorted(diagnostics, key=lambda x: x["fusion_weight"], reverse=True)[:12]
    labels = [d["layer"].replace("backbone.", "") for d in top_layers][::-1]
    vals = [d["fusion_weight"] for d in top_layers][::-1]
    a5.barh(labels, vals)
    a5.set_xlabel("Fusion Weight (Gradient x Activation Energy)")
    a5.set_title("Top Layer Gradient Projection Weights", fontweight="bold")
    a5.tick_params(axis="y", labelsize=8)

    # ---------------- Row 3: Interpretation Notes ----------------
    a6 = fig.add_subplot(gs[3, :2])
    a6.axis("off")
    a6.text(
        0.02, 0.92,
        "Every Conv2d layer contributes a class-specific gradient projection.\n"
        "Maps are normalized, gradient-weighted, fused across scales, and mapped\n"
        "back to original anatomical coordinates, eliminating single-layer edge noise.\n"
        "Grad-CAM represents model attribution, not a definitive clinical diagnosis.",
        va="top",
        fontsize=10,
    )

    fig.suptitle("COVIDx CT-3A — Multi-Layer Gradient-Projected Explainable AI", fontsize=20, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.975])

    # Only save to disk if explicitly requested
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=200, bbox_inches="tight")

    if show_window:
        try:
            fig.canvas.manager.set_window_title(
                f"COVIDx CT-3A: {pred} ({conf*100:.1f}%) — {image_name}"
            )
        except Exception:
            pass

        print(f"\n[+] Pop-up window opened for '{image_name}'.")
        print("    -> Close the result window (or press 'q' or 'Esc') to return to terminal.")

        # Bind 'q' and 'escape' keys to quickly close preview window
        try:
            fig.canvas.mpl_connect(
                "key_press_event",
                lambda evt: plt.close(fig) if evt.key in ("q", "escape") else None,
            )
        except Exception:
            pass

        # Focus / bring window to front
        try:
            mngr = plt.get_current_fig_manager()
            if hasattr(mngr, "window"):
                if hasattr(mngr.window, "attributes"):
                    mngr.window.attributes("-topmost", 1)
                    mngr.window.attributes("-topmost", 0)
                if hasattr(mngr.window, "lift"):
                    mngr.window.lift()
        except Exception:
            pass

        try:
            plt.show(block=True)
        except Exception as e:
            print(f"[!] Interactive window error: {e}.")
            if output_path and sys.platform == "win32":
                try:
                    os.startfile(output_path)
                except Exception:
                    pass
        finally:
            plt.close(fig)
    else:
        plt.close(fig)


# -----------------------------
# Checkpoint & Metrics Auto-Detection
# -----------------------------
def resolve_checkpoint(path=None):
    if path:
        p = Path(clean_image_path(path)).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    candidates = [
        Path(__file__).parent / "covidx_ct_results" / "checkpoints" / "best.pt",
        Path("covidx_ct_results/checkpoints/best.pt"),
        Path(__file__).parent / "checkpoints" / "best.pt",
        Path("checkpoints/best.pt"),
        Path("/kaggle/working/covidx_ct_results/checkpoints/best.pt"),
    ]

    for p in candidates:
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        "Checkpoint not found. Place 'best.pt' in 'covidx_ct_results/checkpoints/' "
        "or provide --checkpoint <path>."
    )


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    arch = args.get("arch", ckpt.get("arch", "b0"))
    names = ckpt.get("class_names", CLASS_NAMES)
    model = EfficientNetClassifier(arch, len(names)).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("WARNING: Missing keys in state_dict:", missing[:5])
    if unexpected:
        print("WARNING: Unexpected keys in state_dict:", unexpected[:5])
    for mod in model.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False
    model.eval()
    return model, arch, names, ckpt


def load_metrics(checkpoint, explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(clean_image_path(explicit)))
    ckpt = Path(checkpoint).resolve()
    candidates += [
        ckpt.parent.parent / "final_results.json",
        ckpt.parent.parent / "results.json",
        Path(__file__).parent / "covidx_ct_results" / "final_results.json",
        Path("covidx_ct_results/final_results.json"),
        Path("/kaggle/working/covidx_ct_results/final_results.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                d = json.loads(p.read_text())
                t = d.get("test", {})
                return {k: t.get(k, np.nan) for k in ["accuracy", "macro_f1", "macro_recall", "macro_auc_ovr"]}
            except Exception:
                pass
    return {}


# -----------------------------
# Prediction Pipeline
# -----------------------------
def predict_single_image(
    image_path,
    model,
    arch,
    class_names,
    device,
    output_path=None,
    model_metrics=None,
    mode="layercam",
    alpha=0.48,
    threshold=0.18,
    no_bbox=False,
    label_file=None,
    filename=None,
    manual_bbox=None,
    bbox_margin=0.04,
    checkpoint_path=None,
    save_components=False,
    show_window=True,
):
    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    print("\n" + "=" * 70)
    print(f"PROCESSING: {image_path.name}")
    print(f"Path      : {image_path}")
    print("=" * 70)

    original = Image.open(image_path).convert("RGB")
    size = 256 if arch == "b0" else 300

    # Select and crop CT region according to priority hierarchy
    crop, crop_box, crop_source = choose_crop(
        original,
        label_file=label_file,
        filename=filename or image_path.name,
        bbox=manual_bbox,
        no_bbox=no_bbox,
        bbox_margin=bbox_margin,
    )

    x = make_tensor(crop, size=size).to(device)

    # Initialize Multi-Layer Grad-CAM extractor (hooks every Conv2d layer)
    extractor = MultiLayerGradCAM(model)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        probs_t = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs_t).item())

        # Single backward pass: all Conv2d layers receive gradients of predicted class
        logits[0, pred_idx].backward()

        # Build fused multi-layer gradient-projected map
        fused_model_cam, diagnostics, _ = extractor.build_maps((size, size), mode=mode)
    finally:
        extractor.remove()

    probs = probs_t.detach().cpu().numpy()
    pred_name = class_names[pred_idx]
    conf = float(probs[pred_idx])
    order = np.argsort(probs)[::-1]
    margin = float(probs[order[0]] - probs[order[1]])

    print("\nPREDICTION SUMMARY")
    print("-" * 70)
    print(f"Predicted class : {pred_name}")
    print(f"Confidence      : {conf*100:.2f}%")
    print(f"Top-1 margin    : {margin*100:.2f}%")
    print(f"Entropy         : {entropy(probs):.4f}")
    print(f"Crop source     : {crop_source}")
    print(f"Crop box        : {tuple(map(int, crop_box))}")

    print("\nCLASS PROBABILITIES")
    print("-" * 70)
    for name, p in zip(class_names, probs):
        bar_len = int(p * 30)
        bar = "=" * bar_len
        print(f"{name:12s}: {p*100:6.2f}%  [{bar:<30}]")

    print(f"\nTOP LAYER GRADIENT CONTRIBUTIONS ({len(diagnostics)} Conv2d layers hooked)")
    print("-" * 70)
    top_diag = sorted(diagnostics, key=lambda d: d["fusion_weight"], reverse=True)[:8]
    for d in top_diag:
        short = d["layer"].replace("backbone.", "")
        print(
            f"{short:24s} "
            f"weight={d['fusion_weight']*100:5.2f}% "
            f"grad_energy={d['mean_abs_gradient']:.5e} "
            f"act_energy={d['mean_abs_activation']:.5e}"
        )

    if model_metrics:
        print("\nCHECKPOINT TEST METRICS")
        print("-" * 70)
        print(f"Accuracy     : {model_metrics.get('accuracy', np.nan):.4f}")
        print(f"Macro F1     : {model_metrics.get('macro_f1', np.nan):.4f}")
        print(f"Macro Recall : {model_metrics.get('macro_recall', np.nan):.4f}")
        print(f"Macro AUC    : {model_metrics.get('macro_auc_ovr', np.nan):.4f}")

    # Project fused model CAM back into original CT slice coordinates
    full_cam = project_to_original(fused_model_cam, crop_box, original.size)
    overlay_img, heatmap = overlay(original, full_cam, alpha=alpha, threshold=threshold)

    # Save / display interactive dashboard
    save_dashboard(
        original=original,
        overlay_img=overlay_img,
        probs=probs,
        pred_idx=pred_idx,
        mode=mode,
        diagnostics=diagnostics,
        metrics=model_metrics,
        crop_box=crop_box,
        crop_source=crop_source,
        image_name=image_path.name,
        output_path=output_path,
        show_window=show_window,
    )

    # Optional saving only if output_path is explicitly requested
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        json_path = out_file.with_suffix(".json")
        prediction_json = {
            "image": str(image_path),
            "checkpoint": str(checkpoint_path) if checkpoint_path else "",
            "architecture": arch,
            "predicted_class": pred_name,
            "confidence": conf,
            "top1_margin": margin,
            "entropy": entropy(probs),
            "probabilities": {name: float(p) for name, p in zip(class_names, probs)},
            "cam_method": f"multi-layer {mode}",
            "layers_used": len(diagnostics),
            "crop_box_original_coordinates": list(map(int, crop_box)),
            "crop_source": crop_source,
            "checkpoint_test_metrics": model_metrics,
            "top_layers": [
                {
                    "layer": d["layer"],
                    "fusion_weight": d["fusion_weight"],
                    "mean_abs_gradient": d["mean_abs_gradient"],
                }
                for d in top_diag
            ],
        }
        json_path.write_text(json.dumps(prediction_json, indent=2))

        if save_components:
            stem = out_file.with_suffix("")
            Image.fromarray(np.asarray(original)).save(str(stem) + "_original.png")
            Image.fromarray(np.uint8(np.clip(full_cam * 255, 0, 255))).save(str(stem) + "_cam.png")
            Image.fromarray(np.uint8(heatmap)).save(str(stem) + "_heatmap.png")
            Image.fromarray(np.uint8(overlay_img)).save(str(stem) + "_overlay.png")

        print("\nSAVED ARTIFACTS")
        print("-" * 70)
        print("Dashboard PNG :", out_file.resolve())
        print("JSON Report   :", json_path.resolve())
        if save_components:
            print("Components    :", f"{stem}_[original|cam|heatmap|overlay].png")
        print("=" * 70)

    return pred_name, conf


# -----------------------------
# Main Entrypoint
# -----------------------------
def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="COVIDx CT prediction + Multi-Layer Gradient-Projected Grad-CAM"
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to best.pt (defaults to covidx_ct_results/checkpoints/best.pt)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to the CT image to test. If omitted, prompts interactively in the terminal.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save dashboard PNG and JSON report (by default, no files are stored)",
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Optional final_results.json (defaults to auto-detecting next to checkpoint)",
    )
    parser.add_argument(
        "--mode",
        choices=["layercam", "gradcam"],
        default="layercam",
        help="Gradient projection mode: 'layercam' (element-wise gradient weighting) or 'gradcam'",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.48,
        help="Grad-CAM overlay blending strength",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.18,
        help="Activation threshold below which colormap is suppressed to keep normal anatomy clear",
    )
    parser.add_argument(
        "--label-file",
        default=None,
        help="Optional COVIDx label txt file for official bbox",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Optional filename to look up in --label-file (defaults to input image basename)",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional manual bounding box",
    )
    parser.add_argument(
        "--bbox-margin",
        type=float,
        default=0.04,
        help="Margin expansion factor around bounding box (default: 0.04)",
    )
    parser.add_argument(
        "--save-components",
        action="store_true",
        help="When --output is specified, also save raw _original.png, _cam.png, _heatmap.png, and _overlay.png",
    )
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="Disable bbox crop and use full original image",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable interactive pop-up window (headless mode)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    print("=" * 70)
    print("COVIDx CT-3A PREDICTION + MULTI-LAYER GRADIENT PROJECTION")
    print("=" * 70)
    print("Device     :", device)
    print("Checkpoint :", checkpoint_path)

    model, arch, class_names, checkpoint = load_model(
        checkpoint_path,
        device,
    )

    print("Architecture:", arch)
    print("Classes     :", class_names)
    print("Epoch       :", checkpoint.get("epoch", "unknown"))

    model_metrics = load_metrics(
        checkpoint_path,
        args.results_json,
    )

    # 1. Single run via command-line argument
    if args.image:
        target_img = clean_image_path(args.image)
        predict_single_image(
            image_path=target_img,
            model=model,
            arch=arch,
            class_names=class_names,
            device=device,
            output_path=args.output,
            model_metrics=model_metrics,
            mode=args.mode,
            alpha=args.alpha,
            threshold=args.threshold,
            no_bbox=args.no_bbox,
            label_file=args.label_file,
            filename=args.filename,
            manual_bbox=args.bbox,
            bbox_margin=args.bbox_margin,
            checkpoint_path=checkpoint_path,
            save_components=args.save_components,
            show_window=not args.no_gui,
        )
        return

    # 2. Interactive terminal mode: prompt user for image address
    print("\n" + "=" * 70)
    print("INTERACTIVE TERMINAL MODE")
    print("Tip: You can paste the image file address or drag & drop the image into this terminal.")
    print("Commands to kill/exit: Type 'kill', 'exit', 'quit', 'q', or press Ctrl+C.")
    print("=" * 70)

    KILL_COMMANDS = {
        "kill",
        "kill terminal",
        "k",
        "exit",
        "quit",
        "q",
        "stop",
        "close",
        "terminate",
        "abort",
        ":q",
        "/exit",
        "/quit",
    }

    while True:
        try:
            user_input = input("\nEnter CT image address/path (or type 'kill' to exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[*] Interrupt signal received (Ctrl+C). Terminal session killed. Goodbye!")
            sys.exit(0)

        if not user_input:
            continue

        cleaned = clean_image_path(user_input)
        if cleaned.lower() in KILL_COMMANDS:
            print("\n[*] 'kill' command received. Terminating process and killing terminal session. Goodbye!")
            sys.exit(0)

        img_p = Path(cleaned)
        if not img_p.is_file():
            print(f"[!] File not found: '{cleaned}'")
            print("    Please check the address or drag & drop the image file into the terminal.")
            print("    (Or type 'kill' to exit the terminal session)")
            continue

        try:
            predict_single_image(
                image_path=img_p,
                model=model,
                arch=arch,
                class_names=class_names,
                device=device,
                output_path=args.output,
                model_metrics=model_metrics,
                mode=args.mode,
                alpha=args.alpha,
                threshold=args.threshold,
                no_bbox=args.no_bbox,
                label_file=args.label_file,
                filename=args.filename,
                manual_bbox=args.bbox,
                bbox_margin=args.bbox_margin,
                checkpoint_path=checkpoint_path,
                save_components=args.save_components,
                show_window=not args.no_gui,
            )
        except Exception as e:
            print(f"[!] Error processing image: {e}")


if __name__ == "__main__":
    main()
