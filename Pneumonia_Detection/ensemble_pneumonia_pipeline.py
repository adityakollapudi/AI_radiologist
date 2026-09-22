#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pneumonia_Detection/ensemble_pneumonia_pipeline.py
=================================================
Multi-Model Ensemble Pipeline for Chest X-Ray Pneumonia Detection.
Integrates multiple deep learning architectures (EfficientNet-V2-S,
EfficientNet-B3, EfficientNet-B7) trained on chest radiographs.

Features:
- Multi-resolution inputs (300px, 384px, 600px)
- Calibrated weighted ensemble and consensus voting
- Grad-CAM attention heatmap from top-performing models
- Full interactive popup dashboard & JSON serialization
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
OUTPUT_DIR = ROOT_DIR / "pipeline_outputs"
GRADCAM_CMAP = "jet"
GRADCAM_ALPHA = 0.45

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


class EnsembleGradCAM:
    """Device-agnostic hook-based Grad-CAM generator."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []

    def __enter__(self):
        self.handles.append(
            self.target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o.detach())
            )
        )
        self.handles.append(
            self.target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0].detach())
            )
        )
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
            logit = self.model(x)
            score = (signed_target * logit).sum()
            self.model.zero_grad(set_to_none=True)
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks captured nothing. Invalid target layer.")

        acts = self.activations[0]
        grads = self.gradients[0]
        alpha = grads.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((alpha * acts).sum(dim=0))
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else torch.zeros_like(cam)

        size = output_size or (x.shape[-2], x.shape[-1])
        cam = F.interpolate(cam[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
        return cam.detach().cpu().numpy()


class PneumoniaEnsemble:
    """Multi-Model Pneumonia Detection Ensemble Manager."""
    def __init__(self, base_dir=None, device=None):
        self.base_dir = Path(base_dir or BASE_DIR).resolve()
        self.device = device or DEVICE
        self.models_info = {}
        self._discover_and_load_models()

    def _discover_and_load_models(self):
        print("\n" + "=" * 76)
        print("LOADING PNEUMONIA MULTI-MODEL ENSEMBLE")
        print("=" * 76)
        print(f"Device : {self.device}")

        def _is_lfs_pointer(p):
            try:
                return p.stat().st_size < 1024
            except Exception:
                return False

        # Model 1: EfficientNet-V2-S (pneumonia_runs/v2_s)
        v2_path = self.base_dir / "pneumonia_runs" / "v2_s" / "best.pth"
        if not v2_path.exists():
            print(f" [MISSING] EfficientNet-V2-S checkpoint not found at: {v2_path}")
        elif _is_lfs_pointer(v2_path):
            print(f" [!] EfficientNet-V2-S is an unpulled Git LFS pointer ({v2_path.stat().st_size} bytes). Run: git lfs pull")
        else:
            try:
                ckpt = torch.load(v2_path, map_location=self.device, weights_only=False)
                m = models.efficientnet_v2_s()
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
                m.load_state_dict(ckpt["model_state_dict"])
                m.eval().to(self.device)
                self.models_info["EfficientNet-V2-S"] = {
                    "model": m,
                    "size": 384,
                    "threshold": float(ckpt.get("threshold", 0.655)),
                    "weight": 0.40,
                    "test_f1": 0.9428,
                    "target_layer": m.features[-1],
                    "checkpoint": str(v2_path),
                }
                print(" [OK] Loaded EfficientNet-V2-S  | Res: 384x384 | Weight: 0.40 | Threshold: 0.655")
            except Exception as e:
                print(f" [!] Could not load EfficientNet-V2-S: {e}")

        # Model 2: EfficientNet-B3 (pneumonia_checkpoints)
        b3_path = self.base_dir / "pneumonia_checkpoints" / "pneumonia_efficientnet_b3_best.pth"
        if not b3_path.exists():
            print(f" [MISSING] EfficientNet-B3 checkpoint not found at: {b3_path}")
        elif _is_lfs_pointer(b3_path):
            print(f" [!] EfficientNet-B3 is an unpulled Git LFS pointer ({b3_path.stat().st_size} bytes). Run: git lfs pull")
        else:
            try:
                ckpt = torch.load(b3_path, map_location=self.device, weights_only=False)
                m = models.efficientnet_b3()
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
                state = {k.replace("backbone.", ""): v for k, v in ckpt["model_state_dict"].items()}
                m.load_state_dict(state)
                m.eval().to(self.device)
                self.models_info["EfficientNet-B3"] = {
                    "model": m,
                    "size": 300,
                    "threshold": float(ckpt.get("threshold", 0.75)),
                    "weight": 0.35,
                    "test_f1": 0.9323,
                    "target_layer": m.features[-1],
                    "checkpoint": str(b3_path),
                }
                print(" [OK] Loaded EfficientNet-B3    | Res: 300x300 | Weight: 0.35 | Threshold: 0.750")
            except Exception as e:
                print(f" [!] Could not load EfficientNet-B3: {e}")

        # Model 3: EfficientNet-B7 (pneumonia_runs/b7)
        b7_path = self.base_dir / "pneumonia_runs" / "b7" / "best.pth"
        if not b7_path.exists():
            print(f" [MISSING] EfficientNet-B7 checkpoint not found at: {b7_path}")
        elif _is_lfs_pointer(b7_path):
            print(f" [!] EfficientNet-B7 is an unpulled Git LFS pointer ({b7_path.stat().st_size} bytes). Run: git lfs pull")
        else:
            try:
                ckpt = torch.load(b7_path, map_location=self.device, weights_only=False)
                m = models.efficientnet_b7()
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
                m.load_state_dict(ckpt["model_state_dict"])
                m.eval().to(self.device)
                self.models_info["EfficientNet-B7"] = {
                    "model": m,
                    "size": 600,
                    "threshold": float(ckpt.get("threshold", 0.95)),
                    "weight": 0.25,
                    "test_f1": 0.9264,
                    "target_layer": m.features[-1],
                    "checkpoint": str(b7_path),
                }
                print(" [OK] Loaded EfficientNet-B7    | Res: 600x600 | Weight: 0.25 | Threshold: 0.950")
            except Exception as e:
                print(f" [!] Could not load EfficientNet-B7: {e}")

        total_models = len(self.models_info)
        print(f" [INFO] Active Ensemble: {total_models} of 3 models loaded ({', '.join(self.models_info.keys())})")
        if not self.models_info:
            raise RuntimeError("No valid pneumonia checkpoints found in pneumonia_runs or pneumonia_checkpoints!")

    def predict(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        gray = image.convert("L")
        rgb = gray.convert("RGB")
        orig_np = np.asarray(gray, dtype=np.float32) / 255.0

        model_results = {}
        total_weight = sum(info["weight"] for info in self.models_info.values())
        weighted_pneu_prob = 0.0
        votes = {"PNEUMONIA": 0, "NORMAL": 0}

        tensors = {}
        for name, info in self.models_info.items():
            size = info["size"]
            tf = T.Compose([
                T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=NORM_MEAN, std=NORM_STD)
            ])
            x = tf(rgb).unsqueeze(0).to(self.device)
            tensors[name] = x

            with torch.no_grad():
                logit = float(info["model"](x).reshape(-1)[0].item())
                p_pneu = float(torch.sigmoid(torch.tensor(logit)).item())
                p_norm = 1.0 - p_pneu

            thresh = info["threshold"]
            pred = "PNEUMONIA" if p_pneu >= thresh else "NORMAL"
            conf = p_pneu if pred == "PNEUMONIA" else p_norm
            votes[pred] += 1

            norm_w = info["weight"] / total_weight
            weighted_pneu_prob += norm_w * p_pneu

            model_results[name] = {
                "prediction": pred,
                "confidence": conf * 100.0,
                "pneumonia_probability": p_pneu * 100.0,
                "normal_probability": p_norm * 100.0,
                "threshold": thresh,
                "logit": logit,
                "resolution": f"{size}x{size}",
                "weight": info["weight"],
            }

        weighted_norm_prob = 1.0 - weighted_pneu_prob
        ensemble_pred = "PNEUMONIA" if votes["PNEUMONIA"] > votes["NORMAL"] else "NORMAL"
        ensemble_conf = weighted_pneu_prob if ensemble_pred == "PNEUMONIA" else weighted_norm_prob

        # Select primary model for Grad-CAM
        primary_model_name = "EfficientNet-V2-S" if "EfficientNet-V2-S" in self.models_info else list(self.models_info.keys())[0]
        signed_target = 1.0 if ensemble_pred == "PNEUMONIA" else -1.0
        primary_info = self.models_info[primary_model_name]
        primary_tensor = tensors[primary_model_name]

        with EnsembleGradCAM(primary_info["model"], primary_info["target_layer"]) as cam_engine:
            cam = cam_engine.generate(primary_tensor, self.device, signed_target=signed_target, output_size=orig_np.shape)

        return {
            "prediction": ensemble_pred,
            "confidence": ensemble_conf * 100.0,
            "negative_class": "NORMAL",
            "positive_class": "PNEUMONIA",
            "negative_probability": weighted_norm_prob * 100.0,
            "positive_probability": weighted_pneu_prob * 100.0,
            "threshold": 0.50,
            "votes": votes,
            "consensus_ratio": f"{votes[ensemble_pred]}/{len(self.models_info)}",
            "models": model_results,
            "primary_cam_model": primary_model_name,
            "cam": cam,
            "gray_image": gray,
            "original_np": orig_np,
        }

    def print_report(self, result):
        print("\n" + "=" * 76)
        print("PNEUMONIA MULTI-MODEL ENSEMBLE RESULTS")
        print("=" * 76)
        print(f"FINAL CONSENSUS : {result['prediction']}")
        print(f"CONFIDENCE      : {result['confidence']:.2f}%")
        print(f"CONSENSUS RATIO : {result['consensus_ratio']} Models Agree")
        print(f"P(Pneumonia)    : {result['positive_probability']:.2f}%")
        print(f"P(Normal)       : {result['negative_probability']:.2f}%\n")
        print(f"{'Model Name':<20} | {'Res':<8} | {'Prediction':<10} | {'P(Pneumonia)':<12} | {'Thresh':<6}")
        print("-" * 76)
        for name, r in result["models"].items():
            print(f"{name:<20} | {r['resolution']:<8} | {r['prediction']:<10} | {r['pneumonia_probability']:>10.2f}% | {r['threshold']:<6.3f}")
        print("=" * 76)

    def render_and_save_figure(self, result, image_path, output_path=None):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        img_stem = Path(image_path).stem
        output_path = output_path or (OUTPUT_DIR / f"{img_stem}_ensemble_pneumonia.png")

        orig = result["original_np"]
        cam = result["cam"]
        pred = result["prediction"]
        conf = result["confidence"]
        pred_color = "#c0392b" if pred == "PNEUMONIA" else "#27ae60"

        fig, axes = plt.subplots(1, 2, figsize=(16, 9))

        # 1. Original X-Ray
        axes[0].imshow(orig, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("1. ORIGINAL INPUT X-RAY", fontsize=15, fontweight="bold", pad=12)
        axes[0].axis("off")

        # 2. Grad-CAM Overlay
        axes[1].imshow(orig, cmap="gray", vmin=0, vmax=1)
        heat = axes[1].imshow(cam, cmap=GRADCAM_CMAP, alpha=GRADCAM_ALPHA, vmin=0, vmax=1)
        axes[1].set_title("2. ENSEMBLE GRAD-CAM ATTENTION OVERLAY", fontsize=15, fontweight="bold", pad=12)
        axes[1].axis("off")

        cbar = fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Model Attention", fontsize=11)

        # Super title & Headers
        fig.suptitle("AI RADIOLOGIST - MULTI-MODEL PNEUMONIA ENSEMBLE REPORT", fontsize=21, fontweight="bold", y=0.96)
        fig.text(0.5, 0.910, f"IMAGE TYPE : CHEST X-RAY    |    CONSENSUS : {pred} ({result['consensus_ratio']} AGREE)    |    CONFIDENCE : {conf:.2f}%",
                 ha="center", fontsize=14, fontweight="bold", color=pred_color)

        breakdown_str = "    |    ".join([
            f"{name.replace('EfficientNet-', '')} ({r['resolution']}): {r['prediction']} ({r['pneumonia_probability']:.1f}%)"
            for name, r in result["models"].items()
        ])
        fig.text(0.5, 0.865, breakdown_str, ha="center", fontsize=11, color="#333333")

        fig.text(0.5, 0.025, f"Ensemble : {len(result['models'])} Connected Models (V2-S, B3, B7) | Grad-CAM Source: {result['primary_cam_model']}",
                 ha="center", fontsize=10, style="italic", color="#555555")

        plt.subplots_adjust(left=0.04, right=0.96, top=0.81, bottom=0.07, wspace=0.08)

        output_str = str(output_path)
        try:
            fig.savefig(output_str, dpi=200, bbox_inches="tight")
        except Exception:
            import time
            alt_path = output_path.parent / f"{output_path.stem}_{int(time.time())}.png"
            fig.savefig(str(alt_path), dpi=200, bbox_inches="tight")
            output_path = alt_path

        print(f"\n[figure] saved -> {output_path}")

        try:
            if hasattr(fig.canvas, "manager") and fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title(f"AI Radiologist Ensemble - {pred} ({conf:.1f}%)")
        except Exception:
            pass

        plt.show()
        plt.close(fig)
        return output_path


def run_ensemble(image_path):
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ensemble = PneumoniaEnsemble()
    result = ensemble.predict(image_path)
    ensemble.print_report(result)
    fig_path = ensemble.render_and_save_figure(result, image_path)

    json_path = OUTPUT_DIR / f"{image_path.stem}_ensemble_pneumonia.json"
    json_export = {
        "status": "success",
        "image": str(image_path),
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 2),
        "positive_probability": round(result["positive_probability"], 2),
        "negative_probability": round(result["negative_probability"], 2),
        "consensus_ratio": result["consensus_ratio"],
        "models": result["models"],
        "figure_path": str(fig_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)
    print(f"[json] saved -> {json_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pneumonia Multi-Model Ensemble Pipeline")
    parser.add_argument("image", nargs="?", help="Path to chest X-ray image")
    args = parser.parse_args()

    image_input = args.image
    if not image_input:
        image_input = input("\nEnter the full path of the chest X-ray image: ").strip().strip('"')

    run_ensemble(image_input)
