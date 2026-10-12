#!/usr/bin/env python
"""predict.py: Unified CLI & GUI diagnostic triage entrypoint with multi-layer Grad-CAM.

Runs end-to-end pipeline analysis:
    Stage 1: CT Anatomy Router (EfficientNet-B0) -> BRAIN_CT vs CHEST_CT
    Stage 2: Specialist Expert:
             - Chest Expert: Normal / Pneumonia / COVID-19 classification
             - Brain Expert: 6 Acute Intracranial Hemorrhage subtype classification
    Explainability: Multi-layer gradient extraction with native coordinate reprojection

Usage:
    # Interactive GUI window (default)
    python predict.py
    python predict.py --image test_brain.png

    # CLI mode
    python predict.py --cli --image test_brain.png
    python predict.py --cli --image test_chest.png
"""
import argparse
import sys
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "ct_anatomy_project"
COVID_DIR = ROOT / "Covid19-CT-Scan--Detection"
RSNA_DIR = ROOT / "RSNA2019_Intracranial-Hemorrhage-Detection"

for d in (PROJECT_DIR, ROOT, COVID_DIR, RSNA_DIR):
    s = str(d)
    if s not in sys.path:
        sys.path.insert(0, s)

from src.inference import MedicalCTPredictor, format_report, save_prediction_json

# Ensure clean UTF-8 console output on Windows
if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def clean_image_path(raw_path: str) -> str:
    """Sanitizes user input from terminal, stripping quotes, powershell prefixes, and carriage returns."""
    if not raw_path:
        return ""
    p = raw_path.replace("\r", "").replace("\n", "").strip()
    if p.startswith("& "):
        p = p[2:].strip()
    p = p.strip('"').strip("'").strip()
    return p


def process_and_report(
    img_path: Path,
    predictor: MedicalCTPredictor,
    output_dir: Path,
    no_gradcam: bool,
    target_class: str = None,
    multi_layer: bool = True,
):
    """Processes a single CT slice, prints the formatted report, and saves prediction.json."""
    safe_name = f"{re.sub(r'[^a-zA-Z0-9_\-]', '_', img_path.stem)[:18]}_{hashlib.md5(img_path.stem.encode('utf-8')).hexdigest()[:8]}"
    out = output_dir / safe_name
    out.mkdir(parents=True, exist_ok=True)

    r = predictor.predict(
        img_path,
        gradcam_dir=None if no_gradcam else out,
        target_class=target_class,
        multi_layer=multi_layer,
    )

    print("\n" + format_report(r))
    js = save_prediction_json(r, out / "prediction.json")
    print(f"\nprediction.json -> {js}\n")
    return r


def main():
    ap = argparse.ArgumentParser(description="Medical CT Multi-Model Triage Pipeline (Research Prototype)")
    ap.add_argument("--image", default=None, help="CT slice (PNG/JPG/DICOM). If omitted or when running interactively, opens GUI window.")
    ap.add_argument("positional_image", nargs="?", default=None, help="Optional positional image path.")
    ap.add_argument("--output-dir", default=str(ROOT / "predictions"), help="Directory where prediction.json and Grad-CAM maps are saved.")
    ap.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM explainability generation.")
    ap.add_argument("--single-layer", action="store_true", help="Standard (final-layer) Grad-CAM only (default is multi-layer gradient projection).")
    ap.add_argument("--target-class", default=None, help="Target class to explain with Grad-CAM instead of predicted one.")
    ap.add_argument("--device", default=None, help="Hardware device ('cuda' or 'cpu'). Default: auto.")
    ap.add_argument("--gui", action="store_true", default=False, help="Launch interactive GUI window displaying results and Grad-CAM.")
    ap.add_argument("--cli", action="store_true", default=False, help="Run only in console mode without opening GUI window.")
    a = ap.parse_args()

    target_image = a.image or a.positional_image

    # Launch GUI window by default unless --cli is explicitly requested
    if not a.cli:
        from src.gui import MedicalCTGUI
        p = MedicalCTPredictor(device=a.device)
        app = MedicalCTGUI(initial_image=target_image, predictor=p)
        app.mainloop()
        return

    # CLI Mode execution
    if target_image is None:
        sys.exit("Error: --image is required in CLI mode. Run without '--cli' to open the GUI window.")

    cleaned_path = clean_image_path(target_image)
    img = Path(cleaned_path)
    if not img.exists():
        sys.exit(f"Error: Image not found: {img}")

    output_base = Path(a.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    p = MedicalCTPredictor(device=a.device)
    process_and_report(img, p, output_base, a.no_gradcam, a.target_class, multi_layer=not a.single_layer)


if __name__ == "__main__":
    main()
