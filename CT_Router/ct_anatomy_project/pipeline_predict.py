#!/usr/bin/env python
"""Single-image inference and GUI Dashboard for the Medical CT Multi-Model Triage Pipeline.

Usage:
    # Launch interactive GUI window (default)
    python pipeline_predict.py
    python pipeline_predict.py --image path/to/scan.png

    # Pure console / CLI mode without GUI window
    python pipeline_predict.py --cli --image path/to/scan.png
    python pipeline_predict.py --cli --image scan.png --no-gradcam --output-dir predictions

Outputs a console report, prediction.json and Grad-CAM images. Not a diagnostic device.
"""
import argparse
import sys
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.inference import MedicalCTPredictor, format_report, save_prediction_json  # noqa: E402

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
