#!/usr/bin/env python
"""Global Entry Point: Medical CT Multi-Model Triage GUI Dashboard Window.

Stage 1: CT Anatomy Router (EfficientNet-B0) -> Brain CT vs Chest CT
Stage 2: Downstream Specialist Expert:
         - Chest Expert (COVID-19 / Pneumonia Detection)
         - Brain Expert (Intracranial Hemorrhage Detection with Multi-Layer Grad-CAM)

Usage:
    python gui_predict.py
    python gui_predict.py --image test_brain.png
"""
import sys
import argparse
from pathlib import Path

# Resolve top-level workspace root and subproject folders
ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "ct_anatomy_project"
COVID_DIR = ROOT / "Covid19-CT-Scan--Detection"
RSNA_DIR = ROOT / "RSNA2019_Intracranial-Hemorrhage-Detection"

for d in (PROJECT_DIR, ROOT, COVID_DIR, RSNA_DIR):
    s = str(d)
    if s not in sys.path:
        sys.path.insert(0, s)

from src.gui import MedicalCTGUI
from src.inference import MedicalCTPredictor


def main():
    ap = argparse.ArgumentParser(description="Medical CT Multi-Model Triage GUI Dashboard")
    ap.add_argument("--image", default=None, help="Initial CT slice (PNG/JPG/DICOM).")
    ap.add_argument("positional_image", nargs="?", default=None, help="Optional positional image path.")
    ap.add_argument("--device", default=None, help="Hardware device ('cuda' or 'cpu'). Default: auto.")
    a = ap.parse_args()

    target_image = a.image or a.positional_image
    p = MedicalCTPredictor(device=a.device)
    app = MedicalCTGUI(initial_image=target_image, predictor=p)
    app.mainloop()


if __name__ == "__main__":
    main()
