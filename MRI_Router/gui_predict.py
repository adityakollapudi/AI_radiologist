#!/usr/bin/env python
"""Global Entry Point: Cascade Brain MRI Diagnosis GUI Window.

Stage 1: Brain Tumor Screening (Custom CNN with Multi-Layer Grad-CAM)
Stage 2: Reflex Dementia Assessment (EfficientNetV2-S)
         - Automatically triggered ONLY when Brain Tumor is Normal (No Tumor)
         - Bypassed when Brain Tumor is detected

Usage:
    python gui_predict.py
    python gui_predict.py --image "path/to/scan.jpg"
"""
import os
import sys
from pathlib import Path

# Resolve top-level workspace root and subproject folders
ROOT = Path(__file__).resolve().parent
DEMENTIA_DIR = ROOT / "dementia_mri_project"
TUMOR_DIR = ROOT / "brain-_tumor_detection"

sys.path.insert(0, str(DEMENTIA_DIR))
sys.path.insert(0, str(TUMOR_DIR))

from src.gui import main

if __name__ == "__main__":
    main()
