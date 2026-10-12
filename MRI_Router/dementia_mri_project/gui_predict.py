#!/usr/bin/env python
"""Launch the Dementia MRI Interactive Popup Window.

Usage:
    python gui_predict.py
    python gui_predict.py --image "path/to/mri.jpg"
    python gui_predict.py --image "gradcam/Mild_Dementia/correct_00_Mild Dementia_augmented_2461_original.png"
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.gui import main

if __name__ == "__main__":
    main()
