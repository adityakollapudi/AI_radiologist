#!/usr/bin/env python
"""Global Entry Point: Cascade Brain MRI Diagnosis.

Running this file automatically accesses and launches the interactive GUI Window (gui_predict).

Usage:
    python predict.py
    python predict.py --image "path/to/scan.jpg"
    python predict.py --cli   (optional: run in terminal text mode)
"""
import sys
from pathlib import Path

# Resolve workspace folders
ROOT = Path(__file__).resolve().parent
DEMENTIA_DIR = ROOT / "dementia_mri_project"
TUMOR_DIR = ROOT / "brain-_tumor_detection"

sys.path.insert(0, str(DEMENTIA_DIR))
sys.path.insert(0, str(TUMOR_DIR))

def main():
    # If explicitly requested with --cli, run terminal mode
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        import dementia_mri_project.predict as sub_predict
        sub_predict.main()
    else:
        # Default: access gui_predict and launch the interactive GUI window
        from src.gui import main as run_gui
        run_gui()

if __name__ == "__main__":
    main()
