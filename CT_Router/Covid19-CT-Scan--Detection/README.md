# COVID-19 CT-Scan Detection & Multi-Layer Grad-CAM Explainable AI

Deep learning diagnosis and interpretability system for **COVID-19**, **Pneumonia**, and **Normal** lung classification from chest computed tomography (CT) slices using an **EfficientNet-B0** backbone trained on the COVIDx CT-3A dataset.

---

## Key Features

- **Multi-Layer Gradient Projection**:
  - Hooks all `Conv2d` layers across the EfficientNet backbone to extract activations and exact target class gradients.
  - Applies element-wise positive gradient projection (Layer-CAM principle) fused with multi-scale gradient-activation energy weighting.
  - Eliminates single-layer edge noise and reveals true focal lesion attributions.
- **Original Grad-CAM Colormap (JET)**:
  - Uses the classic Grad-CAM colormap palette: vibrant **Red** and **Orange** hotspots at focal lesion sites, transitioning through **Yellow**, **Green**, and **Cyan/Blue**.
  - Soft thresholded alpha blending keeps healthy lung parenchyma clear and readable without artificial blue haze.
- **Patient Body Isolation**:
  - Connected-component contour extraction isolates the patient's body and excludes the scanner couch, peripheral artifacts, and empty air margins.
- **Interactive Terminal Workflow**:
  - Accepts image paths via typing or direct drag-and-drop.
  - Supports rapid process termination with `kill`, `exit`, `quit`, or `Ctrl+C`.
- **Interactive Diagnostic Window**:
  - Instant pop-up GUI displaying uploaded CT, aligned Grad-CAM overlay, prediction confidence, class probability bars, and layer gradient contribution diagnostics.
  - Press `q` or `Esc` to instantly close the preview and test another slice.
- **Zero Disk Footprint**:
  - Operates completely in-memory by default without cluttering disk space with temporary files.

---

## Model Performance

Evaluated on the COVIDx CT-3A test set:
- **Accuracy**: `90.92%`
- **Macro F1**: `89.73%`
- **Macro Recall**: `90.54%`
- **Macro AUC (OvR)**: `0.9823`

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/adityakollapudi/Covid19-CT-Scan--Detection.git
cd Covid19-CT-Scan--Detection
pip install -r requirements.txt
```

---

## Quick Start

Launch the interactive prediction session:

```bash
python covid_ct_predict.py
```

### Drag-and-Drop / Interactive Mode
When prompted:
```text
Enter CT Image Path (or 'kill' to exit) > "C:\path\to\scan.png"
```
The model will analyze the slice, print clinical diagnostic metrics, and open an interactive visualization window. Press `q` or `Esc` to close the window and analyze another slice.

### Command-Line Arguments

Run directly on a single image without prompts:
```bash
python covid_ct_predict.py --image path/to/ct_scan.png
```

Optional parameters:
- `--checkpoint`: Path to custom model weights (defaults to `covidx_ct_results/checkpoints/best.pt`).
- `--alpha`: Grad-CAM overlay transparency (default `0.48`).
- `--threshold`: Activation suppression threshold (default `0.18`).
- `--output`: Save diagnostic report figure and JSON summary to disk.
- `--no-gui`: Run in headless mode (console output only).

---

## Project Structure

```text
├── covid_ct_predict.py       # Main prediction pipeline & interactive dashboard
├── grad_cam.py               # Multi-layer gradient projection & JET colormap engine
├── covidx_ct_results/
│   ├── final_results.json    # Test benchmark metrics
│   └── checkpoints/
│       ├── best.pt           # Trained EfficientNet-B0 checkpoint (best validation loss)
│       └── last.pt           # Final training epoch checkpoint
├── requirements.txt          # Python dependencies
└── .gitignore                # Git exclusions
```

---

## License

This project is licensed under the MIT License.
