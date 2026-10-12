# MRI Router: Cascade Brain MRI Diagnosis System

A comprehensive deep learning diagnostic system for brain MRI slice analysis. The router implements a clinically motivated **Two-Stage Cascade Architecture**:
1. **Stage 1 (Primary - Brain Tumor Screening)**: Evaluates brain scans using a **Custom 5-Layer CNN** with **5-layer gradient-fused Grad-CAM** (`conv0`, `bn0`, `activation`, `max_pool0`, `max_pool1`) for acute intracranial mass / tumor screening.
2. **Stage 2 (Reflex - Dementia Severity Assessment)**: Triggered **ONLY** when Stage 1 indicates a normal brain (*No Tumor Detected*). If a brain tumor is detected, dementia assessment is bypassed to prevent gross mass effect and edema from confounding subtle cortical atrophy evaluation. Evaluates categories (*Non Demented*, *Very mild Dementia*, *Mild Dementia*, *Moderate Dementia*) using **EfficientNetV2-S**.

---

## Architecture Overview

```
                      [Input Brain MRI Slice]
                                 │
                                 ▼
                 Stage 1: Brain Tumor Screener (CNN)
                                 │
               ┌─────────────────┴─────────────────┐
               ▼                                   ▼
      [Tumor Detected (YES)]               [No Tumor (Normal)]
               │                                   │
               │ (Acute pathology identified)      ▼
               ▼                     Stage 2: Dementia Model (EfficientNetV2-S)
        Stage 2 Bypassed                           │
       (Screening Skipped)                 ┌───────┴───────┐
                                           ▼               ▼
                                   [Dementia Detected] [Non-Demented]
                                   (Very Mild / Mild)  (Normal Brain)
```

---

## Quick Start

### 1. Launch the Interactive GUI Window (Recommended)
You can directly launch the graphical user interface by running:

```bash
python predict.py
```
*(or `python gui_predict.py`)*

In the GUI, simply click **Browse Image...** or paste your image path and click **Run Cascade Diagnosis**.

### 2. Run via Command Line (Terminal Mode)
```bash
python predict.py --cli --image "path/to/mri.jpg"
```

---

## Directory Structure

```
MRI_Router/
├── predict.py                  # Global entrypoint (launches GUI or CLI)
├── gui_predict.py              # Interactive desktop GUI application
├── with dementia.jpeg          # Sample MRI scan (Dementia positive)
├── with out dementia.jpeg      # Sample MRI scan (Dementia negative)
│
├── dementia_mri_project/       # Dementia classification components
│   ├── checkpoints/            # Trained model weights (best_model.pth)
│   ├── src/                    # Data pipeline, training & inference code
│   │   ├── gui.py              # Cascade GUI implementation
│   │   ├── inference.py        # Dementia inference engine
│   │   ├── tumor_detector.py   # Stage 1 PyTorch screening engine
│   │   └── ...
│   └── predictions/            # Generated Grad-CAM heatmaps & JSON outputs
│
└── brain-_tumor_detection/     # Brain Tumor detection components
    ├── models/                 # Pretrained weights (.h5 / .model)
    ├── gradcam.py              # Multi-layer Grad-CAM algorithm
    └── ...
```

---

## Explainable AI (XAI)
- **Stage 1**: 5-layer gradient-fused Grad-CAM (`conv0`, `bn0`, `activation`, `max_pool0`, `max_pool1`) with pre-sigmoid logit backpropagation and anatomical skull masking.
- **Stage 2**: Multi-layer Grad-CAM for cortical atrophy and dementia patterns.
