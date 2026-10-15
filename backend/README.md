# MEDVISION AI — Unified Backend

Unified FastAPI diagnostic backend connecting deep learning models for **X-Ray**, **Computed Tomography (CT)**, and **Magnetic Resonance Imaging (MRI)** into a single REST API.

---

## 1. Architecture Overview

```
                      FastAPI Application (Port 8000)
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
     X-Ray Service              CT Service              MRI Service
   (xray_service.py)         (ct_service.py)          (mri_service.py)
           │                        │                        │
     X-Ray Router             CT Anatomy Router       Brain Tumor Screener
      (ResNet-50)             (EfficientNet-B0)          (5-Layer CNN)
     ┌─────┴─────┐            ┌─────┴─────┐            ┌─────┴─────┐
     ▼           ▼            ▼           ▼            ▼           ▼
   Chest       MURA         Chest       Brain        Tumor      No Tumor
 Pneumonia   Abnormal     COVID-19   Hemorrhage     Detected        │
 Ensemble    (EffNet-B3)  (EffNet-B0) (ResNeXt-101)    │            ▼
 (V2S+B3+B7)     │            │           │         Bypassed     Reflex
     │           │            │           │                    Dementia
     └─────┬─────┘            └─────┬─────┘                   (EffNetV2-S)
           ▼                        ▼                              │
     Grad-CAM Heatmap        Multi-Layer CAM                 Multi-Layer CAM
```

---

## 2. API Endpoints

### System Endpoints
- `GET /health` / `GET /api/health` — Service status, active compute device, available modalities.
- `GET /modalities` / `GET /api/modalities` — Supported modalities, tasks, and file extensions.
- `GET /models` / `GET /api/models` — Verified neural network metadata and performance benchmarks.

### Inference Endpoints
- `POST /predict/xray` (or `POST /api/xray/predict`)
  - **Form Data**: `file` (Image binary), optional `subtype` (`chest` | `musculoskeletal`)
  - **Task**: Automatic anatomical routing -> Pneumonia Ensemble or MURA Abnormality.
- `POST /predict/ct` (or `POST /api/ct/predict`)
  - **Form Data**: `file` (PNG, JPG, DICOM binary), optional `subtype` (`default` | `brain_bleed`)
  - **Task**: Automatic anatomy triage -> Thoracic COVID/Pneumonia or Cranial Hemorrhage.
- `POST /predict/mri` (or `POST /api/mri/predict`)
  - **Form Data**: `file` (Image binary), optional `subtype` (`default` | `dementia`)
  - **Task**: Two-Stage Cascade (Brain Tumor Screener -> Reflex Dementia Assessment).

---

## 3. Response Contract

All inference routes return a normalized response envelope:

```json
{
  "request_id": "c138d6b8-2a1e-4509-9fc7-789a4cd8d4f6",
  "modality": "xray",
  "routed_to": "chest",
  "prediction": "PNEUMONIA",
  "probabilities": {
    "NORMAL": 0.0438,
    "PNEUMONIA": 0.9562
  },
  "confidence": 0.9562,
  "threshold": 0.50,
  "model_name": "Chest Multi-Model Ensemble (EfficientNet-V2-S, EfficientNet-B3, EfficientNet-B7)",
  "framework": "PyTorch 2.x",
  "input_size": "300-600px",
  "dataset": "NIH ChestX-ray14 & RSNA Pneumonia",
  "metrics": {
    "Consensus Ratio": "3/3",
    "Ensemble Weighting": "V2-S (0.40) + B3 (0.35) + B7 (0.25)"
  },
  "gradcam_png_base64": "data:image/png;base64,...",
  "original_png_base64": "data:image/png;base64,...",
  "disclaimer": "Research and educational use only...",
  "warnings": [],
  "status": "success",
  "metadata": {}
}
```

---

## 4. Setup & Running Locally (Windows PowerShell)

### Prerequisites
Python 3.10+ (tested with Python 3.13) and installed PyTorch dependencies.

### Step 1: Start Backend Server
From repository root:
```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API docs available at: `http://127.0.0.1:8000/docs`.

### Step 2: Start Frontend Development Server
In a separate terminal:
```powershell
cd frontend
npm run dev
```
Access the application at `http://localhost:5173`.

---

## 5. Running Automated Tests

Run the complete test suite (unit + mocked adapter tests):
```powershell
python backend/tests/run_tests.py
```

Run real model inference smoke tests on actual repository test scans:
```powershell
python backend/tests/run_tests.py --smoke
```

---

## 6. Feature Status Matrix

| Feature | Status | Notes |
| :--- | :--- | :--- |
| **X-Ray Pneumonia & MURA** | **Integrated & Verified** | Live 3-model weighted ensemble + router |
| **CT COVID-19 & Brain Hemorrhage** | **Integrated & Verified** | Live multi-model pipeline with multi-layer Grad-CAM |
| **Brain MRI Tumor & Dementia** | **Integrated & Verified** | Live 2-stage clinical cascade architecture |
| **Side-by-Side / Overlay Grad-CAM** | **Integrated & Verified** | Interactive opacity slider & comparison |
| **3D Anatomical Visualizer** | **Integrated & Verified** | WebGL / Three.js interactive hologram shaders |
| **JSON Diagnostic Report Export** | **Integrated & Verified** | Export full verified inference payload |
| **Vision-Language Model (VLM)** | *Roadmap* | Requires multimodal cloud API key |
| **RAG-based "Ask MEDVISION"** | *Roadmap* | Requires external vector database & LLM service |
