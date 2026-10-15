# MEDVISION AI — Unified Multimodal AI Radiologist System

[![Author](https://img.shields.io/badge/Author-Aditya%20Kollapudi-blue.svg)](mailto:adityakollapudi@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-adityakollapudi-181717.svg)](https://github.com/adityakollapudi)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.x-61dafb.svg)](https://react.dev/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL-black.svg)](https://threejs.org/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-orange.svg)](https://git-lfs.github.com/)

A comprehensive, multimodal AI Diagnostic Platform integrating deep learning models for **Chest & Musculoskeletal X-Ray**, **Thoracic & Cranial Computed Tomography (CT)**, and **Brain Magnetic Resonance Imaging (MRI)** into a single unified clinical architecture with interactive **3D Anatomical Visualizations**, **Multi-Layer Grad-CAM Explainability**, and automated **PDF Diagnostic Reporting**.

---

## 👨‍💻 Author

- **Name**: Aditya Kollapudi
- **Email**: [adityakollapudi@gmail.com](mailto:adityakollapudi@gmail.com)
- **GitHub**: [@adityakollapudi](https://github.com/adityakollapudi)

---

## 🏛️ System Architecture

```
                                  FastAPI Unified Backend (Port 8000)
                                                    │
                 ┌──────────────────────────────────┼──────────────────────────────────┐
                 ▼                                  ▼                                  ▼
           X-Ray Service                      CT Service                         MRI Service
         (xray_service.py)                 (ct_service.py)                    (mri_service.py)
                 │                                  │                                  │
           X-Ray Router                     CT Anatomy Router                Brain Tumor Screener
         (EfficientNet-B0)                  (EfficientNet-B0)                    (5-Layer CNN)
          ┌──────┴──────┐                    ┌──────┴──────┐                    ┌──────┴──────┐
          ▼             ▼                    ▼             ▼                    ▼             ▼
        CHEST         MURA                 CHEST         BRAIN                TUMOR        NO TUMOR
      Pneumonia     Abnormal             COVID-19     Hemorrhage             Detected         │
      Ensemble    (EffNet-B3)           (EffNet-B0)  (ResNeXt-101)              │             ▼
     (V2S+B3+B7)        │                    │             │                 Bypassed      Reflex
          │             │                    │             │                              Dementia
          └──────┬──────┘                    └──────┬──────┘                            (EffNetV2-S)
                 ▼                                  ▼                                         │
          Grad-CAM Heatmap                    Multi-Layer CAM                           Multi-Layer CAM
```

---

## 🚀 Key Features

1. **Multimodal Diagnostic Routing**:
   - **X-Ray**: Automatic anatomical routing (Chest vs Musculoskeletal) leading to a live 3-model weighted ensemble (EfficientNet-V2-S, B3, B7) for pneumonia or MURA abnormality detection.
   - **Computed Tomography (CT)**: Automated anatomical triage routing thoracic slices to COVID-19/pneumonia classification or cranial slices to intracranial hemorrhage detection.
   - **Brain MRI Cascade**: Two-stage clinical cascade: Stage 1 Brain Tumor screening; if normal, reflex Stage 2 Dementia assessment (OASIS/ADNI multi-class staging) is triggered.

2. **Explainable AI (XAI)**:
   - High-resolution, multi-layer gradient-weighted class activation mapping (Grad-CAM) heatmaps with interactive opacity slider and side-by-side comparison.

3. **Photorealistic 3D Anatomical Hologram Viewer**:
   - WebGL / Three.js interactive hologram shaders depicting anatomical structures in real time.

4. **Automated Diagnostic Reporting**:
   - Downloadable, publication-grade PDF diagnostic report generation via ReportLab with embedded scans and heatmaps.

---

## 🛠️ Installation & Setup

### Prerequisites

- **Python**: 3.10+ (tested with Python 3.13 / 3.14)
- **Node.js**: v18+ (tested with Node.js v24)
- **Git & Git LFS**: Ensure `git lfs install` is run to track large neural network weights (`*.pth`, `*.pt`, `*.h5`).

### 1. Backend Setup

```bash
# Clone the repository
git clone https://github.com/adityakollapudi/AI_Radiologist-integration-medvision-backend.git
cd AI_Radiologist-integration-medvision-backend

# Install backend dependencies
pip install -r backend/requirements.txt
pip install -r requirements.txt pypdf

# Launch the FastAPI backend server
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation will be available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 2. Frontend Setup

In a new terminal window:

```bash
cd frontend

# Install frontend dependencies
npm install

# Start Vite development server
npm run dev
```
Open your browser at: [http://localhost:5173](http://localhost:5173).

---

## 🧪 Testing

Run the automated test suite:

```bash
python backend/tests/run_tests.py
```

Run real model inference smoke tests on actual test scans:

```bash
python backend/tests/run_tests.py --smoke
```

---

## 📜 Disclaimer

*This application is developed strictly for research, educational, and benchmarking purposes. It does not provide a formal medical diagnosis and must not replace evaluation by a licensed healthcare professional.*
