# Model 1: CT Anatomy Router

**Primary Task:** Binary Classification & Triage Routing between **CHEST_CT** and **BRAIN_CT** with Uncertainty Detection.

---

## 1. Model Overview

The CT Anatomy Router serves as the frontline gatekeeper in the multi-model clinical pipeline. It inspects an incoming axial CT slice and routes it with high confidence to the appropriate specialized downstream neural network.

- **Checkpoint File:** [`best_router.pt`](file:///c:/Users/adity/Downloads/ct_anatomy_router_output/models/router/best_router.pt) (43.5 MB)
- **Base Architecture:** `torchvision.models.efficientnet_b0` (Pretrained on ImageNet-1K)
- **Classification Head:** 
  ```text
  Sequential(
    (0): Dropout(p=0.3)
    (1): Linear(in_features=1280, out_features=256)
    (2): SiLU()
    (3): Dropout(p=0.2)
    (4): Linear(in_features=256, out_features=2)
  )
  ```
- **Total Parameters:** 4,378,063 (~4.38M parameters)
- **Target Classes:** 
  - Index `0`: `CHEST_CT`
  - Index `1`: `BRAIN_CT`

---

## 2. Dataset & Training Protocol

The router was trained on two independent medical imaging datasets with strict patient-level separation:

- **Chest CT Source:** COVIDx-CT Version 3A (Official patient-level split across 4,232 training patients).
- **Brain CT Source:** Qure.ai CQ500 (`qureai-headct`, axial non-contrast plain head CT series across 330 training patients).
- **Leakage Prevention Audit:**
  - 0 patient overlap across train, validation, and test splits.
  - Slices capped at $\le 50$ slices per study/patient to avoid single-patient bias.
- **Dataset Partitioning:**
  - **Train Cohort:** 23,444 slices (11,722 Chest CT / 11,722 Brain CT)
  - **Validation Cohort:** 4,966 slices (2,483 Chest CT / 2,483 Brain CT)
  - **Test Cohort:** 4,850 slices (2,425 Chest CT / 2,425 Brain CT)

---

## 3. Decision Boundary & Uncertainty Policy

- **Calibrated Operating Threshold:** $\mathbf{T = 0.50}$ (Optimal separation via Youden's $J$ statistic sweep on validation set).
- **Uncertainty Band:** $\mathbf{0.45}$
  $$\text{Flag as UNCERTAIN if } |P(\text{BRAIN}) - 0.50| < 0.45 \quad (0.05 < P(\text{BRAIN}) < 0.95)$$
  Borderline or corrupted slices falling inside this band are flagged as `UNCERTAIN (Review Needed)` to prevent automated misrouting.

---

## 4. Performance Metrics

Evaluated on 4,850 independent test slices ([`router_metrics.json`](file:///c:/Users/adity/Downloads/ct_anatomy_router_output/models/router/router_metrics.json)):

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Test Accuracy** | **100.0%** | 4,850 / 4,850 correct |
| **ROC-AUC** | **1.0000** | Perfect separation curve |
| **PR-AUC** | **1.0000** | Zero false positives/negatives |
| **Uncertainty Rate** | **0.06%** | Only 3 of 4,850 slices flagged for review |
| **Routed Accuracy** | **100.0%** | 100% precision on confident classifications |

### Confounder Robustness Checks:
- **Brain Window Swap Test:** 100.0% accuracy when brain scans were forced into lung Hounsfield windows.
- **Head Crop Invariance:** 100.0% accuracy on uncropped brain CTs.
- **Bounding Box Invariance:** 99.6% accuracy on chest CTs without body bounding boxes.

---

## 5. Standalone Python Usage

```python
import torch
from src.router import CTAnatomyRouter

# Load the router
router = CTAnatomyRouter()

# Predict an image
result = router.predict("path/to/scan.png")
print("Anatomy:", result["prediction"])        # 'CHEST_CT' or 'BRAIN_CT'
print("Confidence:", result["confidence_pct"])  # e.g. '100.00%'
print("Status:", result["status"])              # 'ROUTED' or 'UNCERTAIN'
```
