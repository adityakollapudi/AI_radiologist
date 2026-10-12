# Model 2: Chest CT Expert (COVID-19 & Pneumonia Detection)

**Primary Task:** Multi-Class Lung Pathology Diagnosis (**Normal** vs. **Pneumonia** vs. **COVID-19**).

---

## 1. Model Overview

The Chest CT Expert analyzes axial CT scans of the thoracic cavity that have been routed by Model 1. It diagnoses normal lung parenchyma versus infectious consolidations, differentiating between non-COVID pneumonia and COVID-19 viral pneumonia patterns.

- **Checkpoint File:** [`best.pt`](file:///c:/Users/adity/Downloads/ct_anatomy_router_output/models/chest/best.pt) (54.7 MB)
- **Source Repository:** [`Covid19-CT-Scan--Detection`](file:///c:/Users/adity/Downloads/ct_anatomy_router_output/Covid19-CT-Scan--Detection)
- **Base Architecture:** `EfficientNet-B0`
- **Classification Head:**
  ```text
  Sequential(
    (0): BatchNorm1d(1280)
    (1): Dropout(p=0.30)
    (2): Linear(in_features=1280, out_features=512)
    (3): BatchNorm1d(512)
    (4): SiLU()
    (5): Dropout(p=0.20)
    (6): Linear(in_features=512, out_features=128)
    (7): BatchNorm1d(128)
    (8): SiLU()
    (9): Linear(in_features=128, out_features=3)
  )
  ```
- **Target Classes:**
  - `0`: **Normal** (Healthy lung parenchyma, clear airspaces)
  - `1`: **Pneumonia** (Bacterial or other non-COVID viral consolidation)
  - `2`: **COVID-19** (Bilateral ground-glass opacities, peripheral consolidations)

---

## 2. Dataset & Evaluation Results

Trained on the **COVIDx CT-3A** benchmark dataset comprising over 420,000 CT slices from clinical centers worldwide (CNCB, COVID-CT-MD, COVID-19-CT-Seg).

### Benchmark Performance:
- **Test Set Size:** 33,781 slices (17,922 Normal, 7,965 Pneumonia, 7,894 COVID-19)
- **Test Accuracy:** **90.92%**
- **Macro AUC-OvR:** **0.9823**
- **Macro Recall / Sensitivity:** **90.54%**
- **Macro Precision:** **89.05%**

### Per-Class Test Metrics:

| Class | Precision | Recall (Sensitivity) | F1-Score |
| :--- | :--- | :--- | :--- |
| **Normal** | 96.37% | 91.81% | 94.04% |
| **Pneumonia** | 89.39% | 92.98% | 91.15% |
| **COVID-19** | 81.38% | 86.81% | 84.01% |

---

## 3. Preprocessing Specifications

- **Input Dimensions:** $256 \times 256 \times 3$
- **Channel Format:** Grayscale replicated across RGB channels
- **Normalization:** ImageNet distribution ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)
- **Foreground Isolation:** Automatic patient body contour detection excludes the scanner couch and artifacts.

---

## 4. Standalone Python Usage

```python
from src.chest_expert import ChestExpertModel

# Load the Chest Expert
expert = ChestExpertModel()

# Run prediction on a chest CT slice
result = expert.predict("samples/test_chest.png")

print("Diagnosis:", result["diagnosis"])             # 'COVID-19', 'Pneumonia', or 'Normal'
print("Confidence:", result["confidence_pct"])       # e.g. '99.28%'
print("Probabilities:", result["probabilities"])     # {'Normal': 0.0, 'Pneumonia': 0.007, 'COVID-19': 0.993}
print("Summary:", result["clinical_summary"])
```
