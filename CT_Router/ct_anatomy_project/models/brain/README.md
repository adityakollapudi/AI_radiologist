# Model 3: Brain CT Expert (RSNA Intracranial Hemorrhage Detection)

**Primary Task:** Multi-Label Detection of **Acute Intracranial Hemorrhage (ICH)** and its 5 anatomical subtypes.

---

## 1. Model Overview

The Brain CT Expert evaluates axial non-contrast head CT slices routed by Model 1 to detect acute hemorrhages. It implements the winning architecture from the **RSNA 2019 Intracranial Hemorrhage Detection Challenge** (1st Place Solution).

- **Checkpoint File:** [`model_epoch_best_4.pth`](file:///c:/Users/adity/Downloads/ct_anatomy_router_output/models/brain/model_epoch_best_4.pth) (188.4 MB)
- **Source Repository:** [`RSNA2019_Intracranial-Hemorrhage-Detection`](file:///c:/Users/adity/Downloads/ct_anatomy_router_output/RSNA2019_Intracranial-Hemorrhage-Detection)
- **Base Architecture:** `se_resnext101_32x4d` (Squeeze-and-Excitation ResNeXt-101)
- **Target Subtypes (Multi-Label Output):**
  1. **`any`** — Overall presence of any acute intracranial hemorrhage
  2. **`epidural`** — Epidural hematoma (EDH, between dura mater and skull)
  3. **`intraparenchymal`** — Intraparenchymal hemorrhage (IPH, inside brain tissue)
  4. **`intraventricular`** — Intraventricular hemorrhage (IVH, inside ventricular cavities)
  5. **`subarachnoid`** — Subarachnoid hemorrhage (SAH, inside subarachnoid space)
  6. **`subdural`** — Subdural hematoma (SDH, between dura and arachnoid membranes)

---

## 2. Dataset & Clinical Context

Trained on the **RSNA 2019 Brain CT Challenge Dataset** (>25,000 non-contrast head CT studies, 870,000+ annotated slices) contributed by multiple international academic medical centers.

### Clinical Diagnostic Logic:
- A scan is deemed positive for acute hemorrhage when the **`any`** probability exceeds $0.50$ (or a calibrated risk threshold).
- Individual subtype probabilities identify the specific anatomical compartment involved.
- In normal scans, all 6 outputs remain near $0.00$.

---

## 3. Preprocessing Specifications

- **Input Dimensions:** $256 \times 256 \times 3$
- **Cropping:** 80% center crop to remove exterior scanner artifacts and air background.
- **HU Windowing:** Compatible with standard Brain ($W: 80, C: 40$) and Subdural ($W: 130-300, C: 50-100$) Hounsfield windows.
- **Normalization:** $\mu = [0.456, 0.456, 0.456]$, $\sigma = [0.224, 0.224, 0.224]$.

---

## 4. Standalone Python Usage

```python
from src.brain_expert import BrainExpertModel

# Load the Brain Expert
expert = BrainExpertModel()

# Run prediction on a brain CT slice
result = expert.predict("samples/test_brain.png")

print("Diagnosis:", result["diagnosis"])
print("Hemorrhage Detected:", result["hemorrhage_detected"])  # True / False
print("Confidence:", result["confidence_pct"])
print("Subtypes Breakdown:")
for subtype, prob in result["probabilities"].items():
    print(f"  - {subtype}: {prob * 100:.2f}%")
print("Summary:", result["clinical_summary"])
```
