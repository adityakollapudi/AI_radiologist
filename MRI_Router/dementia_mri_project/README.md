# Dementia MRI Detection and Severity Classification

This is a research prototype. It predicts **dataset-defined** dementia categories (Non Demented, Very mild Dementia,
Mild Dementia, Moderate Dementia) from preprocessed JPG brain MRI slices. It is **not** a clinically validated
diagnostic system: model confidence is not clinical certainty, and Grad-CAM is not a lesion segmentation.

The workflow is enforced in code:

audit → leakage analysis → source groups → leakage-free split → preprocessing → baseline → primary → validation
model selection → single test evaluation → calibration → uncertainty → Grad-CAM → multi-layer Grad-CAM → error
analysis → robustness → ablation → final report.

Two stages refuse to run when their prerequisites are missing:
- **Training** refuses to start without a passed leakage check.
- **The test set** is evaluated only once per checkpoint. A lock file blocks evaluating it again with a different checkpoint.

## Run on Kaggle (T4)
1. Zip this folder (`dementia_mri_project.zip`) and upload it as a Kaggle **Dataset**. Kaggle extracts the zip.
2. Create a notebook and set the accelerator to **GPU T4** with **Internet ON**. Add two inputs:
   - the code dataset you uploaded in step 1
   - `prasukjain07/preprocessed-alzheimers-disease-mri-dataset` (if it isn't attached, the notebook downloads it with `kagglehub`)
3. Open `notebooks/00_kaggle_full_pipeline.ipynb` and run all cells. Kaggle notebooks don't share `/kaggle/working`,
   so the all-in-one notebook is the simplest route. Notebooks 01–05 run the same stages with more inspection cells.
   To use them on Kaggle, save each notebook's output as a dataset and attach it to the next notebook.
4. Outputs are written to `/kaggle/working/dementia_mri_project/`.

To change settings, edit `CONFIG_OVERRIDES` in the bootstrap cell, for example:
`{"batch_size": 16}` or `{"ablation": {"enabled": False}}`.

## Run locally (e.g. RTX 4060 laptop)
```
pip install -r requirements.txt          # install a CUDA build of torch from pytorch.org first
python -m src.run_pipeline --stage audit split
python -m src.run_pipeline --stage train evaluate gradcam report package
python predict.py --image /path/to/mri.jpg
```
- Point to a local dataset with `--set data_root=/path/to/dataset`.
- If you have 8 GB of GPU memory and run out of it, add `--set batch_size=16`. CUDA out-of-memory errors also halve the batch size automatically.
- On Windows, DataLoader workers work because every transform is defined at module level.

## Stages and outputs
| stage | what it does | main outputs |
|---|---|---|
| audit | Finds the dataset and discovers classes from folder names. Verifies every image. Computes MD5/SHA256, pHash with 8 dihedral variants plus a pixel check, original/augmented filename parsing and patient/study/exam ID parsing. | `data/class_mapping.json`, `data/audit/{dataset_audit.csv, dataset_summary.json, corrupted_images.csv, duplicate_groups.csv, near_duplicate_report.csv, dataset_audit_report.md}`, `reports/dataset_audit.md` |
| split | Builds `source_group_id` with union-find. Creates a grouped stratified 70/15/15 split and runs the leakage assertions. Also covers RGB-vs-grayscale, the brain-crop decision and shortcut analysis. | `data/{metadata,train,val,test}.csv`, `data/split_summary.json`, `reports/leakage_report.md`, `results/{class,split,split_class}_distribution.png`, `results/shortcut_analysis.json` |
| train | Trains EfficientNet-B0 (baseline), then EfficientNetV2-S (primary). Two stages, AMP, AdamW, ReduceLROnPlateau and early stopping on validation macro-F1. | `checkpoints/{best_model.pth, last_checkpoint.pth, config.json, class_mapping.json, training_history.csv}`, `results/training_*.png` |
| evaluate | Evaluates the test set once and fits the temperature on validation. Produces four-class, derived binary and ordinal metrics, ECE/Brier, entropy and MC-dropout uncertainty, robustness, the ablation (validation only) and external validation. | `results/{metrics.json, classification_report.csv, confusion_matrix.png, normalized_confusion_matrix.png, roc_curves.png, pr_curves.png, calibration.png, robustness_results.csv, ablation_results.csv, model_comparison.csv}` |
| gradcam | Runs standard Grad-CAM first (with checks), then multi-layer Grad-CAM with fusion, sanity checks (randomisation and deletion) and error analysis with Grad-CAM. | `gradcam/`, `results/{gradcam_summary.json, error_analysis.csv}` |
| report / package | Writes the final report and builds the model package. | `FINAL_REPORT.md`, `reports/final_report.md`, `results/model_package.zip` |

## Key design decisions
- **Patient IDs.** Patient, study and exam IDs are used only when an explicit pattern matches, for example OASIS
  `OAS1_0028_MR1`. They are never invented. Without them, the report states that patient-level independence could
  not be established.
- **Perceptual duplicates.** These are merged only when the pHash distance is 4 or less (over 8 flips/rotations),
  the 64×64 pixel correlation is 0.90 or more, and both images don't belong to different identified patients.
  Edges that would chain into very large components are dropped and reported.
- **Too few patients in a class.** A class with fewer than 3 independent groups cannot be in all three splits. The
  pipeline puts it in train and test and records this as a limitation instead of hiding it.
- **Grayscale vs RGB.** Grayscale replicated to 3 channels is chosen automatically when R≈G≈B. RGB is compared in
  the ablation.
- **Horizontal flip.** This is off by default, because swapping left and right anatomy is not assumed to be harmless.
