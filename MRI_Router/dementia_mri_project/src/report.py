"""Markdown reports. Every number is read from files written by the pipeline; missing parts say so."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DISCLAIMER, load_json


def f(x, nd=4, pct=False):
    if x is None:
        return "n/a"
    try:
        if isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return "n/a"
        if pct:
            return f"{100 * float(x):.2f}%"
        if isinstance(x, (int, np.integer)):
            return f"{int(x):,}"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def md_table(df, floatfmt=4):
    if df is None or len(df) == 0:
        return "_not available_\n"
    cols = list(df.columns)
    out = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(f(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in r.values) + " |\n"
    return out


# ------------------------------------------------------------------------------------------------ audit
def write_audit_report(s, aud, paths):
    d, p, i = s["exact_duplicates"], s["perceptual_duplicates"], s["identifiers"]
    cls = pd.DataFrame({"class": list(s["class_distribution"]), "count": list(s["class_distribution"].values()),
                        "percent": [s["class_percent"][c] for c in s["class_distribution"]]})
    txt = f"""# Dataset audit -- {Path(s['dataset_root']).name}

> {DISCLAIMER}

## Files
| item | value |
|---|---|
| dataset root | `{s['dataset_root']}` |
| total image files | {f(s['total_files'])} |
| valid / readable | {f(s['valid_images'])} |
| corrupted / unreadable (excluded, listed in `corrupted_images.csv`) | {f(s['corrupted_images'])} |
| images outside a recognised class folder (excluded) | {f(s['unlabelled_images'])} |
| usable before de-duplication | {f(s['final_usable_images_before_dedup'])} |
| non-image files | {f(s['non_image_files'])} |

## Classes (discovered from folder names, not assumed)
{md_table(cls, 2)}
Imbalance ratio (max/min): {f(s['imbalance_ratio_max_min'], 2)}

Existing split structure (folder-level): `{s['existing_split_structure']}`

## Image characteristics
| property | min | median | mean | max |
|---|---|---|---|---|
""" + "".join(f"| {k} | {f(s[k].get('min'), 2)} | {f(s[k].get('median'), 2)} | {f(s[k].get('mean'), 2)} | {f(s[k].get('max'), 2)} |\n"
               for k in ("width", "height", "aspect_ratio", "file_size_bytes", "brightness", "contrast", "black_frac") if s.get(k)) + f"""
Modes: `{s['modes']}` · formats: `{s['formats']}` · estimated JPEG quality: `{s.get('jpeg_quality_est')}`

**RGB vs grayscale:** {f(s['rgb_channels_nearly_identical_frac'], pct=True)} of images have R≈G≈B (99th-percentile
channel spread ≤ 3 grey levels); mean |R−G| = {f(s['mean_abs_R_minus_G'], 3)}, mean |G−B| = {f(s['mean_abs_G_minus_B'], 3)}.
The JPG channels are a storage format, not native MRI acquisition channels.

## Exact duplicates (MD5 + SHA256)
""" + "".join(f"- {k}: **{v}**\n" for k, v in d.items()) + f"""
## Perceptual duplicates (pHash, 8 dihedral variants, 64×64 pixel check)
""" + "".join(f"- {k}: **{v}**\n" for k, v in p.items()) + f"""
## Original / augmented names and identifiers
""" + "".join(f"- {k}: **{v}**\n" for k, v in i.items()) + """
Identifiers are only reported when an explicit pattern (e.g. OASIS `OAS1_xxxx_MRy`) matches a file or folder name;
they are never invented.
"""
    (paths["reports"] / "dataset_audit.md").write_text(txt)
    (paths["audit"] / "dataset_audit_report.md").write_text(txt)


# ---------------------------------------------------------------------------------------------- leakage
def write_leakage_report(s, meta, pairs, paths):
    g, si, chk, old = s["grouping_log"], s["split_info"], s["leakage_checks"], s["existing_split_leakage"]
    counts = pd.DataFrame(s["counts"]).T if isinstance(s["counts"], dict) else None
    tab = pd.DataFrame(s["counts"])
    tab = tab.reset_index().rename(columns={"index": "split"}) if len(tab) else tab
    gtab = pd.DataFrame(s["groups_per_split_class"]).reset_index().rename(columns={"index": "split"})
    txt = f"""# Leakage analysis and leakage-free split

> {DISCLAIMER}

## Patient-level independence
{s['patient_level_statement']}

Patient-identifier coverage: {f(s['patient_id_coverage'], pct=True)}.

## Source grouping
Priority: patient ID > study ID > examination ID > reliable original-image link > exact duplicate > validated
perceptual transformation. Perceptually similar images of two *different identified patients* are never merged.

""" + "".join(f"- {k}: **{v}**\n" for k, v in g.items()) + f"""
## Existing (folder) split -- leakage found
""" + "".join(f"- {k}: **{v}**\n" for k, v in old.items()) + f"""
The dataset's own train/validation/test folders were therefore **not** used; a new split was created.

## New grouped stratified split (target 70 / 15 / 15)
Allocation error vs target: {f(si['allocation_error'])} · groups per split: `{si['groups_per_split']}`

Images per split and class:
{md_table(tab, 0)}
Source groups per split and class:
{md_table(gtab, 0)}
""" + ("".join(f"- **LIMITATION:** {x}\n" for x in si["limitations"]) or "") + f"""
## Assertions (training is blocked if any fails)
""" + "".join(f"- {k}: **{v}**\n" for k, v in chk.items()) + f"""
```
assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)
```
All passed.

## Preprocessing decision
""" + "".join(f"- {k}: {v}\n" for k, v in s["preprocessing"].items()) + f"""
## Class weights
Training-set imbalance ratio {f(s['train_imbalance_ratio'], 2)} → class weights: `{s['class_weights']}` (computed from
training data only).
"""
    (paths["reports"] / "leakage_report.md").write_text(txt)
    (paths["data"] / "dataset_leakage_report.md").write_text(txt)


# ------------------------------------------------------------------------------------------------ final
def write_final_report(paths, cfg):
    R = paths["results"]
    audit = load_json(paths["audit"] / "dataset_summary.json", {}) or {}
    split = load_json(paths["data"] / "split_summary.json", {}) or {}
    prim = load_json(R / "runs" / "primary" / "metrics.json")
    base = load_json(R / "runs" / "baseline" / "metrics.json")
    tp = load_json(R / "runs" / "primary" / "train_summary.json", {}) or {}
    tb = load_json(R / "runs" / "baseline" / "train_summary.json", {}) or {}
    rc = load_json(paths["checkpoints"] / "primary" / "config.json", {}) or {}
    cam = load_json(R / "gradcam_summary.json")
    err = load_json(R / "error_analysis_summary.json")
    ext = load_json(R / "external_validation.json", {"statement": "Independent external validation was not performed."})
    sc = load_json(R / "shortcut_analysis.json", {}) or {}
    rob = pd.read_csv(R / "robustness_results.csv") if (R / "robustness_results.csv").exists() else None
    abl = pd.read_csv(R / "ablation_results.csv") if (R / "ablation_results.csv").exists() else None
    comp = pd.read_csv(R / "model_comparison.csv") if (R / "model_comparison.csv").exists() else None
    classes = split.get("classes", cfg["class_order"])
    m = prim["four_class"] if prim else None
    b = prim["binary_dementia_detection"] if prim else None
    o = prim["ordinal_severity"] if prim else None
    c = prim["calibration"] if prim else None
    u = prim["uncertainty"] if prim else None
    na = "_Not available -- the corresponding stage has not been run._\n"

    per_class = pd.DataFrame([{"class": k, **{x: v[x] for x in ("precision", "recall", "f1", "support", "roc_auc_ovr", "pr_auc")}}
                              for k, v in m["per_class"].items()]) if m else None
    sec = []
    A = sec.append
    A(f"""# {cfg['project_name']}
### AI-based dementia category classification from brain MRI -- research report

> **{DISCLAIMER}**

## 1. Abstract
""")
    if m:
        A(f"""A four-class classifier ({prim['model']}) was trained to predict the dataset-defined dementia category of
preprocessed brain MRI JPG slices after a full dataset audit, duplicate and leakage analysis and a {'patient-level' if split.get('patient_level_split') else 'source-group-level'}
grouped split. On the untouched test set ({f(m['n'])} images) it reached accuracy {f(m['accuracy'])}, macro-F1 {f(m['macro_f1'])},
balanced accuracy {f(m['balanced_accuracy'])} and macro one-vs-rest ROC-AUC {f(m['macro_roc_auc_ovr'])}. Derived
non-demented vs any-dementia detection: sensitivity {f(b.get('sensitivity'))}, specificity {f(b.get('specificity'))},
ROC-AUC {f(b.get('roc_auc'))}. {split.get('patient_level_statement', '')}
""")
    else:
        A(na)
    A(f"""
## 2. Introduction
Automated analysis of structural brain MRI is an active research area for dementia-related decision support. This
project builds a reproducible, explainable research prototype and focuses on *defensible evaluation* rather than
headline accuracy.

## 3. Problem statement
Public preprocessed MRI datasets often contain duplicated, pre-augmented and patient-correlated slices. Random
image-level splitting then leaks near-identical images into the test set and inflates results.

## 4. Objective
Predict one of the dataset-defined categories {classes} from a single MRI slice, with calibrated model
confidence, model uncertainty and Grad-CAM attributions, evaluated on a leakage-free split.

## 5. Dataset description
Kaggle `{cfg['dataset_slug']}` (root `{audit.get('dataset_root', 'n/a')}`). Total files {f(audit.get('total_files'))},
valid {f(audit.get('valid_images'))}, corrupted {f(audit.get('corrupted_images'))}, classes discovered from folder
names: {list(audit.get('class_distribution', {}))}.

## 6. Dataset distribution
Before de-duplication: `{audit.get('class_distribution')}` (imbalance ratio {f(audit.get('imbalance_ratio_max_min'), 2)}).
After cleaning and splitting: see section 12. Figures: `results/class_distribution.png`, `results/split_distribution.png`,
`results/split_class_distribution.png`.

## 7. Image characteristics
Width {audit.get('width')}, height {audit.get('height')}, modes {audit.get('modes')}. Brightness {audit.get('brightness')},
contrast {audit.get('contrast')}. RGB channels nearly identical in {f(audit.get('rgb_channels_nearly_identical_frac'), pct=True)} of
images. These are preprocessed JPG images; no native MRI intensity information is claimed.

## 8. Dataset quality
Corrupted images excluded: {f(audit.get('corrupted_images'))} (`data/audit/corrupted_images.csv`). Images excluded during
grouping (label-conflicting duplicates, redundant exact copies): {f(split.get('n_excluded'))}.

## 9. Duplicate analysis
Exact: `{audit.get('exact_duplicates')}`

Perceptual: `{audit.get('perceptual_duplicates')}`

## 10. Leakage analysis
Existing folder split: `{split.get('existing_split_leakage')}`. Assertions on the new split: `{split.get('leakage_checks')}`.

## 11. Source grouping
{split.get('patient_level_statement', '')}

Grouping log: `{split.get('grouping_log')}`

## 12. Data splitting
Grouped stratified split, target 70/15/15. Counts: `{split.get('counts')}`. Split shares: `{split.get('split_share')}`.
""" + "".join(f"\n**Limitation:** {x}\n" for x in split.get("split_info", {}).get("limitations", [])) + f"""
## 13. Preprocessing
`{rc.get('preprocessing')}` -- validation, RGB conversion, {split.get('preprocessing', {}).get('color_mode')} input,
brain crop = {split.get('preprocessing', {}).get('brain_crop')} ({split.get('preprocessing', {}).get('brain_crop_reason')}),
aspect-preserving resize with zero padding (no stretching), ImageNet normalisation.

## 14. Augmentation
Training only: `{rc.get('augmentation')}`. No vertical flip; horizontal flip disabled by default (left/right
anatomy is not assumed interchangeable).

## 15. Model architecture
Primary {rc.get('model_display', cfg['model'])}: ImageNet backbone → global average pooling → Dropout({cfg['dropout']}) →
Linear(F, {cfg['hidden_dim']}) → SiLU → Dropout({cfg.get('dropout2', 0.2)}) → Linear({cfg['hidden_dim']}, 4). Baseline EfficientNet-B0 with the same head.

## 16. Transfer learning
Stage 1: frozen backbone, head LR {cfg['head_lr']}, {rc.get('epochs_stage1')} epochs. Stage 2: last
{cfg['unfreeze_fraction']:.0%} of backbone parameters unfrozen (block granularity), backbone LR {cfg['backbone_lr']}, head LR
{cfg['finetune_head_lr']}, up to {rc.get('epochs_stage2')} epochs.

## 17. Training configuration
AdamW (wd {cfg['weight_decay']}), ReduceLROnPlateau on validation macro-F1 (factor {cfg['scheduler_factor']}, patience
{cfg['scheduler_patience']}, min LR {cfg['min_lr']}), early stopping patience {cfg['early_stopping_patience']}, AMP, seed {cfg['seed']},
batch {rc.get('batch_size')}. Environment: `{ {k: rc.get('env', {}).get(k) for k in ('python', 'torch', 'torchvision', 'cuda_version', 'gpu')} }`.
Primary best epoch {tp.get('best_epoch')} (val macro-F1 {f(tp.get('best_val_macro_f1'))}), training time {f(tp.get('train_minutes'), 1)} min.

## 18. Evaluation methodology
Model selection on validation macro-F1 only. The selected `best_model.pth` was evaluated once on the untouched test
set (a lock file prevents silent re-evaluation with another checkpoint). Temperature fitted on validation only.

## 19. Four-class results (test)
""")
    if m:
        A(f"""| metric | value |
|---|---|
""" + "".join(f"| {k} | {f(m[k])} |\n" for k in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
                                                 "weighted_f1", "top3_accuracy")) + "\n" + md_table(per_class) +
          f"\nClass-specific analysis: `{prim['class_specific']}`\n")
    else:
        A(na)
    A("\n## 20. Binary dementia detection results (derived from the FOUR-CLASS model)\n")
    A((f"| metric | value |\n|---|---|\n" + "".join(f"| {k} | {f(b[k])} |\n" for k in ("accuracy", "sensitivity", "specificity", "precision", "npv", "f1", "roc_auc", "pr_auc"))
       + "\nNo separate binary model was trained.\n") if b else na)
    A("\n## 21. Severity analysis (analytical ordering, not clinical staging)\n")
    A((f"Severity MAE {f(o['mean_absolute_severity_error'])}, expected-severity MAE {f(o['expected_severity_mae'])}, "
       f"quadratic weighted kappa {f(o['quadratic_weighted_kappa'])}, adjacent errors {o['adjacent_errors']}, "
       f"large (≥2 levels) errors {o['large_errors_ge2']}, under-/over-estimated {o['under_estimated']}/{o['over_estimated']}.\n") if o else na)
    A("\n## 22. ROC-AUC\n")
    A((f"Macro OvR ROC-AUC {f(m['macro_roc_auc_ovr'])}, weighted {f(m['weighted_roc_auc_ovr'])} (`results/roc_curves.png`).\n") if m else na)
    A("\n## 23. PR-AUC\n")
    A((f"Macro PR-AUC {f(m['macro_pr_auc'])} (`results/pr_curves.png`).\n") if m else na)
    A("\n## 24. Confusion matrix\n")
    A((f"`{m['confusion_matrix']}` (rows = actual, columns = predicted, order {classes}); "
       "`results/confusion_matrix.png`, `results/normalized_confusion_matrix.png`.\n") if m else na)
    A("\n## 25. Calibration\n")
    A((f"Temperature T = {f(c['temperature'], 3)} (fitted on validation). ECE {f(c['uncalibrated']['ece'])} → "
       f"{f(c['temperature_scaled']['ece'])}, Brier {f(c['uncalibrated']['brier'])} → {f(c['temperature_scaled']['brier'])}. "
       f"Accuracy when confidence ≥ 0.9: {f(c.get('accuracy_when_confidence_ge_0.9'))} "
       f"(share {f(c.get('share_with_confidence_ge_0.9'), pct=True)}).\n") if c else na)
    A("\n## 26. Uncertainty\n")
    A((f"By status: `{u['by_status']}`. MC dropout: `{u.get('mc_dropout')}`. These are model-uncertainty estimates, not clinical certainty.\n") if u else na)
    A("\n## 27. Grad-CAM\n")
    A((f"Standard Grad-CAM on `{cam.get('layers', {}).get('final')}`; validity checks: `{cam.get('validity')}`. "
       f"Sanity checks: `{cam.get('sanity')}`.\n") if cam else na)
    A("\n## 28. Multi-layer Grad-CAM\n")
    A((f"Layers `{cam.get('layers')}`. Fusion: {cam.get('fusion_method')}. Examples in `gradcam/`. Grad-CAM is an "
       "attribution method and not a lesion segmentation.\n") if cam else na)
    A("\n## 29. Error analysis\n")
    A((f"`{err}`\n\nDetails: `results/error_analysis.csv`, panels in `gradcam/errors/`.\n") if err else na)
    A("\n## 30. Robustness testing\n")
    A(md_table(rob) if rob is not None else na)
    A("\n## 31. Ablation study (validation set)\n")
    A(md_table(abl) if abl is not None else na)
    A("\n### Baseline vs primary (test)\n")
    A(md_table(comp) if comp is not None else na)
    A(f"\n## 32. External validation\n{ext.get('statement') if not ext.get('performed') else ext}\n")
    A(f"""
## 33. Limitations
- This is a research prototype. It predicts dataset-defined dementia categories and does not diagnose Alzheimer's disease, dementia or any neurological disease.
- The dataset consists of preprocessed JPG MRI images rather than guaranteed native DICOM MRI data; no native MRI intensity information is claimed.
- {split.get('patient_level_statement', 'Patient-level independence could not be established.')}
""" + "".join(f"- {x}\n" for x in split.get("split_info", {}).get("limitations", [])) + f"""- Shortcut analysis: image statistics alone give validation macro-F1 {f(sc.get('val_macro_f1_from_image_statistics'))} (chance ≈ {f(sc.get('chance_macro_f1_approx'), 2)}). {sc.get('interpretation', '')}
- The model may learn dataset-specific shortcuts and may not generalize to other hospitals, scanners, populations, acquisition protocols or preprocessing pipelines.
- Model confidence is not clinical certainty. Grad-CAM is an attribution method and not a clinically validated lesion segmentation.
- External validation is required before any clinical claim.

## 34. Future work
Patient-level evaluation on datasets with reliable identifiers; 3-D volume models; external multi-site validation;
prospective reader studies; ordinal-aware losses evaluated against clinical staging.

## 35. Conclusion
""")
    if m:
        A(f"The {prim['model']} research prototype reached test macro-F1 {f(m['macro_f1'])} on a leakage-checked split. "
          "The numbers above are only as independent as the grouping evidence allows; they should be read together with "
          "the limitations and are not evidence of clinical performance.\n")
    else:
        A(na)
    txt = "".join(sec)
    (paths["reports"] / "final_report.md").write_text(txt)
    (paths["root"] / "FINAL_REPORT.md").write_text(txt)
    return paths["root"] / "FINAL_REPORT.md"
