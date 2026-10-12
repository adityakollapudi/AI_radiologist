"""Final evaluation on the untouched TEST set (once per selected checkpoint), calibration, uncertainty,
binary / ordinal analyses, class-specific analysis, error analysis, model comparison and external validation."""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .calibration import calibration_report, fit_temperature, plot_reliability, softmax_np
from .dataset import load_split_frames, make_loader
from .metrics import (binary_dementia_metrics, class_specific_analysis, multiclass_metrics, ordinal_metrics,
                      plot_binary_curves, plot_confusion, plot_roc_pr)
from .model import DementiaNet, count_params, model_size_mb
from .uncertainty import (mc_dropout_summary, normalized_entropy, plot_uncertainty, uncertainty_status, uncertainty_table)
from .utils import amp_settings, get_device, load_json, print_header, save_json, sha256_file


def load_model(ckpt_path, device=None):
    device = device or get_device()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    h = ck.get("head", {})
    m = DementiaNet(ck["arch"], ck["num_classes"], pretrained=False, dropout=h.get("dropout", 0.3),
                    dropout2=h.get("dropout2", 0.2), hidden=h.get("hidden", 256))
    m.load_state_dict(ck["model_state"])
    return m.to(device).eval(), ck


@torch.no_grad()
def predict_logits(model, loader, device, amp=(False, None, False), mc_passes=0):
    model.eval()
    logits, ys, idxs, mc = [], [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp[1], enabled=amp[0]):
            out = model(x).float()
        logits.append(out.cpu()); ys.append(y); idxs.append(idx)
        if mc_passes:
            from .uncertainty import enable_mc_dropout
            enable_mc_dropout(model)
            with torch.autocast(device_type=device.type, dtype=amp[1], enabled=amp[0]):
                mc.append(torch.stack([model(x).float().cpu() for _ in range(mc_passes)]))
            model.eval()
    res = {"logits": torch.cat(logits).numpy(), "y": torch.cat(ys).numpy(), "idx": torch.cat(idxs).numpy()}
    if mc_passes:
        res["mc_logits"] = torch.cat(mc, 1).numpy()
    return res


def benchmark(model, image_size, device, n=30, batch=32):
    model.eval()
    x1 = torch.randn(1, 3, image_size, image_size, device=device)
    xb = torch.randn(batch, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model(x1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat = (time.time() - t0) / n * 1000
        t0 = time.time()
        for _ in range(max(3, n // 5)):
            model(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        thr = batch * max(3, n // 5) / (time.time() - t0)
    return {"latency_ms_batch1": lat, "images_per_sec_batch": thr, "benchmark_batch": batch, "device": str(device)}


def evaluate_run(cfg, paths, run_name, publish=False):
    """Load ONLY <run>/best_model.pth, fit temperature on VALIDATION, evaluate ONCE on TEST."""
    device = get_device()
    amp = amp_settings(cfg)
    run_dir = paths["checkpoints"] / run_name
    out = paths["results"] / "runs" / run_name
    out.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "best_model.pth"
    model, ck = load_model(ckpt, device)
    classes, spec = ck["class_names"], ck["preprocessing"]
    print_header(f"FINAL TEST EVALUATION -- {ck['model_display']} ({run_name}) | checkpoint epoch {ck['epoch']} "
                 f"(val macro-F1 {ck['val_macro_f1']:.4f})")
    dfs = load_split_frames(paths)
    ck_sha = sha256_file(ckpt)
    lock = out / "test_evaluation.lock"
    cache = out / "test_predictions_raw.npz"
    if lock.exists():
        L = json.load(open(lock))
        if L["checkpoint_sha256"] != ck_sha and not cfg.get("allow_test_reevaluation", False):
            raise RuntimeError(f"The test set was already evaluated for {run_name} with a DIFFERENT checkpoint "
                               f"({L['checkpoint_sha256'][:12]} vs {ck_sha[:12]}). Re-evaluating would turn the test set into a "
                               "selection set. Set allow_test_reevaluation=true only if this is intended and documented.")
    val = predict_logits(model, make_loader(dfs["val"], spec, cfg), device, amp)
    T, tinfo = fit_temperature(val["logits"], val["y"])
    print(f"Temperature fitted on VALIDATION: T = {T:.3f} (val NLL {tinfo['val_nll_before']:.4f} -> {tinfo['val_nll_after']:.4f})")
    if lock.exists() and cache.exists() and json.load(open(lock))["checkpoint_sha256"] == ck_sha:
        z = np.load(cache)
        test = {k: z[k] for k in z.files}
        print("Test predictions for this exact checkpoint already exist -> reusing them (the test set is evaluated once).")
    else:
        passes = int(cfg["uncertainty"].get("mc_dropout_passes", 0))
        test = predict_logits(model, make_loader(dfs["test"], spec, cfg), device, amp, mc_passes=passes)
        np.savez_compressed(cache, **test)
        json.dump({"checkpoint": str(ckpt), "checkpoint_sha256": ck_sha, "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "n_test": int(len(test["y"]))}, open(lock, "w"), indent=2)
    y, logits = test["y"], test["logits"]
    prob_raw, prob = softmax_np(logits, 1.0), softmax_np(logits, T)
    m = multiclass_metrics(y, prob_raw, classes)
    m_cal = multiclass_metrics(y, prob, classes)
    m["log_loss_temperature_scaled"] = m_cal["log_loss"]
    sev = [ {"Non Demented": 0, "Very mild Dementia": 1, "Mild Dementia": 2, "Moderate Dementia": 3}[c] for c in classes]
    binm, yb, score = binary_dementia_metrics(y, prob, classes) if "Non Demented" in classes else ({}, None, None)
    ordm = ordinal_metrics(y, prob, sev)
    calib = calibration_report(logits, y, T, cfg["calibration"]["n_bins"])
    calib["fit"] = tinfo
    unc, conf, nent, status = uncertainty_table(prob, y, cfg)
    mc_info = {}
    if "mc_logits" in test:
        ps = np.stack([softmax_np(l, T) for l in test["mc_logits"]])
        s = mc_dropout_summary(ps)
        err = (prob.argmax(1) != y).astype(int)
        from sklearn.metrics import roc_auc_score
        mc_info = {"passes": int(ps.shape[0]), "mean_mutual_information": float(s["mutual_information"].mean()),
                   "mc_accuracy": float((s["mean_prob"].argmax(1) == y).mean()),
                   "auroc_mutual_information_detects_errors": float(roc_auc_score(err, s["mutual_information"])) if 0 < err.sum() < len(err) else None,
                   "auroc_entropy_detects_errors": float(roc_auc_score(err, nent)) if 0 < err.sum() < len(err) else None}
        mi = s["mutual_information"]
    else:
        mi = np.full(len(y), np.nan)
    unc["mc_dropout"] = mc_info
    csa = class_specific_analysis(m, classes)

    rows = dfs["test"].iloc[test["idx"]].reset_index(drop=True).copy()
    rows["actual"] = [classes[i] for i in y]
    rows["predicted"] = [classes[i] for i in prob.argmax(1)]
    rows["correct"] = rows.actual == rows.predicted
    for k, c in enumerate(classes):
        rows[f"p_{c}"] = prob[:, k]
    rows["confidence"], rows["normalized_entropy"], rows["uncertainty_status"] = conf, nent, status
    rows["mc_mutual_information"] = mi
    rows["severity_error"] = np.abs(np.array(sev)[y] - np.array(sev)[prob.argmax(1)])
    rows.to_csv(out / "test_predictions.csv", index=False)

    rep = pd.DataFrame(m["per_class"]).T[["precision", "recall", "f1", "support", "roc_auc_ovr", "pr_auc"]]
    rep.loc["macro avg"] = [m["macro_precision"], m["macro_recall"], m["macro_f1"], m["n"], m["macro_roc_auc_ovr"], m["macro_pr_auc"]]
    rep.loc["weighted avg"] = [np.nan, np.nan, m["weighted_f1"], m["n"], m["weighted_roc_auc_ovr"], np.nan]
    rep.to_csv(out / "classification_report.csv")
    plot_confusion(m["confusion_matrix"], classes, out / "confusion_matrix.png")
    plot_confusion(m["confusion_matrix"], classes, out / "normalized_confusion_matrix.png", normalized=True)
    plot_roc_pr(y, prob, classes, out / "roc_curves.png", out / "precision_recall_curves.png")
    shutil.copy(out / "precision_recall_curves.png", out / "pr_curves.png")
    plot_reliability(logits, y, T, out / "calibration.png", cfg["calibration"]["n_bins"])
    plot_uncertainty(prob, y, conf, nent, out / "uncertainty.png")
    if yb is not None:
        plot_binary_curves(yb, score, out / "binary_dementia_curves.png")
    plot_confusion(np.array(binm["confusion_matrix"]), binm["labels"], out / "binary_confusion_matrix.png",
                   title="Non Demented vs Any Dementia (derived)") if binm else None

    bench = benchmark(model, spec["image_size"], device)
    perf = {"total_params": count_params(model), "trainable_params_stage2": None, "model_size_mb": model_size_mb(model),
            "input_resolution": [spec["image_size"], spec["image_size"]], **bench,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    result = {"model": ck["model_display"], "run_name": run_name, "checkpoint": str(ckpt), "checkpoint_sha256": ck_sha,
              "checkpoint_epoch": ck["epoch"], "selection": "validation macro-F1", "val_macro_f1_at_selection": ck["val_macro_f1"],
              "four_class": m, "four_class_temperature_scaled_note": "temperature scaling does not change argmax predictions",
              "binary_dementia_detection": binm, "ordinal_severity": ordm, "calibration": calib, "uncertainty": unc,
              "class_specific": csa, "performance": perf, "preprocessing": spec, "temperature": T}
    save_json(result, out / "metrics.json")
    save_json({"temperature": T, "uncertainty_thresholds": cfg["uncertainty"], "checkpoint_sha256": ck_sha,
               "class_names": classes, "preprocessing": spec, "model": ck["model_display"], "arch": ck["arch"]},
              run_dir / "inference_meta.json")
    _print_results(result, classes)
    if publish:
        for f in ("metrics.json", "classification_report.csv", "confusion_matrix.png", "normalized_confusion_matrix.png",
                  "roc_curves.png", "pr_curves.png", "precision_recall_curves.png", "calibration.png", "uncertainty.png",
                  "binary_dementia_curves.png", "binary_confusion_matrix.png", "test_predictions.csv"):
            if (out / f).exists():
                shutil.copy(out / f, paths["results"] / f)
        shutil.copy(run_dir / "inference_meta.json", paths["checkpoints"] / "inference_meta.json")
    return result


def _print_results(r, classes):
    m, b, o, c = r["four_class"], r["binary_dementia_detection"], r["ordinal_severity"], r["calibration"]
    print("\nFOUR-CLASS (test):")
    for k in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
              "macro_roc_auc_ovr", "weighted_roc_auc_ovr", "macro_pr_auc", "top3_accuracy"):
        print(f"  {k:<24} {m[k]:.4f}" if m[k] is not None and np.isfinite(m[k]) else f"  {k:<24} n/a")
    print("  per class:")
    for cname in classes:
        pc = m["per_class"][cname]
        print(f"    {cname:<20} P {pc['precision']:.3f} R {pc['recall']:.3f} F1 {pc['f1']:.3f} support {pc['support']}")
    if b:
        print("\nBINARY Non Demented vs Any Dementia (derived from the four-class probabilities):")
        for k in ("accuracy", "sensitivity", "specificity", "precision", "npv", "f1", "roc_auc", "pr_auc"):
            print(f"  {k:<24} {b[k]:.4f}" if b[k] is not None and np.isfinite(b[k]) else f"  {k:<24} n/a")
    print(f"\nORDINAL (analytical): severity MAE {o['mean_absolute_severity_error']:.3f} | QWK {o['quadratic_weighted_kappa']:.3f} "
          f"| adjacent errors {o['adjacent_errors']} | large (>=2) errors {o['large_errors_ge2']}")
    print(f"CALIBRATION: ECE {c['uncalibrated']['ece']:.4f} -> {c['temperature_scaled']['ece']:.4f} (T={c['temperature']:.2f}) | "
          f"Brier {c['uncalibrated']['brier']:.4f} -> {c['temperature_scaled']['brier']:.4f}")
    u = r["uncertainty"]["by_status"]
    print("UNCERTAINTY: " + " | ".join(f"{k}: n={v['n']} acc={v['accuracy']:.3f}" if v["accuracy"] is not None else f"{k}: n=0" for k, v in u.items()))
    cs = r["class_specific"]
    print(f"CLASS ANALYSIS: easiest {cs['easiest_class_by_f1']} | hardest {cs['hardest_class_by_f1']} | most confused "
          f"{cs['most_confused_pairs'][0]['pair'] if cs['most_confused_pairs'] else 'n/a'}")


def compare_models(paths, runs):
    rows = []
    for name in runs:
        r = load_json(paths["results"] / "runs" / name / "metrics.json")
        if not r:
            continue
        m = r["four_class"]
        rows.append({"run": name, "model": r["model"], "test_accuracy": m["accuracy"], "test_balanced_accuracy": m["balanced_accuracy"],
                     "test_macro_f1": m["macro_f1"], "test_macro_auc": m["macro_roc_auc_ovr"],
                     "val_macro_f1_at_selection": r["val_macro_f1_at_selection"],
                     "binary_roc_auc": r["binary_dementia_detection"].get("roc_auc"),
                     "severity_mae": r["ordinal_severity"]["mean_absolute_severity_error"],
                     "qwk": r["ordinal_severity"]["quadratic_weighted_kappa"],
                     "ece_temperature_scaled": r["calibration"]["temperature_scaled"]["ece"],
                     "params_M": r["performance"]["total_params"] / 1e6, "size_mb": r["performance"]["model_size_mb"],
                     "latency_ms": r["performance"]["latency_ms_batch1"], "images_per_sec": r["performance"]["images_per_sec_batch"]})
    df = pd.DataFrame(rows)
    df.to_csv(paths["results"] / "model_comparison.csv", index=False)
    return df


# ------------------------------------------------------------------------------------------- error analysis
def error_analysis(cfg, paths, run_name, n_examples=None):
    from scipy.stats import mannwhitneyu
    from .gradcam import cam_layers_for, explain_image
    out = paths["results"] / "runs" / run_name
    pred = pd.read_csv(out / "test_predictions.csv")
    model, ck = load_model(paths["checkpoints"] / run_name / "best_model.pth")
    classes, spec = ck["class_names"], ck["preprocessing"]
    layers, _, _ = cam_layers_for(model, spec["image_size"])
    k = n_examples or cfg["gradcam"]["examples_per_group"]
    groups = {
        "confident_correct": pred[pred.correct].sort_values("confidence", ascending=False).head(k),
        "confident_incorrect": pred[~pred.correct].sort_values("confidence", ascending=False).head(k),
        "low_confidence": pred.sort_values("confidence").head(k),
    }
    mis = pred[~pred.correct].copy()
    cam_dir = paths["gradcam"] / "errors"
    recs = []
    for g, d in list(groups.items()) + [("misclassified", mis.head(4 * k))]:
        for i, (_, r) in enumerate(d.iterrows()):
            info, _, _ = explain_image(model, r.filepath, spec, classes, out_dir=cam_dir / g,
                                       name=f"{g}_{i:02d}_{Path(r.filepath).stem[:40]}",
                                       true_label=classes.index(r.actual), layers=layers, multi=True)
            recs.append({"group": g, "filepath": r.filepath, "actual": r.actual, "predicted": r.predicted,
                         **{c: r[f"p_{c}"] for c in classes}, "confidence": r.confidence,
                         "entropy": r.normalized_entropy, "gradcam_panel": info["files"]["panel"]})
    ea = pd.DataFrame(recs)
    full = pred.copy()
    full["error_group"] = np.where(full.correct, "correct", "misclassified")
    cols = ["filepath", "actual", "predicted", "correct", "confidence", "normalized_entropy", "uncertainty_status",
            "brightness", "contrast", "sharpness", "black_frac", "source_group_id", "grouping_method", "severity_error"] + \
           [f"p_{c}" for c in classes]
    err_csv = full[~full.correct][[c for c in cols if c in full]].sort_values("confidence", ascending=False)
    err_csv = err_csv.merge(ea[["filepath", "gradcam_panel"]].drop_duplicates("filepath"), on="filepath", how="left")
    err_csv.to_csv(out / "error_analysis.csv", index=False)
    summary = {"n_test": int(len(pred)), "n_errors": int((~pred.correct).sum()),
               "errors_by_actual_class": pred[~pred.correct].actual.value_counts().to_dict(),
               "error_rate_by_actual_class": (1 - pred.groupby("actual").correct.mean()).round(4).to_dict(),
               "confusions": pred[~pred.correct].groupby(["actual", "predicted"]).size().sort_values(ascending=False).head(10).to_dict(),
               "errors_concentrated_in_groups": None, "image_property_tests": {}}
    if (~pred.correct).any() and "source_group_id" in pred:
        eg = pred[~pred.correct].source_group_id.value_counts()
        summary["errors_concentrated_in_groups"] = {"groups_with_errors": int(len(eg)),
                                                    "share_of_errors_in_top5_groups": float(eg.head(5).sum() / eg.sum())}
    for col in ("brightness", "contrast", "sharpness", "black_frac"):
        if col in pred and pred.correct.any() and (~pred.correct).any():
            a, b = pred.loc[pred.correct, col].dropna(), pred.loc[~pred.correct, col].dropna()
            if len(a) > 1 and len(b) > 1:
                summary["image_property_tests"][col] = {"mean_correct": float(a.mean()), "mean_incorrect": float(b.mean()),
                                                        "mannwhitney_p": float(mannwhitneyu(a, b).pvalue)}
    q = pd.qcut(pred.brightness, 4, duplicates="drop") if "brightness" in pred else None
    if q is not None:
        summary["error_rate_by_brightness_quartile"] = {str(k): float(v) for k, v in (1 - pred.groupby(q, observed=True).correct.mean()).items()}
    save_json(summary, out / "error_analysis_summary.json")
    shutil.copy(out / "error_analysis.csv", paths["results"] / "error_analysis.csv")
    save_json(summary, paths["results"] / "error_analysis_summary.json")
    return summary, err_csv


# ---------------------------------------------------------------------------------------- external
def external_validation(cfg, paths, run_name):
    root = cfg.get("external_data_root")
    statement = "Independent external validation was not performed."
    if not root:
        save_json({"performed": False, "statement": statement}, paths["results"] / "external_validation.json")
        print(statement)
        return {"performed": False, "statement": statement}
    from .data_audit import discover_classes, list_images
    files, _ = list_images(root)
    disc, unknown = discover_classes(files, root)
    model, ck = load_model(paths["checkpoints"] / run_name / "best_model.pth")
    classes = ck["class_names"]
    disc = disc[disc["class"].isin(classes)].copy()
    if disc.empty:
        out = {"performed": False, "statement": statement + f" (no compatible class folders under {root})"}
        save_json(out, paths["results"] / "external_validation.json"); print(out["statement"]); return out
    disc["class_index"] = disc["class"].map({c: i for i, c in enumerate(classes)})
    dev = get_device()
    pred = predict_logits(model, make_loader(disc, ck["preprocessing"], cfg), dev, amp_settings(cfg))
    m = multiclass_metrics(pred["y"], softmax_np(pred["logits"]), classes)
    plot_confusion(m["confusion_matrix"], classes, paths["results"] / "external_confusion_matrix.png", title="External validation")
    out = {"performed": True, "root": str(root), "n": m["n"], "external_accuracy": m["accuracy"], "external_macro_f1": m["macro_f1"],
           "external_per_class_f1": {c: m["per_class"][c]["f1"] for c in classes}, "confusion_matrix": m["confusion_matrix"],
           "note": "Evaluated without retraining; not used for model selection."}
    save_json(out, paths["results"] / "external_validation.json")
    print(f"External validation: accuracy {out['external_accuracy']:.4f}, macro-F1 {out['external_macro_f1']:.4f} (n={out['n']})")
    return out
