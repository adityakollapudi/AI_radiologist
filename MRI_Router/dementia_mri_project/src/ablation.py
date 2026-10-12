"""Ablation study on the SAME leakage-free split, scored on VALIDATION only (the test set is not used)."""
import pandas as pd

from .calibration import ece_mce, softmax_np
from .dataset import load_split_frames, make_loader
from .evaluate import benchmark, load_model, predict_logits
from .metrics import multiclass_metrics
from .model import count_params, model_size_mb
from .train import train_model
from .utils import amp_settings, get_device, load_json, print_header, save_json


def ablation_plan(cfg, paths):
    pre = load_json(paths["data"] / "split_summary.json")["preprocessing"]
    primary_color = pre["color_mode"] if cfg["color_mode"] == "auto" else cfg["color_mode"]
    other = "rgb" if primary_color == "gray3" else "gray3"
    P, B = cfg["model"], cfg["baseline_model"]
    return [
        {"id": "E1", "name": "EfficientNet-B0 (baseline)", "run": "baseline", "arch": B, "color": primary_color, "aug": True},
        {"id": "E2", "name": "EfficientNetV2-S (primary)", "run": "primary", "arch": P, "color": primary_color, "aug": True},
        {"id": "E3", "name": "RGB input", "run": "primary" if primary_color == "rgb" else "ablation_rgb", "arch": P, "color": "rgb", "aug": True},
        {"id": "E4", "name": "Grayscale -> 3-channel", "run": "primary" if primary_color == "gray3" else "ablation_gray3", "arch": P, "color": "gray3", "aug": True},
        {"id": "E5", "name": "Without augmentation", "run": "ablation_noaug", "arch": P, "color": primary_color, "aug": False},
        {"id": "E6", "name": "With conservative augmentation", "run": "primary", "arch": P, "color": primary_color, "aug": True},
    ]


def run_ablation(cfg, paths):
    print_header("ABLATION STUDY (validation set only)")
    ab = cfg.get("ablation", {})
    if not ab.get("enabled", True):
        print("Ablation disabled in config."); return None
    e1 = ab.get("epochs_stage1") or cfg["epochs_stage1"]
    e2 = ab.get("epochs_stage2") or cfg["epochs_stage2"]
    plan = ablation_plan(cfg, paths)
    dev = get_device()
    val = load_split_frames(paths)["val"]
    cache = {}
    rows = []
    for e in plan:
        run = e["run"]
        if run not in cache:
            ck = paths["checkpoints"] / run / "best_model.pth"
            budget_note = "main training budget"
            if not ck.exists():
                same = (e1, e2) == (cfg["epochs_stage1"], cfg["epochs_stage2"])
                budget_note = "main training budget" if same else f"reduced budget ({e1}+{e2} epochs)"
                train_model(cfg, paths, run, e["arch"], color_mode=e["color"], aug_enabled=e["aug"], epochs=(e1, e2))
            model, ckd = load_model(ck, dev)
            p = predict_logits(model, make_loader(val, ckd["preprocessing"], cfg), dev, amp_settings(cfg))
            prob = softmax_np(p["logits"])
            m = multiclass_metrics(p["y"], prob, ckd["class_names"])
            ece, _, _ = ece_mce(prob, p["y"], cfg["calibration"]["n_bins"])
            b = benchmark(model, ckd["preprocessing"]["image_size"], dev)
            cache[run] = {"val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"], "val_macro_auc": m["macro_roc_auc_ovr"],
                          "val_ece_uncalibrated": ece, "params_M": count_params(model) / 1e6, "size_mb": model_size_mb(model),
                          "latency_ms": b["latency_ms_batch1"], "images_per_sec": b["images_per_sec_batch"],
                          "best_epoch": ckd["epoch"], "budget": budget_note}
        rows.append({"experiment": e["id"], "description": e["name"], "run": run, "model": e["arch"], "color_mode": e["color"],
                     "augmentation": e["aug"], **cache[run]})
        print(f"  {e['id']} {e['name']:<34} run={run:<16} val acc {cache[run]['val_accuracy']:.4f} "
              f"macro-F1 {cache[run]['val_macro_f1']:.4f} AUC {cache[run]['val_macro_auc']:.4f}")
    df = pd.DataFrame(rows)
    pre = load_json(paths["data"] / "split_summary.json")["preprocessing"]
    if pre.get("rgb_near_identical_fraction", 0) >= 0.99:
        print("  note: R=G=B in this dataset, so RGB and grayscale->3-channel inputs are (nearly) pixel-identical; "
              "E3 and E4 are expected to match.")
    df.to_csv(paths["results"] / "ablation_results.csv", index=False)
    save_json({"note": "All experiments use the same split, preprocessing family, metrics and training strategy; "
                       "scores are on VALIDATION (test not used). Rows sharing a run are the same trained model.",
               "rows": df.to_dict(orient="records")}, paths["results"] / "ablation_results.json")
    return df
