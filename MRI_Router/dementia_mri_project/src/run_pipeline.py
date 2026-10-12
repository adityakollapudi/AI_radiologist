"""Pipeline stages (used by the notebooks and the command line).

    python -m src.run_pipeline --stage audit
    python -m src.run_pipeline --stage split
    python -m src.run_pipeline --stage train
    python -m src.run_pipeline --stage evaluate
    python -m src.run_pipeline --stage gradcam
    python -m src.run_pipeline --stage all  [--set batch_size=16 --set epochs_stage2=30]
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import PROJECT_ROOT, get_paths, load_config, load_json, parse_cli_overrides, print_env, print_header, save_json

PRIMARY, BASELINE = "primary", "baseline"


def stage_audit(cfg, paths):
    from .data_audit import run_audit
    return run_audit(cfg, paths)


def stage_split(cfg, paths):
    from .grouping import run_split
    if not (paths["audit"] / "dataset_audit.csv").exists():
        raise RuntimeError("Run the audit stage first (RULE 1: no training before the dataset audit).")
    return run_split(cfg, paths)


def stage_train(cfg, paths, which=("baseline", "primary")):
    from .train import train_model
    if not (paths["data"] / "split_summary.json").exists():
        raise RuntimeError("Run the split stage first.")
    out = load_json(paths["results"] / "train_runs.json", {}) or {}
    if "baseline" in which:
        out[BASELINE] = train_model(cfg, paths, BASELINE, cfg["baseline_model"])
    if "primary" in which:
        out[PRIMARY] = train_model(cfg, paths, PRIMARY, cfg["model"], publish=True)
    save_json(out, paths["results"] / "train_runs.json")
    return out


def stage_evaluate(cfg, paths, robustness=True, ablation=True):
    from .evaluate import compare_models, evaluate_run, external_validation
    res = {}
    for run in (BASELINE, PRIMARY):
        if (paths["checkpoints"] / run / "best_model.pth").exists():
            res[run] = evaluate_run(cfg, paths, run, publish=(run == PRIMARY))
    comp = compare_models(paths, [BASELINE, PRIMARY])
    print_header("BASELINE vs PRIMARY (test, each evaluated once)")
    print(comp.round(4).to_string(index=False))
    if robustness and PRIMARY in res:
        from .robustness import run_robustness
        run_robustness(cfg, paths, PRIMARY)
    if ablation:
        from .ablation import run_ablation
        run_ablation(cfg, paths)
    if PRIMARY in res:
        external_validation(cfg, paths, PRIMARY)
    return res


def stage_gradcam(cfg, paths, run=PRIMARY):
    import torch
    from .evaluate import error_analysis, load_model
    from .gradcam import FUSION_METHOD, cam_layers_for, explain_image, sanity_checks
    from .preprocessing import load_rgb, model_input_image, normalize_tensor
    print_header("GRAD-CAM ANALYSIS")
    model, ck = load_model(paths["checkpoints"] / run / "best_model.pth")
    classes, spec = ck["class_names"], ck["preprocessing"]
    layers, sel, shapes = cam_layers_for(model, spec["image_size"])
    print("Feature-map shapes:", shapes)
    print("Selected layers (from measured shapes):", sel)
    pred = pd.read_csv(paths["results"] / "runs" / run / "test_predictions.csv")
    k = cfg["gradcam"]["examples_per_group"]
    infos = []
    # 1) standard Grad-CAM verified first, predicted class + every manually selected target class
    r0 = pred.iloc[0]
    for tgt in [None] + list(range(len(classes))):
        tag = "predicted" if tgt is None else f"target_{classes[tgt].replace(' ', '_')}"
        info, _, _ = explain_image(model, r0.filepath, spec, classes, target=tgt, out_dir=paths["gradcam"] / "standard",
                                   name=f"standard_{tag}", true_label=classes.index(r0.actual), layers=layers, multi=False)
        infos.append(info)
    std_ok = all(i["checks"]["all_layer_checks_passed"] and i["checks"]["model_output_consistent"] for i in infos)
    print(f"Standard Grad-CAM validity checks passed: {std_ok}")
    if not std_ok:
        raise RuntimeError(f"Standard Grad-CAM failed its checks: {[i['checks'] for i in infos]}")
    # 2) multi-layer Grad-CAM examples per class (correct and incorrect)
    for cname in classes:
        sub = pred[pred.actual == cname]
        for kind, d in (("correct", sub[sub.correct].sort_values("confidence", ascending=False).head(k)),
                        ("incorrect", sub[~sub.correct].sort_values("confidence", ascending=False).head(k))):
            for i, (_, r) in enumerate(d.iterrows()):
                info, _, _ = explain_image(model, r.filepath, spec, classes, out_dir=paths["gradcam"] / cname.replace(" ", "_"),
                                           name=f"{kind}_{i:02d}_{Path(r.filepath).stem[:40]}", true_label=classes.index(cname),
                                           layers=layers, multi=True)
                infos.append(info)
    validity = {"maps_checked": len(infos),
                "maps_with_empty_layer_cam": sum(bool(i["checks"]["empty_maps"]) for i in infos),
                "empty_final_layer_cam_for_predicted_class": sum("final" in i["checks"]["empty_maps"] and i["target_class"] == i["predicted_class"] for i in infos),
                "all_layer_checks_passed": all(i["checks"]["all_layer_checks_passed"] for i in infos),
                "model_output_consistent": all(i["checks"]["model_output_consistent"] for i in infos),
                "target_class_used": all(i["checks"]["target_class_used"] for i in infos)}
    print("Grad-CAM validity:", validity)
    # 3) sanity checks
    n = min(cfg["gradcam"]["sanity_samples"], len(pred))
    xs = []
    for p_ in pred.sample(n, random_state=cfg["seed"]).filepath:
        inp = model_input_image(load_rgb(p_), spec)
        xs.append(normalize_tensor(torch.from_numpy(np.asarray(inp, dtype=np.float32) / 255.0).permute(2, 0, 1), spec)[None])
    sanity, sdf = sanity_checks(model, xs, layers, cfg["gradcam"]["occlusion_fraction"], cfg["seed"])
    sdf.to_csv(paths["results"] / "gradcam_sanity_checks.csv", index=False)
    print("Sanity checks:", sanity)
    summary = {"layers": layers, "layer_details": sel, "feature_shapes": shapes, "fusion_method": FUSION_METHOD,
               "validity": validity, "sanity": sanity,
               "note": "Grad-CAM is an attribution technique, not a lesion segmentation, biomarker, proof of pathology or diagnosis."}
    save_json(summary, paths["results"] / "gradcam_summary.json")
    # 4) error analysis (with Grad-CAM)
    print_header("ERROR ANALYSIS")
    es, _ = error_analysis(cfg, paths, run)
    print(es)
    return summary


def stage_report(cfg, paths):
    from .report import write_final_report
    p = write_final_report(paths, cfg)
    print("Final report:", p)
    return p


def stage_package(cfg, paths):
    pk = paths["results"] / "model_package"
    if pk.exists():
        shutil.rmtree(pk)
    (pk / "checkpoints").mkdir(parents=True)
    for f in ("best_model.pth", "inference_meta.json", "class_mapping.json"):
        if (paths["checkpoints"] / f).exists():
            shutil.copy(paths["checkpoints"] / f, pk / "checkpoints" / f)
    shutil.copytree(PROJECT_ROOT / "src", pk / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(PROJECT_ROOT / "predict.py", pk / "predict.py")
    (pk / "configs").mkdir()
    for f in ("training_config_used.json",):
        if (paths["configs"] / f).exists():
            shutil.copy(paths["configs"] / f, pk / "configs" / f)
    shutil.copy(PROJECT_ROOT / "configs" / "config.json", pk / "configs" / "config.json")
    if (paths["root"] / "FINAL_REPORT.md").exists():
        shutil.copy(paths["root"] / "FINAL_REPORT.md", pk / "FINAL_REPORT.md")
    (pk / "README.md").write_text("Dementia MRI research prototype -- model package\n\n"
                                  "    pip install torch torchvision pillow numpy pandas scipy scikit-learn matplotlib\n"
                                  "    python predict.py --image /path/to/mri.jpg\n\n"
                                  "Not a diagnostic device. See FINAL_REPORT.md for evaluation and limitations.\n")
    z = shutil.make_archive(str(paths["results"] / "model_package"), "zip", pk)
    print("Model package:", z)
    return z


STAGES = ["audit", "split", "train", "evaluate", "gradcam", "report", "package"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", default=["all"], choices=STAGES + ["all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override config keys, e.g. batch_size=16 augmentation.enabled=false")
    a = ap.parse_args()
    cfg = load_config(a.config, parse_cli_overrides(a.set))
    paths = get_paths(cfg)
    print_env()
    stages = STAGES if "all" in a.stage else a.stage
    for s in stages:
        globals()[f"stage_{s}"](cfg, paths)


if __name__ == "__main__":
    main()
