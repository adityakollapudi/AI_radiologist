"""Two-stage transfer learning with AMP, discriminative learning rates, ReduceLROnPlateau on validation
Macro-F1, early stopping, best/last checkpoints, resume and OOM recovery. The test set is never touched here."""
import copy
import hashlib
import json
import math
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .dataset import load_split_frames, make_loader
from .metrics import macro_scores, plot_training_curves
from .model import DISPLAY, build_model, count_params, hold_frozen_bn, model_size_mb, param_groups, set_trainable
from .preprocessing import preprocessing_spec
from .utils import amp_settings, env_info, get_device, load_json, print_header, save_json, seed_everything, sha256_file


def _is_oom(e):
    return isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()


def run_epoch(model, loader, criterion, device, amp, opt=None, scaler=None, grad_clip=None, desc=""):
    train = opt is not None
    if train:
        model.train(); hold_frozen_bn(model)
    else:
        model.eval()
    enabled, dtype, _ = amp
    tot_loss, ys, ps, logits_all, idx_all = 0.0, [], [], [], []
    t0 = time.time()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                out = model(x)
                loss = criterion(out.float(), y)
            if train:
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss {loss.item()}")
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
                scaler.step(opt); scaler.update()
            tot_loss += loss.item() * len(y)
            ys.append(y.cpu()); ps.append(out.argmax(1).cpu())
            if not train:
                logits_all.append(out.float().cpu()); idx_all.append(idx)
    y = torch.cat(ys).numpy(); p = torch.cat(ps).numpy()
    m = macro_scores(y, p)
    m["loss"] = tot_loss / max(1, len(y))
    m["seconds"] = time.time() - t0
    m["images_per_sec"] = len(y) / max(m["seconds"], 1e-9)
    if not train:
        m["_logits"] = torch.cat(logits_all).numpy(); m["_y"] = y; m["_idx"] = torch.cat(idx_all).numpy()
    return m


def _make_scaler(device, enabled):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled and device.type == "cuda")
    except (AttributeError, TypeError):                      # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def _config_hash(rc):
    keep = {k: v for k, v in rc.items() if k not in ("env", "created", "resume_note")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def train_model(cfg, paths, run_name, arch, color_mode=None, aug_enabled=None, epochs=None, publish=False):
    """Train one model on the saved leakage-free split. Returns a summary dict."""
    seed_everything(cfg["seed"])
    device = get_device()
    amp = amp_settings(cfg)
    split_summary = load_json(paths["data"] / "split_summary.json")
    if split_summary is None:
        raise RuntimeError("Run the split stage first (data/split_summary.json missing).")
    if not all(split_summary["leakage_checks"].get(k, True) for k in split_summary["leakage_checks"] if not k.startswith("info_")):
        raise RuntimeError("Leakage checks did not pass -- training blocked.")
    classes = split_summary["classes"]
    dfs = load_split_frames(paths)
    c = copy.deepcopy(cfg)
    if color_mode:
        c["color_mode"] = color_mode
    spec = preprocessing_spec(c, split_summary["preprocessing"])
    aug = copy.deepcopy(cfg["augmentation"])
    if aug_enabled is not None:
        aug["enabled"] = bool(aug_enabled)
    e1, e2 = (epochs or (cfg["epochs_stage1"], cfg["epochs_stage2"]))
    run_dir = paths["checkpoints"] / run_name
    res_dir = paths["results"] / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(parents=True, exist_ok=True)

    rc = {"run_name": run_name, "model": arch, "model_display": DISPLAY.get(arch, arch), "classes": classes,
          "num_classes": len(classes), "input_size": [spec["image_size"], spec["image_size"]], "preprocessing": spec,
          "augmentation": aug, "batch_size": cfg["batch_size"], "eval_batch_size": cfg["eval_batch_size"],
          "epochs_stage1": e1, "epochs_stage2": e2, "head_lr": cfg["head_lr"], "finetune_head_lr": cfg["finetune_head_lr"],
          "backbone_lr": cfg["backbone_lr"], "weight_decay": cfg["weight_decay"], "optimizer": "AdamW",
          "scheduler": cfg["scheduler"], "scheduler_patience": cfg["scheduler_patience"],
          "scheduler_factor": cfg["scheduler_factor"], "min_lr": cfg["min_lr"],
          "early_stopping_patience": cfg["early_stopping_patience"], "unfreeze_fraction": cfg["unfreeze_fraction"],
          "label_smoothing": cfg["label_smoothing"], "class_weights": split_summary.get("class_weights"),
          "dropout": cfg["dropout"], "dropout2": cfg.get("dropout2", 0.2), "hidden_dim": cfg.get("hidden_dim", 256),
          "pretrained": cfg["pretrained"], "amp": bool(amp[0]), "amp_dtype": str(amp[1]), "seed": cfg["seed"],
          "dataset_version": cfg["dataset_version"], "selection_metric": "validation macro-F1",
          "split_files_sha256": {s: sha256_file(paths["data"] / f"{s}.csv")[:16] for s in ("train", "val", "test")}}
    rc["config_hash"] = _config_hash(rc)
    rc["env"] = env_info()
    save_json(rc, run_dir / "config.json")
    shutil.copy(paths["data"] / "class_mapping.json", run_dir / "class_mapping.json")

    print_header(f"TRAINING {run_name}: {rc['model_display']} | {spec['color_mode']} | aug={aug['enabled']} | "
                 f"{spec['image_size']}px | device {device} ({rc['env'].get('gpu')}) | AMP {amp[1]}")
    print(f"train {len(dfs['train']):,} | val {len(dfs['val']):,} images (test untouched)")

    model = build_model(arch, cfg).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    weights = rc["class_weights"]
    wt = torch.tensor([weights[cn] for cn in classes], dtype=torch.float32, device=device) if weights else None
    criterion = nn.CrossEntropyLoss(weight=wt, label_smoothing=float(cfg["label_smoothing"]))
    bs = int(cfg["batch_size"])
    train_loader = make_loader(dfs["train"], spec, cfg, train=True, aug=aug, batch_size=bs)
    val_loader = make_loader(dfs["val"], spec, cfg, train=False)

    state = {"stage": 1, "epoch_in_stage": 0, "global_epoch": 0, "best_f1": -1.0, "best_loss": float("inf"),
             "best_epoch": None, "es_counter": 0, "history": [], "batch_size": bs}
    last_path, best_path = run_dir / "last_checkpoint.pth", run_dir / "best_model.pth"
    resume = None
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck.get("config_hash") == rc["config_hash"]:
            model.load_state_dict(ck["model_state"]); state = ck["state"]; resume = ck
            print(f"Resuming {run_name}: stage {state['stage']}, epoch {state['epoch_in_stage']} in stage, best val macro-F1 {state['best_f1']:.4f}")
            if state["batch_size"] != bs:
                bs = state["batch_size"]; train_loader = make_loader(dfs["train"], spec, cfg, train=True, aug=aug, batch_size=bs)
        else:
            print("Existing last_checkpoint.pth belongs to a different configuration -> starting fresh (old file kept as .old)")
            os.replace(last_path, str(last_path) + ".old")

    t_start = time.time()
    for stage, n_ep in ((1, e1), (2, e2)):
        if state["stage"] > stage or n_ep == 0:
            continue
        info = set_trainable(model, stage, cfg["unfreeze_fraction"])
        opt = torch.optim.AdamW(param_groups(model, stage, cfg), weight_decay=cfg["weight_decay"])
        if cfg["scheduler"] == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_ep), eta_min=cfg["min_lr"])
        else:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=cfg["scheduler_factor"],
                                                               patience=cfg["scheduler_patience"], min_lr=cfg["min_lr"])
        scaler = _make_scaler(device, amp[2])
        if resume is not None and resume["state"]["stage"] == stage and state["epoch_in_stage"] > 0:
            opt.load_state_dict(resume["optimizer"]); sched.load_state_dict(resume["scheduler"]); scaler.load_state_dict(resume["scaler"])
        if stage == 2 and state["epoch_in_stage"] == 0:
            state["es_counter"] = 0
        print(f"\n--- Stage {stage}: {n_ep} epochs | trainable params {count_params(model, True):,} / {count_params(model):,}"
              + (f" | unfrozen {len(info['unfrozen_units'])} backbone units = {info['unfrozen_backbone_fraction']:.1%} of backbone "
                 f"({info['unfrozen_units'][0]} ... {info['unfrozen_units'][-1]})" if stage == 2 else " | backbone frozen"))
        stop = False
        for ep in range(state["epoch_in_stage"], n_ep):
            for attempt in range(int(cfg.get("max_oom_retries", 3)) + 1):
                try:
                    tr = run_epoch(model, train_loader, criterion, device, amp, opt, scaler, cfg.get("grad_clip"))
                    va = run_epoch(model, val_loader, criterion, device, amp)
                    break
                except RuntimeError as e:
                    if not _is_oom(e) or attempt == int(cfg.get("max_oom_retries", 3)) or bs <= 2:
                        raise
                    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    bs = max(2, bs // 2); state["batch_size"] = bs
                    print(f"  !! CUDA out of memory -> retrying this epoch with batch size {bs}")
                    train_loader = make_loader(dfs["train"], spec, cfg, train=True, aug=aug, batch_size=bs)
                    val_loader = make_loader(dfs["val"], spec, cfg, train=False, batch_size=bs)
            state["global_epoch"] += 1
            f1 = va["macro_f1"]
            improved = f1 > state["best_f1"] + 1e-6 or (abs(f1 - state["best_f1"]) <= 1e-6 and va["loss"] < state["best_loss"])
            if cfg["scheduler"] == "cosine":
                sched.step()
            else:
                sched.step(f1)
            if improved:
                state.update(best_f1=f1, best_loss=va["loss"], best_epoch=state["global_epoch"])
                if stage == 2:
                    state["es_counter"] = 0
                torch.save({"model_state": model.state_dict(), "arch": arch, "model_display": rc["model_display"],
                            "class_names": classes, "num_classes": len(classes), "preprocessing": spec,
                            "head": {"dropout": cfg["dropout"], "hidden": cfg.get("hidden_dim", 256), "dropout2": cfg.get("dropout2", 0.2)},
                            "epoch": state["global_epoch"], "stage": stage, "val_macro_f1": f1, "val_loss": va["loss"],
                            "run_config": rc, "temperature": None}, best_path)
            elif stage == 2:
                state["es_counter"] += 1
            lrs = {f"lr_{g.get('name', i)}": g["lr"] for i, g in enumerate(opt.param_groups)}
            row = {"epoch": state["global_epoch"], "stage": stage,
                   **{f"train_{k}": tr[k] for k in ("loss", "accuracy", "macro_precision", "macro_recall", "macro_f1")},
                   **{f"val_{k}": va[k] for k in ("loss", "accuracy", "macro_precision", "macro_recall", "macro_f1")},
                   **lrs, "best_val_macro_f1": state["best_f1"], "early_stopping_counter": state["es_counter"],
                   "train_seconds": tr["seconds"], "train_images_per_sec": tr["images_per_sec"], "batch_size": bs}
            state["history"].append(row)
            state["epoch_in_stage"] = ep + 1
            print(f"\nEpoch {state['global_epoch']}/{e1 + e2}  (stage {stage}, {ep + 1}/{n_ep})")
            print(f"  TRAIN  loss {tr['loss']:.4f} | acc {tr['accuracy']:.4f} | macro P {tr['macro_precision']:.4f} "
                  f"R {tr['macro_recall']:.4f} F1 {tr['macro_f1']:.4f}")
            print(f"  VAL    loss {va['loss']:.4f} | acc {va['accuracy']:.4f} | macro P {va['macro_precision']:.4f} "
                  f"R {va['macro_recall']:.4f} F1 {va['macro_f1']:.4f}{'  * new best' if improved else ''}")
            print(f"  LR {', '.join(f'{k}={v:.2e}' for k, v in lrs.items())} | best val macro-F1 {state['best_f1']:.4f} "
                  f"(epoch {state['best_epoch']}) | early-stopping counter {state['es_counter']}/{cfg['early_stopping_patience']} "
                  f"| {tr['seconds']:.0f}s, {tr['images_per_sec']:.0f} img/s")
            torch.save({"model_state": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "scaler": scaler.state_dict(), "state": state, "config_hash": rc["config_hash"], "arch": arch}, last_path)
            pd.DataFrame(state["history"]).to_csv(run_dir / "training_history.csv", index=False)
            if stage == 2 and state["es_counter"] >= cfg["early_stopping_patience"]:
                print(f"Early stopping: validation macro-F1 has not improved for {state['es_counter']} epochs.")
                stop = True
                break
        state["stage"], state["epoch_in_stage"] = stage + 1, 0
        resume = None
        torch.save({"model_state": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "state": state, "config_hash": rc["config_hash"], "arch": arch}, last_path)
        if stop:
            break

    hist = pd.DataFrame(state["history"])
    hist.to_csv(run_dir / "training_history.csv", index=False)
    hist.to_csv(res_dir / "training_history.csv", index=False)
    plot_training_curves(hist, res_dir)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state"])
    summary = {"run_name": run_name, "model": arch, "model_display": rc["model_display"], "color_mode": spec["color_mode"],
               "augmentation": aug["enabled"], "best_epoch": state["best_epoch"], "best_val_macro_f1": state["best_f1"],
               "best_val_loss": state["best_loss"], "epochs_run": len(hist), "train_minutes": (time.time() - t_start) / 60,
               "total_params": count_params(model), "model_size_mb": model_size_mb(model),
               "checkpoint": str(best_path), "run_dir": str(run_dir), "results_dir": str(res_dir)}
    save_json(summary, res_dir / "train_summary.json")
    if publish:
        shutil.copy(best_path, paths["checkpoints"] / "best_model.pth")
        shutil.copy(last_path, paths["checkpoints"] / "last_checkpoint.pth")
        hist.to_csv(paths["results"] / "training_history.csv", index=False)
        plot_training_curves(hist, paths["results"])
        save_json(rc, paths["configs"] / "training_config_used.json")
        shutil.copy(paths["data"] / "class_mapping.json", paths["checkpoints"] / "class_mapping.json")
        save_json(rc, paths["checkpoints"] / "config.json")
        hist.to_csv(paths["checkpoints"] / "training_history.csv", index=False)
    print(f"\nBest validation macro-F1 {state['best_f1']:.4f} at epoch {state['best_epoch']} -> {best_path}")
    return summary
