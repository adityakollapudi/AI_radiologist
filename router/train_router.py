#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
router/train_router.py
======================
Trains the CHEST vs MUSCULOSKELETAL router used by pipeline.py.

    python router/train_router.py

DATASETS (both already used by the two downstream branches -- nothing invented)
------------------------------------------------------------------------------
  CLASS 0  CHEST            paultimothymooney/chest-xray-pneumonia
  CLASS 1  MUSCULOSKELETAL  cjinny/mura-v11  (XR_ELBOW, XR_FINGER, XR_FOREARM,
                                              XR_HAND, XR_HUMERUS, XR_SHOULDER,
                                              XR_WRIST)

THE LABEL IS BODY REGION, NOT DISEASE. This is the point of the whole router:
  * a chest film is CHEST whether it is NORMAL or PNEUMONIA
  * a wrist film is MUSCULOSKELETAL whether it is NORMAL or ABNORMAL
Both disease classes of each dataset are pooled into one router class, and the
router is never shown a disease label.

LEAKAGE CONTROL -- three separate layers
----------------------------------------
1. DOWNSTREAM ISOLATION. The router is trained ONLY from the `train/` portion of
   each dataset: chest_xray/train/ and MURA-v1.1/train/. The chest test/ folder
   and MURA's official valid/ folder -- which are the held-out test sets of the
   two downstream models -- are never touched. So a pipeline demo run on a
   downstream test image is genuinely unseen by every model in the chain.
2. GROUP-LEVEL SPLIT. The router's own train/val/test split is made on patient
   or study keys (MURA patient ID, chest person ID / accession), never on
   individual images, so no patient straddles two router splits.
3. ARCHIVE HYGIENE. The chest archive ships a byte-identical nested copy
   (chest_xray/chest_xray/) and a __MACOSX/ tree of AppleDouble sidecars that
   carry .jpeg extensions but hold no pixel data. Both are excluded, as is
   MURA's own AppleDouble residue.

Filename and path text is used HERE, for building the training set, and nowhere
else. At inference the router sees only pixels.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import sys
import time
import json
import math
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_model import (                                     # noqa: E402
    XRayRouter, RouterDataset, router_seed_worker,
    build_router_transform, save_router_checkpoint,
    CLASS_NAMES, CLASS_TO_INDEX, NUM_CLASSES,
    ROUTER_IMAGE_SIZE, ROUTER_MEAN, ROUTER_STD, VALID_EXT,
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SEED = 42

# Set these to skip the kagglehub download if you already have the data locally.
CHEST_ROOT = None        # folder containing train/ val/ test/  (chest_xray)
MURA_ROOT = None         # folder containing train/ valid/     (MURA-v1.1)

CHEST_DATASET = "paultimothymooney/chest-xray-pneumonia"
MURA_DATASET = "cjinny/mura-v11"

# MURA train has ~37k images vs the chest set's ~5.2k. Capping both keeps the
# router balanced without any loss weighting, and routing is an easy task -- a
# few thousand images per class is ample.
MAX_PER_CLASS = 4000

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

NUM_EPOCHS_STAGE1 = 2          # frozen backbone, head only
NUM_EPOCHS_STAGE2 = 6          # partial unfreeze
LEARNING_RATE_S1 = 1e-3
LEARNING_RATE_S2 = 1e-4
BACKBONE_LR_MULT = 0.25
WEIGHT_DECAY = 1e-4
UNFREEZE_FROM = 4              # EfficientNet-B0 has 9 feature stages
PATIENCE = 3
GRAD_CLIP_NORM = 5.0

HERE = Path(__file__).resolve().parent
CHECKPOINT_PATH = HERE / "xray_router_best.pth"
OUTPUT_DIR = HERE / "router_outputs"

EXCLUDE_DIRS = {"__macosx"}


# ==============================================================================
# ENVIRONMENT
# ==============================================================================

def setup_environment():
    print("=" * 78)
    print("ROUTER TRAINING -- CHEST vs MUSCULOSKELETAL")
    print("=" * 78)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch        :", torch.__version__)
    print("CUDA available :", torch.cuda.is_available())

    total_gb = 0.0
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print("GPU            :", torch.cuda.get_device_name(0))
        print("GPU memory     : {:.2f} GB".format(total_gb))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print("Device         :", device)

    amp_dtype = None
    if device.type == "cuda":
        try:
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            amp_dtype = torch.float16
    scaler_enabled = amp_dtype == torch.float16
    print("AMP dtype      :",
          "disabled" if amp_dtype is None else str(amp_dtype).replace("torch.", ""),
          "| GradScaler:", scaler_enabled)

    # EfficientNet-B0 at 224 is far lighter than the B3 disease models, so an
    # 8 GB card takes a large batch comfortably.
    if device.type != "cuda":
        batch_size = 8
    elif total_gb >= 14:
        batch_size = 64
    else:
        batch_size = 32                      # RTX 4060 8 GB
    num_workers = min(4, os.cpu_count() or 2) if device.type == "cuda" else 0

    print(f"Batch size     : {batch_size}")
    print(f"Num workers    : {num_workers}")

    return {"device": device, "amp_dtype": amp_dtype,
            "scaler_enabled": scaler_enabled, "batch_size": batch_size,
            "num_workers": num_workers, "pin_memory": device.type == "cuda"}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


def make_autocast(env):
    if env["amp_dtype"] is None:
        import contextlib
        return contextlib.nullcontext
    dt, dev = env["amp_dtype"], env["device"].type
    return lambda: torch.amp.autocast(device_type=dev, dtype=dt, enabled=True)


def make_scaler(env):
    try:
        return torch.amp.GradScaler(env["device"].type, enabled=env["scaler_enabled"])
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=env["scaler_enabled"])


# ==============================================================================
# DATASET DISCOVERY
# ==============================================================================

def kagglehub_download(slug):
    try:
        import kagglehub
        p = kagglehub.dataset_download(slug)
        print(f"  kagglehub -> {p}")
        return Path(p)
    except Exception as e:
        print(f"  kagglehub failed for {slug} -> {repr(e)}")
        return None


def find_root(candidates, required_subdirs):
    """Return the shallowest directory containing all required subdirectories."""
    found = []
    for root in candidates:
        if root is None:
            continue
        root = Path(root)
        if not root.exists():
            continue
        if all((root / s).is_dir() for s in required_subdirs):
            found.append(root)
        for depth in range(1, 4):
            for p in root.glob("/".join(["*"] * depth)):
                if (p.is_dir() and "__MACOSX" not in str(p)
                        and all((p / s).is_dir() for s in required_subdirs)):
                    found.append(p)
    if not found:
        return None
    # Shallowest wins -- the chest archive nests a duplicate one level deeper.
    return min(found, key=lambda p: len(p.parts))


PERSON_RE = re.compile(r"person(\d+)", re.IGNORECASE)


def collect_chest(root):
    """CHEST images from chest_xray/train/{NORMAL,PNEUMONIA} only.

    The test/ folder is the pneumonia model's held-out set and is left alone.
    Group key: person<ID> where present, otherwise the accession with its
    trailing capture index stripped, so repeat images of one study stay together.
    """
    rows, seen = [], set()
    train_dir = Path(root) / "train"

    for cls in ("NORMAL", "PNEUMONIA"):
        d = train_dir / cls
        if not d.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = [x for x in dirnames
                           if x.lower() not in EXCLUDE_DIRS and not x.startswith(".")]
            for fn in sorted(filenames):
                if fn.startswith("._"):                       # AppleDouble sidecar
                    continue
                if Path(fn).suffix.lower() not in VALID_EXT:
                    continue
                if (cls, fn) in seen:                         # nested duplicate tree
                    continue
                seen.add((cls, fn))

                stem = Path(fn).stem
                m = PERSON_RE.search(stem)
                group = "C:P" + m.group(1) if m else "C:" + re.sub(r"-\d+$", "", stem)

                rows.append({
                    "image_path": os.path.join(dirpath, fn),
                    "label": CLASS_TO_INDEX["CHEST"],
                    "label_name": "CHEST",
                    "group_key": group,
                    "source": "chest-xray-pneumonia",
                    "subgroup": cls,            # disease label -- recorded, NEVER trained on
                })
    return pd.DataFrame(rows)


def collect_mura(root):
    """MUSCULOSKELETAL images from MURA-v1.1/train/XR_*/patientXXXXX/studyY_*/.

    MURA's official valid/ folder is the MURA model's held-out test set and is
    left alone. Group key: the MURA patient ID.
    """
    rows = []
    train_dir = Path(root) / "train"

    for bp_entry in sorted(os.scandir(train_dir), key=lambda e: e.name):
        if not bp_entry.is_dir() or not bp_entry.name.startswith("XR_"):
            continue
        body_part = bp_entry.name
        for pat in sorted(os.scandir(bp_entry.path), key=lambda e: e.name):
            if not pat.is_dir():
                continue
            for study in sorted(os.scandir(pat.path), key=lambda e: e.name):
                if not study.is_dir():
                    continue
                for f in sorted(os.scandir(study.path), key=lambda e: e.name):
                    if not f.is_file() or f.name.startswith("._"):
                        continue
                    if Path(f.name).suffix.lower() not in VALID_EXT:
                        continue
                    rows.append({
                        "image_path": f.path,
                        "label": CLASS_TO_INDEX["MUSCULOSKELETAL"],
                        "label_name": "MUSCULOSKELETAL",
                        "group_key": "M:" + pat.name,      # patient-level
                        "source": "mura-v1.1",
                        "subgroup": body_part,             # body part, for stratified capping
                    })
    return pd.DataFrame(rows)


def cap_by_group(df, max_n, seed=SEED, stratify_col="subgroup"):
    """Sample down to ~max_n images by dropping whole GROUPS, never individual
    images, so a patient is either fully in or fully out. Sampling is spread
    across subgroups (body parts / disease classes) so the router still sees the
    full variety of each class."""
    if len(df) <= max_n:
        return df

    rng = np.random.default_rng(seed)
    keep_groups = []
    subs = df[stratify_col].unique()
    quota = max_n // max(len(subs), 1)

    for s in subs:
        sub = df[df[stratify_col] == s]
        groups = sub.group_key.unique()
        rng.shuffle(groups)
        taken, n = [], 0
        sizes = sub.groupby("group_key").size().to_dict()
        for g in groups:
            if n >= quota:
                break
            taken.append(g)
            n += sizes[g]
        keep_groups.extend(taken)

    return df[df.group_key.isin(set(keep_groups))].reset_index(drop=True)


def build_router_dataset():
    print("\n" + "=" * 78)
    print("BUILDING ROUTER DATASET")
    print("=" * 78)
    print("Label = BODY REGION. Disease labels are recorded for reporting only and")
    print("are never used as a training target.\n")

    print("CHEST source:")
    chest_root = find_root(
        [CHEST_ROOT] if CHEST_ROOT else
        [kagglehub_download(CHEST_DATASET), Path("/kaggle/input/chest-xray-pneumonia"),
         Path("."), Path("./chest_xray")],
        ["train", "test"])
    if chest_root is None:
        raise FileNotFoundError(
            "Chest dataset not found. Set CHEST_ROOT at the top of this file to "
            "the folder containing train/ and test/.")
    print("  root:", chest_root)
    chest = collect_chest(chest_root)
    print(f"  collected {len(chest):,} chest images from train/ "
          f"({chest.group_key.nunique():,} groups)")
    print("  ", dict(chest.subgroup.value_counts()))

    print("\nMUSCULOSKELETAL source:")
    mura_root = find_root(
        [MURA_ROOT] if MURA_ROOT else
        [kagglehub_download(MURA_DATASET), Path("."), Path("./MURA-v1.1")],
        ["train", "valid"])
    if mura_root is None:
        raise FileNotFoundError(
            "MURA dataset not found. Set MURA_ROOT at the top of this file to "
            "the folder containing train/ and valid/.")
    print("  root:", mura_root)
    mura = collect_mura(mura_root)
    print(f"  collected {len(mura):,} MSK images from train/ "
          f"({mura.group_key.nunique():,} patients)")
    print("  ", dict(mura.subgroup.value_counts()))

    if len(chest) == 0 or len(mura) == 0:
        raise RuntimeError("One of the two router classes is empty.")

    print(f"\nCapping each class to ~{MAX_PER_CLASS:,} images (whole groups only):")
    chest_c = cap_by_group(chest, MAX_PER_CLASS)
    mura_c = cap_by_group(mura, MAX_PER_CLASS)
    print(f"  CHEST           {len(chest):,} -> {len(chest_c):,} "
          f"({chest_c.group_key.nunique():,} groups)")
    print(f"  MUSCULOSKELETAL {len(mura):,} -> {len(mura_c):,} "
          f"({mura_c.group_key.nunique():,} groups)")

    df = pd.concat([chest_c, mura_c], ignore_index=True)

    # A group key must never map to two different router classes.
    bad = df.groupby("group_key").label.nunique()
    assert (bad == 1).all(), "a group key spans both router classes"

    print(f"\nRouter dataset: {len(df):,} images, "
          f"{df.group_key.nunique():,} groups")
    print(df.label_name.value_counts().to_string())
    return df


# ==============================================================================
# SPLITTING
# ==============================================================================

def make_splits(df, seed=SEED):
    print("\n" + "=" * 78)
    print("SPLITTING (group level, stratified by class)")
    print("=" * 78)

    parts = {"train": [], "val": [], "test": []}

    # Split each class separately so both stay balanced in every split.
    for label in sorted(df.label.unique()):
        sub = df[df.label == label].reset_index(drop=True)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC + TEST_FRAC,
                                 random_state=seed)
        tr, rest = next(gss1.split(sub, groups=sub.group_key))
        rest_df = sub.iloc[rest].reset_index(drop=True)
        rel = TEST_FRAC / (VAL_FRAC + TEST_FRAC)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel, random_state=seed + 1)
        va, te = next(gss2.split(rest_df, groups=rest_df.group_key))

        parts["train"].append(sub.iloc[tr])
        parts["val"].append(rest_df.iloc[va])
        parts["test"].append(rest_df.iloc[te])

    splits = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}

    total = sum(len(d) for d in splits.values())
    table = []
    for name, d in splits.items():
        table.append({
            "split": name,
            "groups": d.group_key.nunique(),
            "images": len(d),
            "%": round(100 * len(d) / total, 2),
            "CHEST": int((d.label == 0).sum()),
            "MUSCULOSKELETAL": int((d.label == 1).sum()),
        })
    print(pd.DataFrame(table).set_index("split").to_string())

    print("\nDisease/body-part composition per split (proof the router sees both")
    print("disease classes of each region, and is not learning a disease label):")
    print(pd.crosstab(
        pd.concat([pd.Series(n, index=range(len(d))) for n, d in splits.items()],
                  ignore_index=True),
        pd.concat([d.subgroup for d in splits.values()], ignore_index=True)
    ).to_string())

    return splits


def verify_no_leakage(splits):
    print("\n" + "=" * 78)
    print("LEAKAGE CHECK")
    print("=" * 78)
    names = list(splits.keys())
    clean = True
    for key in ("group_key", "image_path"):
        print(f"\n--- {key} ---")
        sets = {n: set(splits[n][key]) for n in names}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                inter = sets[a] & sets[b]
                print(f"  {a:<5} vs {b:<5}: "
                      f"{'PASS (no overlap)' if not inter else f'FAIL ({len(inter)})'}")
                if inter:
                    clean = False
    print("\n" + ("RESULT: no patient/study or image spans two router splits."
                  if clean else "RESULT: LEAKAGE DETECTED."))
    print("\nAlso by construction: the router never saw chest_xray/test/ or")
    print("MURA-v1.1/valid/, which are the two downstream models' held-out sets.")
    if not clean:
        raise RuntimeError("Leakage detected -- fix the split before training.")


# ==============================================================================
# TRAINING
# ==============================================================================

def make_loader(df, transform, shuffle, env, drop_last=False):
    ds = RouterDataset(df, transform=transform)
    kwargs = dict(batch_size=env["batch_size"], shuffle=shuffle,
                  num_workers=env["num_workers"], pin_memory=env["pin_memory"],
                  drop_last=drop_last, worker_init_fn=router_seed_worker)
    if env["num_workers"] > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    if shuffle:
        g = torch.Generator()
        g.manual_seed(SEED)
        kwargs["generator"] = g
    return DataLoader(ds, **kwargs)


def run_epoch(model, loader, criterion, env, autocast_ctx,
              optimizer=None, scaler=None, desc="", eval_modules=None):
    training = optimizer is not None
    model.train(training)
    if training and eval_modules:
        for m in eval_modules:
            m.eval()

    device, pin = env["device"], env["pin_memory"]
    total_loss, n_seen = 0.0, 0
    all_probs, all_labels = [], []

    for step, (xb, yb, _) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=pin)
        yb = yb.to(device, non_blocking=pin)

        with torch.set_grad_enabled(training):
            with autocast_ctx():
                logits = model(xb)
                loss = criterion(logits, yb)

            if training:
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if GRAD_CLIP_NORM:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()

        lv = loss.item()
        if math.isfinite(lv):
            total_loss += lv * xb.size(0)
            n_seen += xb.size(0)
        all_probs.append(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())
        all_labels.append(yb.detach().cpu().numpy())

        if step % 20 == 0 or step == len(loader):
            print(f"\r  {desc} {step}/{len(loader)}  "
                  f"loss {total_loss / max(n_seen, 1):.4f}", end="", flush=True)
    print()

    return (total_loss / max(n_seen, 1),
            np.concatenate(all_probs), np.concatenate(all_labels))


def metrics_from(probs, labels):
    preds = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_msk": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_msk": recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }


def freeze_backbone(model, unfreeze_from=None):
    for p in model.backbone.features.parameters():
        p.requires_grad = False
    if unfreeze_from is not None:
        for block in model.backbone.features[unfreeze_from:]:
            for p in block.parameters():
                p.requires_grad = True
    for p in model.backbone.classifier.parameters():
        p.requires_grad = True
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  unfreeze_from={unfreeze_from} -> trainable {tr:,}/{tot:,} "
          f"({100 * tr / tot:.1f}%)")
    return model


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="Train the X-ray body-region router")
    ap.add_argument("--chest-root", default=CHEST_ROOT)
    ap.add_argument("--mura-root", default=MURA_ROOT)
    ap.add_argument("--max-per-class", type=int, default=MAX_PER_CLASS)
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS_STAGE2)
    args = ap.parse_args()

    globals()["CHEST_ROOT"] = args.chest_root
    globals()["MURA_ROOT"] = args.mura_root
    globals()["MAX_PER_CLASS"] = args.max_per_class

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = setup_environment()
    set_seed(SEED)
    device = env["device"]
    autocast_ctx = make_autocast(env)

    df = build_router_dataset()
    splits = make_splits(df)
    verify_no_leakage(splits)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    df.to_csv(OUTPUT_DIR / "router_metadata.csv", index=False)

    train_tf = build_router_transform(train=True)
    eval_tf = build_router_transform(train=False)
    print("\nTRAIN transform:\n", train_tf)
    print("\nEVAL transform (deterministic):\n", eval_tf)

    train_loader = make_loader(train_df, train_tf, True, env, drop_last=True)
    val_loader = make_loader(val_df, eval_tf, False, env)
    test_loader = make_loader(test_df, eval_tf, False, env)

    print(f"\ntrain {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

    set_seed(SEED)
    model = XRayRouter(pretrained=True).to(device)
    total = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 78)
    print("ROUTER MODEL")
    print("=" * 78)
    print(f"EfficientNet-B0 | pretrained {model.pretrained}")
    print(f"Total params    : {total:,}  ({total * 4 / 1024 ** 2:.1f} MB fp32)")
    print(f"Head            : Dropout -> Linear({model.in_features}, {NUM_CLASSES})")
    print("For scale, each downstream EfficientNet-B3 has ~10.7M params; the")
    print("router is deliberately a fraction of that.")

    criterion = nn.CrossEntropyLoss()
    history = []

    # ---- STAGE 1 ------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"STAGE 1 -- frozen backbone, head only | {NUM_EPOCHS_STAGE1} epochs")
    print("=" * 78)
    model = freeze_backbone(model, None)
    opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=LEARNING_RATE_S1, weight_decay=WEIGHT_DECAY)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=max(NUM_EPOCHS_STAGE1, 1), eta_min=LEARNING_RATE_S1 * 0.05)
    scaler = make_scaler(env)
    best = -np.inf

    for ep in range(1, NUM_EPOCHS_STAGE1 + 1):
        t0 = time.time()
        trl, trp, trY = run_epoch(model, train_loader, criterion, env, autocast_ctx,
                                  opt1, scaler, f"S1 train {ep}/{NUM_EPOCHS_STAGE1}",
                                  eval_modules=[model.backbone.features])
        vl, vp, vY = run_epoch(model, val_loader, criterion, env, autocast_ctx,
                               desc=f"S1 val   {ep}/{NUM_EPOCHS_STAGE1}")
        sched1.step()
        vm = metrics_from(vp, vY)
        print(f"[S1 {ep}] {time.time() - t0:.0f}s | train loss {trl:.4f} | "
              f"val loss {vl:.4f} | val acc {vm['accuracy']:.4f} | "
              f"f1 {vm['f1_macro']:.4f}")
        history.append({"stage": 1, "epoch": ep, "train_loss": trl,
                        "val_loss": vl, **{f"val_{k}": v for k, v in vm.items()}})
        if vm["accuracy"] > best:
            best = vm["accuracy"]
            save_router_checkpoint(model, CHECKPOINT_PATH,
                                   extra={"stage": 1, "epoch": ep, "val_metrics": vm})
            print(f"        -> new best (acc {best:.4f}); saved")

    # ---- STAGE 2 ------------------------------------------------------------
    ck = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])

    n_ep2 = args.epochs
    print("\n" + "=" * 78)
    print(f"STAGE 2 -- fine-tune features[{UNFREEZE_FROM}:] | max {n_ep2} epochs")
    print("=" * 78)
    model = freeze_backbone(model, UNFREEZE_FROM)
    bb = [p for p in model.backbone.features.parameters() if p.requires_grad]
    hd = [p for p in model.backbone.classifier.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(
        [{"params": bb, "lr": LEARNING_RATE_S2 * BACKBONE_LR_MULT},
         {"params": hd, "lr": LEARNING_RATE_S2}], weight_decay=WEIGHT_DECAY)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt2, mode="max", factor=0.3, patience=max(PATIENCE // 2, 1), min_lr=1e-7)
    scaler = make_scaler(env)
    best2, stale = best, 0

    for ep in range(1, n_ep2 + 1):
        t0 = time.time()
        trl, trp, trY = run_epoch(model, train_loader, criterion, env, autocast_ctx,
                                  opt2, scaler, f"S2 train {ep}/{n_ep2}")
        vl, vp, vY = run_epoch(model, val_loader, criterion, env, autocast_ctx,
                               desc=f"S2 val   {ep}/{n_ep2}")
        vm = metrics_from(vp, vY)
        sched2.step(vm["accuracy"])
        print(f"[S2 {ep}] {time.time() - t0:.0f}s | train loss {trl:.4f} | "
              f"val loss {vl:.4f} | val acc {vm['accuracy']:.4f} | "
              f"f1 {vm['f1_macro']:.4f} | mean conf {vm['mean_confidence']:.4f}")
        history.append({"stage": 2, "epoch": ep, "train_loss": trl,
                        "val_loss": vl, **{f"val_{k}": v for k, v in vm.items()}})

        if vm["accuracy"] > best2 + 1e-5:
            best2, stale = vm["accuracy"], 0
            save_router_checkpoint(model, CHECKPOINT_PATH,
                                   extra={"stage": 2, "epoch": ep, "val_metrics": vm})
            print(f"        -> new best (acc {best2:.4f}); saved")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"\nEarly stopping at epoch {ep}.")
                break
        if device.type == "cuda":
            torch.cuda.empty_cache()

    hist = pd.DataFrame(history)
    hist.to_csv(OUTPUT_DIR / "router_history.csv", index=False)
    plot_history(hist)

    # ---- TEST ---------------------------------------------------------------
    best_ck = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(best_ck["model_state_dict"])
    model.to(device).eval()
    print(f"\nLoaded best router: stage {best_ck.get('stage')} "
          f"epoch {best_ck.get('epoch')}")

    _, tp_, tY = run_epoch(model, test_loader, criterion, env, autocast_ctx,
                           desc="router test")
    preds = tp_.argmax(axis=1)
    tm = metrics_from(tp_, tY)
    cm = confusion_matrix(tY, preds, labels=[0, 1])

    print("\n" + "=" * 78)
    print(f"ROUTER TEST EVALUATION | n = {len(tY):,}")
    print("=" * 78)
    print(f"Accuracy        : {tm['accuracy']:.4f}")
    print(f"Macro F1        : {tm['f1_macro']:.4f}")
    print(f"Mean confidence : {tm['mean_confidence']:.4f}")
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(f"                     CHEST   MUSCULOSKELETAL")
    print(f"  CHEST           {cm[0, 0]:>8,} {cm[0, 1]:>17,}")
    print(f"  MUSCULOSKELETAL {cm[1, 0]:>8,} {cm[1, 1]:>17,}")
    print("\n" + classification_report(tY, preds,
                                       target_names=["CHEST", "MUSCULOSKELETAL"],
                                       digits=4))

    conf = tp_.max(axis=1)
    thr = 0.80
    print(f"Images below the {thr:.2f} routing threshold: "
          f"{(conf < thr).sum():,}/{len(conf):,} ({100 * (conf < thr).mean():.2f}%)")
    wrong = preds != tY
    if wrong.any():
        print(f"Misrouted images: {wrong.sum():,}; their mean confidence "
              f"{conf[wrong].mean():.4f}")
        print("If misroutes carry high confidence, raising ROUTER_THRESHOLD will")
        print("not catch them -- inspect those images directly.")
    else:
        print("No misroutes on the held-out split.")

    save_router_checkpoint(model, CHECKPOINT_PATH, extra={
        "stage": best_ck.get("stage"), "epoch": best_ck.get("epoch"),
        "val_metrics": best_ck.get("val_metrics"),
        "test_metrics": {k: float(v) for k, v in tm.items()},
        "confusion_matrix": cm.tolist(),
        "n_train_images": len(train_df), "n_val_images": len(val_df),
        "n_test_images": len(test_df),
        "chest_source": CHEST_DATASET, "msk_source": MURA_DATASET,
        "training_note": ("trained ONLY on the train/ portion of both datasets; "
                          "chest test/ and MURA valid/ (the downstream models' "
                          "held-out sets) were never seen"),
        "seed": SEED,
    })
    print(f"\nRouter checkpoint -> {CHECKPOINT_PATH.resolve()}")

    with open(OUTPUT_DIR / "router_test_metrics.json", "w") as f:
        json.dump({"test_metrics": {k: float(v) for k, v in tm.items()},
                   "confusion_matrix": cm.tolist()}, f, indent=2)

    print("\n" + "=" * 78)
    print("DONE. Next:  python pipeline.py \"path/to/xray.jpg\"")
    print("=" * 78)
    print("Note: the router has exactly two classes. It has never seen a CT slice,")
    print("an MRI, an abdominal film or a photograph, and softmax will still")
    print("produce a confident-looking answer for any of them. ROUTER_THRESHOLD")
    print("catches ambiguity between chest and limb, not out-of-distribution input.")


def plot_history(h):
    if h.empty:
        return
    x = np.arange(1, len(h) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(x, h.train_loss, "-o", ms=4, label="train")
    ax[0].plot(x, h.val_loss, "-o", ms=4, label="validation")
    ax[0].set_title("Router loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(x, h.val_accuracy, "-o", ms=4, label="val accuracy")
    ax[1].plot(x, h.val_f1_macro, "-s", ms=4, label="val macro F1")
    ax[1].set_title("Router validation")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    s2 = h.index[h.stage == 2].min() + 1 if (h.stage == 2).any() else None
    if s2:
        for a in ax:
            a.axvline(s2 - 0.5, color="gray", ls="--", lw=1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "router_history.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
