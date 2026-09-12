#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pneumonia.py
==================
Paediatric Chest X-ray Pneumonia Detection -- EfficientNet-B3 training pipeline.

Dataset : paultimothymooney/chest-xray-pneumonia
Task    : binary classification, 0 = NORMAL, 1 = PNEUMONIA
Hardware: tuned for an 8 GB NVIDIA RTX 4060 (Laptop/Desktop). Falls back cleanly to
          any other CUDA GPU or to CPU.

WHAT THIS SCRIPT FIXES (from the dataset analysis of Untitled0 (2).ipynb)
------------------------------------------------------------------------
1. ARCHIVE CONTAMINATION. The Kaggle archive unpacks into three parallel trees:
   the real data, a byte-identical nested copy (chest_xray/chest_xray/), and a
   __MACOSX/ tree of AppleDouble sidecars ("._name.jpeg") that carry a .jpeg
   extension but hold resource-fork metadata, not pixels. A naive os.walk counts
   all three -> 17,568 "images" where 5,856 exist. Here they are excluded, and
   the script ASSERTS the resulting count is sane before training.

2. THE 16-IMAGE VALIDATION SET. The shipped val/ folder holds 8 images per class
   (0.27% of the data). One misclassification moves validation accuracy by 6.25
   points, which makes early stopping, checkpoint selection and threshold tuning
   statistically meaningless. This script discards that split and re-partitions
   the pooled train+val data at the PATIENT/STUDY level, keeping the official
   test/ folder untouched as the held-out set.

3. UNASSESSED PATIENT-LEVEL LEAKAGE. PNEUMONIA filenames encode a patient ID
   ("person1946_bacteria_4875.jpeg"); NORMAL filenames encode an accession that
   groups repeat images ("NORMAL2-IM-0246-0001-0001/-0002"). Both are extracted
   and used as grouping keys so no patient/study spans two splits. An explicit
   leakage assertion halts the run if any key crosses a boundary.
   LIMITATION, stated plainly: NORMAL filenames do not expose a true patient ID,
   so grouping for that class is at the accession level, not the patient level.
   A fully patient-disjoint split is NOT achievable with the metadata this
   dataset ships. Do not claim otherwise in a write-up.

4. EXACT-DUPLICATE REMOVAL. The dataset contains genuine MD5-identical images
   under different filenames (person128_bacteria_606 / _607). These are removed,
   and any duplicate pair that spanned two splits is reported as leakage.

OTHER DESIGN CHOICES
--------------------
* bfloat16 autocast (not float16). float16 tops out at 65504; EfficientNet's wide
  expanded channels through SiLU can overflow to inf, and that inf is written into
  BatchNorm's running_mean/running_var during the forward pass where GradScaler
  cannot see it. Training then looks healthy while every eval-mode pass returns
  NaN. Ada GPUs support bfloat16, which has float32's exponent range.
* NO horizontal flip. Unlike limb radiography, the thorax is not left-right
  symmetric (heart left, liver right, 2 lobes left / 3 right). A flipped chest
  film is situs inversus, and flipping destroys the laterality of any finding.
* Letterbox resize (pad to square, then scale) rather than a squashing resize.
  Aspect ratios in this dataset run 0.835-3.379; forcing them to 1:1 distorts the
  cardiothoracic ratio differently for every image.
* Decision threshold selected on validation, locked, then applied to test.

Run:
    python train_pneumonia.py
"""

import os

# Must be set before torch initialises CUDA. Reduces fragmentation on 8 GB cards.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import sys
import json
import time
import math
import random
import hashlib
import argparse
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless-safe; figures are saved, not shown
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.models import efficientnet_b3

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SEED = 42

# Leave DATA_ROOT as None to download via kagglehub, or point it at an already
# extracted copy, e.g. r"C:\Users\vemul\Downloads\chest_xray".
DATA_ROOT = None
KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"

IMAGE_SIZE = 300                 # EfficientNet-B3's native resolution
USE_LETTERBOX = True             # pad to square before resizing (preserves anatomy)

# ---- splitting --------------------------------------------------------------
VAL_FRACTION = 0.15              # of the pooled train+val GROUPS
DEDUPE_BY_HASH = True            # remove MD5-identical images (slower start, correct)

# ---- training ---------------------------------------------------------------
NUM_EPOCHS_STAGE1 = 3            # frozen backbone, head only
NUM_EPOCHS_STAGE2 = 12           # partial unfreeze
LEARNING_RATE_S1 = 1e-3
LEARNING_RATE_S2 = 1e-4
BACKBONE_LR_MULT = 0.25
WEIGHT_DECAY = 1e-4
DROPOUT_P = 0.4
UNFREEZE_FROM = 5                # unfreeze backbone.features[5:] in stage 2
PATIENCE = 4
GRAD_CLIP_NORM = 5.0
MONITOR_METRIC = "roc_auc"       # "roc_auc" | "f1" | "loss"

# ---- threshold --------------------------------------------------------------
# "f1"        -> maximise F1 on validation
# "youden"    -> maximise sensitivity + specificity - 1
# "recall_at" -> highest threshold still reaching TARGET_RECALL (screening bias)
THRESHOLD_CRITERION = "recall_at"
TARGET_RECALL = 0.95             # missing paediatric pneumonia is the costly error

# ---- output -----------------------------------------------------------------
OUTPUT_DIR = Path("./pneumonia_outputs")
CHECKPOINT_DIR = Path("./pneumonia_checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "pneumonia_efficientnet_b3_best.pth"

CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_NAME = "efficientnet_b3"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXCLUDE_DIRS = {"__macosx"}


# ==============================================================================
# MODULE-LEVEL DEFINITIONS (required for Windows "spawn" DataLoader workers)
# ==============================================================================
# On Windows, DataLoader worker processes pickle the Dataset class and
# worker_init_fn by import path. Anything defined inside a function is a "local
# object" and cannot be pickled, raising:
#   _pickle.PicklingError: Can't pickle local object ...
# So these live at true module scope, and everything else runs under
# `if __name__ == "__main__":` at the bottom.

class LetterboxResize:
    """Pad to square with a constant, then resize. Preserves aspect ratio and the
    full field of view -- no anatomy is squashed and none is cropped away.

    Black padding reads as "outside the collimated field", which is anatomically
    plausible on a radiograph, unlike a stretched thorax.
    """

    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        side = max(w, h)
        canvas = Image.new(img.mode, (side, side), self.fill)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas.resize((self.size, self.size), Image.BILINEAR)

    def __repr__(self):
        return f"LetterboxResize(size={self.size}, fill={self.fill})"


class ChestXrayDataset(Dataset):
    """Chest radiographs -> (3xHxW float tensor, float label, index).

    Grayscale is forced with convert("L") -- the dataset mixes 'L' and 'RGB'
    files, and without this the collate function receives inconsistent channel
    counts -- then replicated to 3 channels for the ImageNet-pretrained stem.

    Replication adds no information. It satisfies an interface: the stem computes
    sum_c W_c * x_c, and with x_0 = x_1 = x_2 = x that equals (sum_c W_c) * x,
    i.e. exactly the response of a channel-summed 1-channel kernel. The pretrained
    edge and texture filters transfer intact with no surgery on the checkpoint.
    """

    def __init__(self, df, transform=None):
        self.paths = df["image_path"].values
        self.labels = df["label"].values.astype(np.float32)
        self.transform = transform
        self._warned = 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            with Image.open(path) as im:
                img = im.convert("L").copy()
        except Exception as e:
            if self._warned < 5:
                print(f"[dataset] unreadable: {path} ({e}); using blank placeholder")
                self._warned += 1
            img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)

        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32), idx


class InferenceDataset(Dataset):
    """Paths only, for batch prediction."""

    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                img = im.convert("L").convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        return self.transform(img), i


class PneumoniaEfficientNetB3(nn.Module):
    """EfficientNet-B3 with a single-logit binary head.

    NO sigmoid inside the model. BCEWithLogitsLoss fuses the sigmoid with the log
    using the log-sum-exp trick for numerical stability; adding a separate sigmoid
    applies it twice, flattening gradients. Sigmoid is applied explicitly at
    inference only.
    """

    def __init__(self, pretrained=True, dropout_p=DROPOUT_P):
        super().__init__()
        weights = None
        if pretrained:
            try:
                from torchvision.models import EfficientNet_B3_Weights
                weights = EfficientNet_B3_Weights.IMAGENET1K_V1
                print("Loading ImageNet-pretrained EfficientNet-B3 weights ...")
            except Exception as e:
                print("Could not resolve pretrained weights ->", repr(e))

        try:
            self.backbone = efficientnet_b3(weights=weights)
            self.pretrained = weights is not None
        except Exception as e:
            print("!! Pretrained download failed ->", repr(e))
            print("!! Falling back to RANDOM init. Results will be much weaker.")
            self.backbone = efficientnet_b3(weights=None)
            self.pretrained = False

        in_features = self.backbone.classifier[1].in_features   # 1536
        self.in_features = in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(in_features, 1),
        )
        nn.init.zeros_(self.backbone.classifier[1].bias)
        nn.init.normal_(self.backbone.classifier[1].weight, std=0.01)

    def forward(self, x):
        return self.backbone(x).squeeze(1)      # (B,) raw logits

    @property
    def gradcam_target_layer(self):
        # Deepest layer that still has spatial axes: 1536 channels, 10x10 at 300px.
        return self.backbone.features[-1]


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


# ==============================================================================
# 1. ENVIRONMENT
# ==============================================================================

def setup_environment():
    print("=" * 78)
    print("ENVIRONMENT")
    print("=" * 78)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("PyTorch        :", torch.__version__)
    print("CUDA available :", torch.cuda.is_available())

    total_gb = 0.0
    gpu_name = "N/A (CPU)"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print("CUDA version   :", torch.version.cuda)
        print("GPU            :", gpu_name)
        print("GPU memory     : {:.2f} GB".format(total_gb))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Device         :", device)

    # ---- AMP dtype ----------------------------------------------------------
    amp_dtype = None
    if device.type == "cuda":
        try:
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            amp_dtype = torch.float16

    scaler_enabled = amp_dtype == torch.float16
    dtype_name = "disabled" if amp_dtype is None else str(amp_dtype).replace("torch.", "")
    print("AMP dtype      :", dtype_name, "| GradScaler:", scaler_enabled)
    if amp_dtype == torch.bfloat16:
        print("  bfloat16 has float32's exponent range -> the float16 overflow that")
        print("  poisons BatchNorm running stats with inf cannot occur.")

    # ---- batch size / accumulation for the detected card --------------------
    if device.type != "cuda":
        batch_size, accum = 4, 1
    elif total_gb >= 20:
        batch_size, accum = 48, 1
    elif total_gb >= 14:
        batch_size, accum = 32, 1
    elif total_gb >= 10:
        batch_size, accum = 24, 1
    else:
        # 8 GB RTX 4060 / 4060 Ti / 4060 Laptop lands here. Physical batch 16 with
        # 2 accumulation steps gives an effective batch of 32 at half the peak
        # activation memory.
        batch_size, accum = 16, 2

    num_workers = min(4, (os.cpu_count() or 2)) if device.type == "cuda" else 0

    print(f"Batch size     : {batch_size} (grad accum {accum} -> effective {batch_size * accum})")
    print(f"Num workers    : {num_workers}")

    return {
        "device": device, "gpu_name": gpu_name, "total_gb": total_gb,
        "amp_dtype": amp_dtype, "scaler_enabled": scaler_enabled,
        "batch_size": batch_size, "accum": accum,
        "num_workers": num_workers, "pin_memory": device.type == "cuda",
    }


def set_seed(seed=SEED, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


# ==============================================================================
# 2. DATASET DISCOVERY AND CLEANING
# ==============================================================================

def resolve_dataset_root():
    """Find the directory that directly contains train/, val/ and test/."""
    candidates = []

    if DATA_ROOT:
        candidates.append(Path(DATA_ROOT))
    else:
        try:
            import kagglehub
            p = kagglehub.dataset_download(KAGGLE_DATASET)
            print("kagglehub downloaded to:", p)
            candidates.append(Path(p))
        except Exception as e:
            print("kagglehub unavailable ->", repr(e))
        candidates += [Path("/kaggle/input/chest-xray-pneumonia"), Path("."),
                       Path("./chest_xray"), Path("/content")]

    def has_splits(d):
        return (d / "train").is_dir() and (d / "test").is_dir()

    found = []
    for root in candidates:
        if not root.exists():
            continue
        if has_splits(root):
            found.append(root)
        for depth in range(1, 4):
            for p in root.glob("/".join(["*"] * depth)):
                if p.is_dir() and has_splits(p) and "__MACOSX" not in str(p):
                    found.append(p)

    if not found:
        raise FileNotFoundError(
            "Could not locate the chest X-ray dataset (a folder containing "
            "train/ and test/).\nSet DATA_ROOT at the top of this file."
        )

    # The archive nests a duplicate at chest_xray/chest_xray/. Prefer the
    # SHALLOWEST match so we walk one tree, not two.
    root = min(found, key=lambda p: len(p.parts))
    print("Dataset root   :", root.resolve())
    return root


def md5_of(path, chunk=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


PERSON_RE = re.compile(r"person(\d+)", re.IGNORECASE)


def group_key_for(filename, label_name):
    """Derive a grouping key so images of one patient/study stay in one split.

    PNEUMONIA: filenames embed a real patient ID, e.g.
        person1946_bacteria_4875.jpeg -> "P1946"
        person1946_bacteria_4874.jpeg -> "P1946"   (same patient, 2 images)

    NORMAL: filenames embed an accession, not a patient ID, e.g.
        NORMAL2-IM-0246-0001-0001.jpeg -> "N:NORMAL2-IM-0246-0001"
        NORMAL2-IM-0246-0001-0002.jpeg -> "N:NORMAL2-IM-0246-0001"
        IM-0162-0001.jpeg              -> "N:IM-0162"
    Stripping the trailing numeric segment groups repeat captures of one study.

    LIMITATION: this is accession-level, not patient-level, for NORMAL. Two
    studies of the same healthy child cannot be linked from the filename alone.
    """
    stem = Path(filename).stem
    if label_name == "PNEUMONIA":
        m = PERSON_RE.search(stem)
        if m:
            return "P" + m.group(1)
    return "N:" + re.sub(r"-\d+$", "", stem)


def build_metadata(root, dedupe=DEDUPE_BY_HASH):
    """Walk the tree, excluding archive contamination, and build a dataframe."""
    print("\n" + "=" * 78)
    print("DATASET DISCOVERY AND CLEANING")
    print("=" * 78)

    rows = []
    stats = Counter()
    seen_canonical = {}          # (split, class, filename) -> path

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune junk directories IN PLACE so os.walk never descends into them.
        pruned = [d for d in dirnames
                  if d.lower() in EXCLUDE_DIRS or d.startswith(".")]
        stats["dirs_pruned"] += len(pruned)
        dirnames[:] = [d for d in dirnames if d not in pruned]

        lower = dirpath.replace("\\", "/").lower()

        if "/train/" in lower + "/" or lower.endswith("/train"):
            split = "train"
        elif "/val/" in lower + "/" or lower.endswith("/val"):
            split = "val"
        elif "/test/" in lower + "/" or lower.endswith("/test"):
            split = "test"
        else:
            continue

        if "/normal" in lower:
            label_name, label = "NORMAL", 0
        elif "/pneumonia" in lower:
            label_name, label = "PNEUMONIA", 1
        else:
            continue

        for fn in sorted(filenames):
            stats["files_seen"] += 1

            # AppleDouble sidecar: .jpeg extension, resource-fork content.
            if fn.startswith("._"):
                stats["appledouble_skipped"] += 1
                continue
            if Path(fn).suffix.lower() not in VALID_EXT:
                stats["non_image_skipped"] += 1
                continue

            # Collapse the nested duplicate tree: the same (split, class, name)
            # under a second root is the same radiograph.
            canonical = (split, label_name, fn)
            full = os.path.join(dirpath, fn)
            if canonical in seen_canonical:
                stats["nested_copy_skipped"] += 1
                continue
            seen_canonical[canonical] = full

            rows.append({
                "image_path": full,
                "filename": fn,
                "orig_split": split,
                "label_name": label_name,
                "label": label,
                "group_key": group_key_for(fn, label_name),
            })

    df = pd.DataFrame(rows)

    print(f"Files encountered        : {stats['files_seen']:,}")
    print(f"  __MACOSX dirs pruned   : {stats['dirs_pruned']}")
    print(f"  AppleDouble '._' files : {stats['appledouble_skipped']:,}  "
          f"(these are NOT corrupt radiographs -- they are not images at all)")
    print(f"  non-image extensions   : {stats['non_image_skipped']}")
    print(f"  nested duplicate copies: {stats['nested_copy_skipped']:,}")
    print(f"Unique radiographs kept  : {len(df):,}")

    if len(df) == 0:
        raise RuntimeError("No images found -- check the dataset root.")

    # ---- exact-duplicate removal -------------------------------------------
    if dedupe:
        print("\nHashing files for exact-duplicate detection ...")
        t0 = time.time()
        seen_hash, keep, dropped, cross_split = {}, [], 0, []
        for i, r in df.iterrows():
            try:
                h = md5_of(r.image_path)
            except Exception:
                keep.append(i)
                continue
            if h in seen_hash:
                j = seen_hash[h]
                dropped += 1
                if df.at[j, "orig_split"] != r.orig_split:
                    cross_split.append((df.at[j, "image_path"], r.image_path))
            else:
                seen_hash[h] = i
                keep.append(i)
        print(f"  hashed {len(df):,} files in {time.time() - t0:.1f}s")
        print(f"  exact duplicates removed : {dropped:,}")
        if cross_split:
            print(f"  !! {len(cross_split)} duplicate pair(s) SPANNED TWO SPLITS "
                  f"-- that is image-level leakage in the source data.")
            for a, b in cross_split[:3]:
                print("     ", os.path.basename(a), "<->", os.path.basename(b))
        df = df.loc[keep].reset_index(drop=True)
        print(f"  images remaining         : {len(df):,}")

    # ---- sanity assertion ---------------------------------------------------
    if len(df) > 8000:
        raise RuntimeError(
            f"{len(df):,} images found. The real dataset has ~5,856. The archive "
            "contamination filter did not work -- inspect the paths before training."
        )

    print("\nClass distribution:")
    for name, n in df.label_name.value_counts().items():
        print(f"  {name:<10}: {n:,} ({100 * n / len(df):.2f}%)")
    print("\nOfficial split sizes:")
    print(df.orig_split.value_counts().to_string())
    print(f"\nUnique group keys        : {df.group_key.nunique():,}")
    multi = (df.groupby('group_key').size() > 1).sum()
    print(f"Groups with >1 image     : {multi:,}  "
          f"(these would leak under a random image-level split)")

    return df


# ==============================================================================
# 3. SPLITTING AND LEAKAGE VERIFICATION
# ==============================================================================

def make_splits(df, val_fraction=VAL_FRACTION, seed=SEED):
    print("\n" + "=" * 78)
    print("SPLITTING (group-level)")
    print("=" * 78)
    print("The shipped val/ folder holds 8 images per class. That is too small to")
    print("select a checkpoint or a threshold from, so it is pooled back into the")
    print("training data and a proper validation set is carved out by group.")
    print("The official test/ folder is left untouched as the held-out set.\n")

    test_df = df[df.orig_split == "test"].reset_index(drop=True)
    pool = df[df.orig_split.isin(["train", "val"])].reset_index(drop=True)

    # A group present in the official test set must not appear in the pool.
    overlap = set(pool.group_key) & set(test_df.group_key)
    if overlap:
        print(f"[warn] {len(overlap)} group key(s) appear in both the pool and the "
              f"official test set; removing them from the POOL to keep test clean.")
        pool = pool[~pool.group_key.isin(overlap)].reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_idx, va_idx = next(gss.split(pool, groups=pool.group_key))
    train_df = pool.iloc[tr_idx].reset_index(drop=True)
    val_df = pool.iloc[va_idx].reset_index(drop=True)

    splits = {"train": train_df, "val": val_df, "test": test_df}

    total = sum(len(d) for d in splits.values())
    table = []
    for name, d in splits.items():
        table.append({
            "split": name,
            "groups": d.group_key.nunique(),
            "images": len(d),
            "% of all": round(100 * len(d) / total, 2),
            "NORMAL": int((d.label == 0).sum()),
            "PNEUMONIA": int((d.label == 1).sum()),
            "% pneumonia": round(100 * d.label.mean(), 2),
        })
    print(pd.DataFrame(table).set_index("split").to_string())
    return splits


def verify_no_leakage(splits):
    print("\n" + "=" * 78)
    print("LEAKAGE CHECK")
    print("=" * 78)
    names = list(splits.keys())
    clean = True

    for key in ("group_key", "image_path", "filename"):
        print(f"\n--- {key} ---")
        sets = {n: set(splits[n][key]) for n in names}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                inter = sets[a] & sets[b]
                print(f"  {a:<5} vs {b:<5}: "
                      f"{'PASS (no overlap)' if not inter else f'FAIL ({len(inter)} shared)'}")
                if inter:
                    clean = False
                    print("     examples:", list(inter)[:5])

    print("\n" + ("RESULT: no group, path or filename spans two splits."
                  if clean else "RESULT: LEAKAGE DETECTED."))
    print("\nHONEST CAVEAT: grouping is patient-level for PNEUMONIA (person<ID> is a")
    print("real patient ID) but only accession-level for NORMAL, whose filenames do")
    print("not expose a patient ID. Two studies of the same healthy child cannot be")
    print("linked from this dataset's metadata. State this limitation in the report.")

    if not clean:
        raise RuntimeError("Leakage detected -- fix the split before training.")
    return clean


# ==============================================================================
# 4. TRANSFORMS AND LOADERS
# ==============================================================================

def build_transforms():
    resize = (LetterboxResize(IMAGE_SIZE) if USE_LETTERBOX
              else transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)))

    train_tf = transforms.Compose([
        resize,
        transforms.RandomAffine(
            degrees=7,                       # positioning varies between technicians
            translate=(0.05, 0.05),          # centring varies
            scale=(0.95, 1.05),              # source-to-detector distance varies
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        # NO RandomHorizontalFlip. The thorax is not left-right symmetric.
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        resize,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print("\nTRAIN transform:\n", train_tf)
    print("\nEVAL transform (no randomness):\n", eval_tf)
    return train_tf, eval_tf


def make_loader(df, transform, shuffle, env, drop_last=False):
    ds = ChestXrayDataset(df, transform=transform)
    kwargs = dict(
        batch_size=env["batch_size"],
        shuffle=shuffle,
        num_workers=env["num_workers"],
        pin_memory=env["pin_memory"],
        drop_last=drop_last,
        worker_init_fn=seed_worker,
    )
    if env["num_workers"] > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    if shuffle:
        g = torch.Generator()
        g.manual_seed(SEED)
        kwargs["generator"] = g
    return DataLoader(ds, **kwargs)


# ==============================================================================
# 5. METRICS AND TRAINING UTILITIES
# ==============================================================================

def sanitize_probs(y_prob, context=""):
    y_prob = np.asarray(y_prob, dtype=float)
    bad = ~np.isfinite(y_prob)
    if bad.any():
        print(f"  [warn] {bad.sum():,}/{len(y_prob):,} non-finite predictions"
              f"{' in ' + context if context else ''}; substituting 0.5. "
              f"This signals numerical instability upstream.")
        y_prob = np.where(bad, 0.5, y_prob)
    return y_prob


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = sanitize_probs(y_prob, "compute_metrics")
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def fmt_metrics(m):
    return (f"acc {m['accuracy']:.4f} | prec {m['precision']:.4f} | "
            f"rec {m['recall']:.4f} | spec {m['specificity']:.4f} | "
            f"f1 {m['f1']:.4f} | auc {m['roc_auc']:.4f} | pr-auc {m['pr_auc']:.4f}")


def model_is_finite(model):
    """Check parameters AND buffers. BatchNorm running stats are updated in the
    forward pass, outside the autograd graph, so GradScaler never inspects them --
    they are exactly where an inf hides."""
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return False
    for b in model.buffers():
        if b.is_floating_point() and not torch.isfinite(b).all():
            return False
    return True


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


def run_epoch(model, loader, criterion, env, autocast_ctx,
              optimizer=None, scaler=None, desc="", eval_modules=None):
    training = optimizer is not None
    model.train(training)
    if training and eval_modules:
        for m in eval_modules:
            m.eval()          # keep frozen BatchNorm statistics fixed

    device, pin = env["device"], env["pin_memory"]
    accum = env["accum"] if training else 1
    n_batches = len(loader)

    total_loss, n_seen, n_bad = 0.0, 0, 0
    probs, labels = [], []

    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, (xb, yb, _) in enumerate(loader, start=1):
        xb = xb.to(device, non_blocking=pin)
        yb = yb.to(device, non_blocking=pin)

        with torch.set_grad_enabled(training):
            with autocast_ctx():
                logits = model(xb)
                loss = criterion(logits, yb)

            if training:
                if not torch.isfinite(loss):
                    n_bad += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss / accum).backward()
                if step % accum == 0 or step == n_batches:
                    if GRAD_CLIP_NORM:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        lv = loss.item()
        if math.isfinite(lv):
            total_loss += lv * xb.size(0)
            n_seen += xb.size(0)
        probs.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
        labels.append(yb.detach().cpu().numpy())

        if step % 50 == 0 or step == n_batches:
            print(f"\r  {desc} {step}/{n_batches}  loss {total_loss / max(n_seen, 1):.4f}",
                  end="", flush=True)

    print()
    if n_bad:
        print(f"  [warn] {n_bad} non-finite micro-batch(es) skipped.")

    return (total_loss / max(n_seen, 1),
            np.concatenate(probs), np.concatenate(labels))


def freeze_backbone(model, unfreeze_from=None):
    for p in model.backbone.features.parameters():
        p.requires_grad = False
    if unfreeze_from is not None:
        for block in model.backbone.features[unfreeze_from:]:
            for p in block.parameters():
                p.requires_grad = True
    for p in model.backbone.classifier.parameters():
        p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  unfreeze_from={unfreeze_from} -> trainable {train:,}/{total:,} "
          f"({100 * train / total:.1f}%)")
    return model


def save_checkpoint(model, path, extra=None):
    """Self-sufficient checkpoint: someone with only this .pth must be able to run
    correct inference without reading this file. That is why the class mapping,
    input size, normalisation and threshold all live inside it -- re-declaring any
    of them in a separate inference script is a silent failure waiting to happen."""
    payload = {
        "model_state_dict": model.state_dict(),
        "model_name": MODEL_NAME,
        "architecture": "torchvision.models.efficientnet_b3",
        "wrapper_class": "PneumoniaEfficientNetB3 (backbone. prefix on all keys)",
        "num_outputs": 1,
        "class_names": CLASS_NAMES,
        "class_to_index": {v: k for k, v in CLASS_NAMES.items()},
        "image_size": IMAGE_SIZE,
        "letterbox": USE_LETTERBOX,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "grayscale_handling": "PIL convert('L') then replicate to 3 channels",
        "dropout_p": DROPOUT_P,
        "loss": "BCEWithLogitsLoss",
        "activation": "sigmoid at inference only",
        "gradcam_target_layer": "backbone.features[-1]",
        "seed": SEED,
        "torch_version": torch.__version__,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": KAGGLE_DATASET,
        "task": "binary: 0=NORMAL, 1=PNEUMONIA (paediatric chest X-ray)",
        "disclaimer": ("Research/educational model. Not a clinical diagnosis. "
                       "Trained on paediatric (~1-5y) single-centre data; does not "
                       "generalise to adults or to other institutions."),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# ==============================================================================
# 6. THRESHOLD SELECTION
# ==============================================================================

def select_threshold(y_true, y_prob, criterion=THRESHOLD_CRITERION,
                     target_recall=TARGET_RECALL):
    grid = np.linspace(0.01, 0.99, 197)
    rows = []
    for t in grid:
        m = compute_metrics(y_true, y_prob, t)
        rows.append({"threshold": t, "f1": m["f1"], "precision": m["precision"],
                     "recall": m["recall"], "specificity": m["specificity"],
                     "youden": m["recall"] + m["specificity"] - 1})
    sweep = pd.DataFrame(rows)

    if criterion == "f1":
        t = float(sweep.loc[sweep.f1.idxmax(), "threshold"])
        why = "maximises F1 on the validation set"
    elif criterion == "youden":
        t = float(sweep.loc[sweep.youden.idxmax(), "threshold"])
        why = "maximises Youden's J = sensitivity + specificity - 1"
    elif criterion == "recall_at":
        ok = sweep[sweep.recall >= target_recall]
        t = float(ok.threshold.max()) if len(ok) else float(sweep.threshold.min())
        why = (f"highest threshold still reaching recall >= {target_recall:.0%}. "
               f"A missed pneumonia in a 1-5 year old is far costlier than a "
               f"false alarm, so the operating point favours sensitivity")
    else:
        raise ValueError(criterion)
    return t, why, sweep


# ==============================================================================
# 7. GRAD-CAM
# ==============================================================================

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.layer = target_layer or model.gradcam_target_layer
        self.acts = self.grads = None
        self.handles = []

    def __enter__(self):
        self.handles.append(self.layer.register_forward_hook(
            lambda m, i, o: setattr(self, "acts", o.detach())))
        self.handles.append(self.layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grads", go[0].detach())))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False

    def generate(self, x, device, signed=1.0, output_size=None):
        """Runs in float32 with autocast OFF -- half-precision gradients through
        the hook can underflow to zero and produce a blank map."""
        self.model.eval()
        x = x.to(device).float().requires_grad_(True)

        with torch.enable_grad():
            logit = self.model(x)
            self.model.zero_grad(set_to_none=True)
            (signed * logit).sum().backward()

        if self.acts is None or self.grads is None:
            raise RuntimeError("Grad-CAM hooks captured nothing.")

        a, g = self.acts[0], self.grads[0]
        alpha = g.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((alpha * a).sum(dim=0))
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else torch.zeros_like(cam)

        size = output_size or (x.shape[-2], x.shape[-1])
        cam = F.interpolate(cam[None, None], size=size,
                            mode="bilinear", align_corners=False)[0, 0]
        return cam.detach().cpu().numpy(), float(logit.detach().item())


# ==============================================================================
# 8. INFERENCE
# ==============================================================================

def predict_xray(image_path, model, env, threshold, eval_tf,
                 save_dir=None, show_gradcam=True, verbose=True):
    """Two-panel output: original X-ray | original + Grad-CAM overlay."""
    device = env["device"]

    with Image.open(image_path) as im:
        gray = im.convert("L").copy()
    gray_np = np.asarray(gray, dtype=np.float32) / 255.0

    x = eval_tf(gray.convert("RGB")).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logit = model(x.to(device)).float()
        p_pneu = torch.sigmoid(logit).item()
    p_norm = 1.0 - p_pneu

    idx = int(p_pneu >= threshold)
    pred = CLASS_NAMES[idx]
    conf = p_pneu if idx == 1 else p_norm

    result = {
        "image_path": os.path.abspath(str(image_path)),
        "prediction": pred,
        "normal_probability": round(p_norm * 100, 2),
        "pneumonia_probability": round(p_pneu * 100, 2),
        "confidence": round(conf * 100, 2),
        "logit": round(float(logit.item()), 6),
        "threshold": round(float(threshold), 6),
        "model_name": MODEL_NAME,
        "disclaimer": ("Research/educational model output. Not a clinical "
                       "diagnosis. Grad-CAM shows model-attributed importance, "
                       "not lesion segmentation."),
    }

    if verbose:
        print(f"\n  {os.path.basename(str(image_path))}")
        print(f"    Prediction : {pred}")
        print(f"    NORMAL     : {result['normal_probability']:.2f}%")
        print(f"    PNEUMONIA  : {result['pneumonia_probability']:.2f}%")
        print(f"    Confidence : {result['confidence']:.2f}%")

    if show_gradcam:
        signed = 1.0 if idx == 1 else -1.0
        with GradCAM(model) as cam_engine:
            cam, _ = cam_engine.generate(x, device, signed,
                                         output_size=gray_np.shape)

        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
        axes[0].imshow(gray_np, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original X-ray", fontsize=19, pad=12)
        axes[0].axis("off")

        axes[1].imshow(gray_np, cmap="gray", vmin=0, vmax=1)
        heat = axes[1].imshow(cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM Overlay", fontsize=19, pad=12)
        axes[1].axis("off")
        cb = fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
        cb.set_label("Model-attributed importance", fontsize=12)

        colour = "#C1445A" if idx == 1 else "#2E7D4F"
        fig.suptitle(f"Model Prediction: {pred}    |    Confidence: {conf * 100:.2f}%",
                     fontsize=21, fontweight="bold", y=0.97, color=colour)
        fig.text(0.5, 0.05,
                 f"NORMAL: {result['normal_probability']:.2f}%   |   "
                 f"PNEUMONIA: {result['pneumonia_probability']:.2f}%   |   "
                 f"threshold {threshold:.4f}",
                 ha="center", fontsize=13, family="monospace")
        fig.text(0.5, 0.01,
                 "Research visualisation. Model attention, not lesion "
                 "segmentation, and not a clinical diagnosis.",
                 ha="center", fontsize=9.5, style="italic", color="#555")
        plt.tight_layout(rect=[0, 0.07, 1, 0.93])

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem
            out = save_dir / f"{stem}_gradcam.png"
            plt.savefig(out, dpi=180, bbox_inches="tight")
            with open(save_dir / f"{stem}_prediction.json", "w") as f:
                json.dump(result, f, indent=2)
            result["saved_figure"] = str(out)
        plt.close(fig)

    return result


def predict_folder(folder, model, env, threshold, eval_tf, output_csv=None):
    folder = Path(folder)
    paths = sorted(str(p) for p in folder.rglob("*")
                   if p.is_file() and not p.name.startswith("._")
                   and p.suffix.lower() in VALID_EXT)
    if not paths:
        print("No images found under", folder)
        return pd.DataFrame()

    loader = DataLoader(InferenceDataset(paths, eval_tf),
                        batch_size=env["batch_size"], shuffle=False,
                        num_workers=env["num_workers"], pin_memory=env["pin_memory"])
    autocast_ctx = make_autocast(env)

    model.eval()
    probs = np.zeros(len(paths), dtype=np.float32)
    with torch.no_grad():
        for xb, idx in loader:
            xb = xb.to(env["device"], non_blocking=env["pin_memory"])
            with autocast_ctx():
                out = model(xb)
            probs[idx.numpy()] = torch.sigmoid(out.float()).cpu().numpy()

    preds = (probs >= threshold).astype(int)
    df = pd.DataFrame({
        "image_path": paths,
        "prediction": [CLASS_NAMES[p] for p in preds],
        "normal_probability": np.round((1 - probs) * 100, 2),
        "pneumonia_probability": np.round(probs * 100, 2),
        "confidence": np.round(np.where(preds == 1, probs, 1 - probs) * 100, 2),
    })
    out = Path(output_csv or (OUTPUT_DIR / "batch_predictions.csv"))
    df.to_csv(out, index=False)
    print(f"Batch predictions -> {out.resolve()}")
    return df


# ==============================================================================
# 9. MAIN
# ==============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    env = setup_environment()
    set_seed(SEED)
    device = env["device"]
    autocast_ctx = make_autocast(env)

    # ---- data ---------------------------------------------------------------
    root = resolve_dataset_root()
    df = build_metadata(root)
    splits = make_splits(df)
    verify_no_leakage(splits)

    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    train_tf, eval_tf = build_transforms()

    train_loader = make_loader(train_df, train_tf, True, env, drop_last=True)
    val_loader = make_loader(val_df, eval_tf, False, env)
    test_loader = make_loader(test_df, eval_tf, False, env)

    print(f"\ntrain {len(train_df):,} img / {len(train_loader):,} batches")
    print(f"val   {len(val_df):,} img / {len(val_loader):,} batches")
    print(f"test  {len(test_df):,} img / {len(test_loader):,} batches")

    # ---- class imbalance ----------------------------------------------------
    n_neg = int((train_df.label == 0).sum())
    n_pos = int((train_df.label == 1).sum())
    pos_weight_value = n_neg / max(n_pos, 1)

    print("\n" + "=" * 78)
    print("CLASS IMBALANCE (training split only)")
    print("=" * 78)
    print(f"NORMAL    : {n_neg:,} ({100 * n_neg / len(train_df):.2f}%)")
    print(f"PNEUMONIA : {n_pos:,} ({100 * n_pos / len(train_df):.2f}%)")
    print(f"pos_weight = N_normal / N_pneumonia = {pos_weight_value:.4f}")
    print("PNEUMONIA is the MAJORITY class here, so pos_weight < 1 and the loss")
    print("down-weights it -- which is what balances the gradient signal.")
    print(f"Majority-class baseline on test: "
          f"{100 * test_df.label.mean():.2f}% accuracy for a constant predictor.")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))

    # ---- model --------------------------------------------------------------
    set_seed(SEED)
    model = PneumoniaEfficientNetB3(pretrained=True).to(device)
    total = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 78)
    print("MODEL")
    print("=" * 78)
    print(f"EfficientNet-B3 | pretrained: {model.pretrained}")
    print(f"Feature dim     : {model.in_features}")
    print(f"Total params    : {total:,}  ({total * 4 / 1024 ** 2:.1f} MB fp32)")
    print(f"Head            : Dropout({DROPOUT_P}) -> Linear({model.in_features}, 1)")
    print(f"Grad-CAM layer  : backbone.features[-1] "
          f"({type(model.gradcam_target_layer).__name__})")

    history = []

    # ---- STAGE 1: head only -------------------------------------------------
    print("\n" + "=" * 78)
    print(f"STAGE 1 -- frozen backbone, head only | epochs {NUM_EPOCHS_STAGE1} "
          f"| lr {LEARNING_RATE_S1}")
    print("=" * 78)
    print("The head is randomly initialised. Unfreezing the backbone now would let")
    print("its large, noisy gradients wash out the pretrained filters within a few")
    print("hundred steps. Train the head first, then fine-tune.\n")

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
        trm, vm = compute_metrics(trY, trp), compute_metrics(vY, vp)
        print(f"[S1 {ep}] {time.time() - t0:.0f}s | train loss {trl:.4f} "
              f"acc {trm['accuracy']:.4f}")
        print(f"        val loss {vl:.4f} | {fmt_metrics(vm)}")
        history.append({"stage": 1, "epoch": ep, "train_loss": trl, "val_loss": vl,
                        "train_acc": trm["accuracy"], **{f"val_{k}": v for k, v in vm.items()}})

        score = vm[MONITOR_METRIC] if MONITOR_METRIC != "loss" else -vl
        if score > best:
            best = score
            save_checkpoint(model, CHECKPOINT_PATH,
                            extra={"stage": 1, "epoch": ep, "val_metrics": vm,
                                   "monitor_metric": MONITOR_METRIC,
                                   "monitor_value": float(score)})
            print(f"        -> new best ({MONITOR_METRIC} {score:.4f}); saved")

    # ---- STAGE 2: partial unfreeze -----------------------------------------
    ck = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])

    print("\n" + "=" * 78)
    print(f"STAGE 2 -- fine-tune features[{UNFREEZE_FROM}:] | max epochs "
          f"{NUM_EPOCHS_STAGE2}")
    print("=" * 78)
    print("Early blocks encode generic edges and texture that transfer unchanged.")
    print("Later blocks must move from 'ImageNet objects' to 'lung parenchyma'.\n")

    model = freeze_backbone(model, UNFREEZE_FROM)
    bb = [p for p in model.backbone.features.parameters() if p.requires_grad]
    hd = [p for p in model.backbone.classifier.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(
        [{"params": bb, "lr": LEARNING_RATE_S2 * BACKBONE_LR_MULT},
         {"params": hd, "lr": LEARNING_RATE_S2}], weight_decay=WEIGHT_DECAY)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt2, mode="min" if MONITOR_METRIC == "loss" else "max",
        factor=0.3, patience=max(PATIENCE // 2, 1), min_lr=1e-7)
    scaler = make_scaler(env)

    best2, best_ep, stale, recoveries = -np.inf, None, 0, 0

    for ep in range(1, NUM_EPOCHS_STAGE2 + 1):
        t0 = time.time()
        trl, trp, trY = run_epoch(model, train_loader, criterion, env, autocast_ctx,
                                  opt2, scaler, f"S2 train {ep}/{NUM_EPOCHS_STAGE2}")

        if not model_is_finite(model):
            recoveries += 1
            print(f"  [recovery {recoveries}/3] non-finite weights or BatchNorm "
                  f"buffers after epoch {ep}.")
            if recoveries > 3:
                print("  Too many recoveries; stopping and keeping the last good "
                      "checkpoint.")
                break
            _c = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(_c["model_state_dict"])
            model.to(device)
            for g in opt2.param_groups:
                g["lr"] *= 0.25
            scaler = make_scaler(env)
            print(f"  Restored best checkpoint, cut LRs 4x "
                  f"(head lr {opt2.param_groups[-1]['lr']:.2e}). Retrying.")
            continue

        vl, vp, vY = run_epoch(model, val_loader, criterion, env, autocast_ctx,
                               desc=f"S2 val   {ep}/{NUM_EPOCHS_STAGE2}")
        trm, vm = compute_metrics(trY, trp), compute_metrics(vY, vp)
        score = vl if MONITOR_METRIC == "loss" else vm[MONITOR_METRIC]
        sched2.step(score)

        print(f"[S2 {ep}] {time.time() - t0:.0f}s | head lr "
              f"{opt2.param_groups[-1]['lr']:.2e} | train loss {trl:.4f} "
              f"acc {trm['accuracy']:.4f}")
        print(f"        val loss {vl:.4f} | {fmt_metrics(vm)}")
        history.append({"stage": 2, "epoch": ep, "train_loss": trl, "val_loss": vl,
                        "train_acc": trm["accuracy"], **{f"val_{k}": v for k, v in vm.items()}})

        cmp_score = -score if MONITOR_METRIC == "loss" else score
        if cmp_score > best2 + 1e-4:
            best2, best_ep, stale = cmp_score, ep, 0
            save_checkpoint(model, CHECKPOINT_PATH,
                            extra={"stage": 2, "epoch": ep, "val_metrics": vm,
                                   "monitor_metric": MONITOR_METRIC,
                                   "monitor_value": float(score)})
            print(f"        -> new best ({MONITOR_METRIC} {score:.4f}); saved")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"\nEarly stopping at epoch {ep}: no improvement for "
                      f"{PATIENCE} epochs.")
                break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    plot_history(hist_df)

    # ---- load best, select threshold on VALIDATION --------------------------
    best_ck = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(best_ck["model_state_dict"])
    model.to(device).eval()
    print(f"\nLoaded BEST checkpoint: stage {best_ck.get('stage')} "
          f"epoch {best_ck.get('epoch')} | val {MONITOR_METRIC} "
          f"{best_ck.get('monitor_value', float('nan')):.4f}")

    _, vp, vY = run_epoch(model, val_loader, criterion, env, autocast_ctx,
                          desc="validation predictions")
    threshold, why, sweep = select_threshold(vY, vp)

    print("\n" + "=" * 78)
    print("THRESHOLD SELECTION (validation only -- test set untouched)")
    print("=" * 78)
    print(f"Criterion : {THRESHOLD_CRITERION}")
    print(f"Rationale : {why}")
    print(f"Selected  : {threshold:.4f}")
    print(f"  @ 0.5000 : {fmt_metrics(compute_metrics(vY, vp, 0.5))}")
    print(f"  @ {threshold:.4f} : {fmt_metrics(compute_metrics(vY, vp, threshold))}")
    print("\nTHRESHOLD IS NOW LOCKED. It is not re-tuned on the test set.")
    plot_threshold(sweep, vY, vp, threshold)

    # ---- test evaluation ----------------------------------------------------
    _, tp_, tY = run_epoch(model, test_loader, criterion, env, autocast_ctx,
                           desc="test predictions")
    tm = compute_metrics(tY, tp_, threshold)

    print("\n" + "=" * 78)
    print(f"TEST EVALUATION | n = {len(tY):,} | threshold {threshold:.4f} (locked)")
    print("=" * 78)
    print(f"Accuracy             : {tm['accuracy']:.4f}  "
          f"(majority baseline {max(tY.mean(), 1 - tY.mean()):.4f})")
    print(f"Precision (PPV)      : {tm['precision']:.4f}")
    print(f"Recall / Sensitivity : {tm['recall']:.4f}   <- the safety-critical one")
    print(f"Specificity (TNR)    : {tm['specificity']:.4f}")
    print(f"F1                   : {tm['f1']:.4f}")
    print(f"ROC-AUC              : {tm['roc_auc']:.4f}")
    print(f"PR-AUC               : {tm['pr_auc']:.4f}")
    print(f"\nTP {tm['tp']:,}  TN {tm['tn']:,}  FP {tm['fp']:,}  FN {tm['fn']:,}")
    print(f"Missed pneumonia (FN): {tm['fn']:,} of {tm['tp'] + tm['fn']:,} "
          f"({100 * tm['fn'] / max(tm['tp'] + tm['fn'], 1):.2f}%)")
    print("\n" + classification_report(tY, (tp_ >= threshold).astype(int),
                                       target_names=["NORMAL", "PNEUMONIA"], digits=4))
    plot_evaluation(tY, tp_, threshold, tm)

    # ---- final checkpoint with everything inference needs -------------------
    save_checkpoint(model, CHECKPOINT_PATH, extra={
        "stage": best_ck.get("stage"), "epoch": best_ck.get("epoch"),
        "threshold": float(threshold),
        "threshold_criterion": THRESHOLD_CRITERION,
        "threshold_rationale": why,
        "val_metrics": {k: float(v) for k, v in compute_metrics(vY, vp, threshold).items()},
        "test_metrics": {k: float(v) for k, v in tm.items()},
        "n_train_images": len(train_df), "n_val_images": len(val_df),
        "n_test_images": len(test_df),
        "split_note": ("official test/ untouched; train/val re-partitioned at "
                       "group level because the shipped val/ had 16 images"),
    })
    print(f"\nCheckpoint saved -> {CHECKPOINT_PATH.resolve()}")
    with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump({"threshold": threshold,
                   "test_metrics": {k: float(v) for k, v in tm.items()}}, f, indent=2)

    # ---- demo: one NORMAL and one PNEUMONIA from the test set ---------------
    print("\n" + "=" * 78)
    print("INFERENCE DEMONSTRATION")
    print("=" * 78)
    demo_dir = OUTPUT_DIR / "demo"
    for lab in (0, 1):
        sub = test_df[test_df.label == lab]
        if len(sub):
            row = sub.sample(1, random_state=SEED).iloc[0]
            print(f"\n[ground truth: {row.label_name}]")
            predict_xray(row.image_path, model, env, threshold, eval_tf,
                         save_dir=demo_dir)
    print(f"\nDemo figures -> {demo_dir.resolve()}")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print("Research/educational system. Every output is a MODEL PREDICTION, not a")
    print("clinical diagnosis. Trained on paediatric single-centre data; it does")
    print("not transfer to adults, to other hospitals, or to any other pathology.")
    print("\nNEXT: inspect the Grad-CAM figures. If confident PNEUMONIA predictions")
    print("consistently highlight image borders, corner markers or burned-in text")
    print("rather than lung parenchyma, the model is a marker detector and the test")
    print("score is measuring a dataset artifact, not diagnosis.")


# ==============================================================================
# PLOTS
# ==============================================================================

def plot_history(h):
    if h.empty:
        return
    h = h.reset_index(drop=True)
    x = np.arange(1, len(h) + 1)
    s2 = h.index[h.stage == 2].min() + 1 if (h.stage == 2).any() else None

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(x, h.train_loss, "-o", ms=4, label="train")
    ax[0, 0].plot(x, h.val_loss, "-o", ms=4, label="validation")
    ax[0, 0].set_title("Loss"); ax[0, 0].legend(); ax[0, 0].set_xlabel("epoch")

    ax[0, 1].plot(x, h.train_acc, "-o", ms=4, label="train")
    ax[0, 1].plot(x, h.val_accuracy, "-o", ms=4, label="validation")
    ax[0, 1].set_title("Accuracy"); ax[0, 1].legend(); ax[0, 1].set_xlabel("epoch")

    ax[1, 0].plot(x, h.val_roc_auc, "-o", ms=4, label="ROC-AUC")
    ax[1, 0].plot(x, h.val_pr_auc, "-s", ms=4, label="PR-AUC")
    ax[1, 0].set_title("Validation AUC"); ax[1, 0].legend(); ax[1, 0].set_xlabel("epoch")

    ax[1, 1].plot(x, h.val_f1, "-o", ms=4, label="F1")
    ax[1, 1].plot(x, h.val_recall, "--", label="recall")
    ax[1, 1].plot(x, h.val_precision, "--", label="precision")
    ax[1, 1].set_title("Validation F1 / P / R"); ax[1, 1].legend(); ax[1, 1].set_xlabel("epoch")

    for a in ax.ravel():
        if s2:
            a.axvline(s2 - 0.5, color="gray", ls="--", lw=1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_history.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_threshold(sweep, y, p, t):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for col in ("f1", "precision", "recall", "specificity"):
        ax[0].plot(sweep.threshold, sweep[col], label=col)
    ax[0].axvline(t, c="k", ls="--", label=f"selected {t:.3f}")
    ax[0].axvline(0.5, c="gray", ls=":", label="default 0.5")
    ax[0].set_xlabel("threshold"); ax[0].set_title("Validation metrics vs threshold")
    ax[0].legend(fontsize=8)

    ax[1].hist(p[y == 0], bins=40, alpha=0.65, label="NORMAL")
    ax[1].hist(p[y == 1], bins=40, alpha=0.65, label="PNEUMONIA")
    ax[1].axvline(t, c="k", ls="--")
    ax[1].set_xlabel("predicted P(pneumonia)")
    ax[1].set_title("Validation probability by true class"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_selection.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_evaluation(y, p, t, m):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    cm = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1])
    ax[0].imshow(cm, cmap="Blues")
    ax[0].set_xticks([0, 1], ["Pred NORMAL", "Pred PNEU"])
    ax[0].set_yticks([0, 1], ["True NORMAL", "True PNEU"])
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=15)
    ax[0].set_title(f"Confusion matrix @ {t:.3f}")

    fpr, tpr, _ = roc_curve(y, p)
    ax[1].plot(fpr, tpr, lw=2, label=f"AUC = {m['roc_auc']:.4f}")
    ax[1].plot([0, 1], [0, 1], "--", c="gray", lw=1)
    ax[1].set_xlabel("1 - specificity"); ax[1].set_ylabel("sensitivity")
    ax[1].set_title("ROC curve"); ax[1].legend(loc="lower right")

    pr, rc, _ = precision_recall_curve(y, p)
    ax[2].plot(rc, pr, lw=2, c="#C1445A", label=f"PR-AUC = {m['pr_auc']:.4f}")
    ax[2].axhline(y.mean(), ls="--", c="gray", lw=1, label=f"prevalence {y.mean():.3f}")
    ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision")
    ax[2].set_title("Precision-Recall curve"); ax[2].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_evaluation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================

if __name__ == "__main__":
    # The __main__ guard is required on Windows: DataLoader workers use the
    # "spawn" start method and re-import this module in each child process.
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
