#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MURA v1.1 -- Musculoskeletal X-ray Abnormality Detection
EfficientNet-B3 | Binary classification (Normal vs. Abnormal) | Patient-level splits | Grad-CAM

Converted from MURA_EfficientNetB3_GradCAM.ipynb into a single runnable .py script.
Tuned for GPU support on NVIDIA GeForce RTX 4060 (8 GB and 16 GB variants), and falls
back cleanly to any other CUDA GPU or to CPU. See the "RTX 4060 tuning" block near the
top of the Configuration section for the specific changes made for this GPU.

Fixes in this revision (over the previous .py conversion):
  1. Mixed precision now uses bfloat16 on GPUs that support it (Ampere/Ada, incl. the
     RTX 4060 family) instead of float16. This is the fix for the
     "ValueError: Input contains NaN" crash in stage 2: float16 activations inside
     EfficientNet's MBConv blocks can overflow to +inf, and that inf is written into
     BatchNorm's running_mean/running_var during the forward pass, where GradScaler
     cannot see or undo it. Training then continues to look healthy (train mode uses
     batch statistics) while every eval-mode forward pass returns NaN. bfloat16 has
     float32's exponent range, so the overflow cannot happen, and it needs no
     GradScaler. float16 + GradScaler remains the automatic fallback for older cards.
  2. run_epoch skips any micro-batch whose loss is non-finite instead of
     backpropagating it.
  3. After every stage-2 training epoch, all parameters AND buffers are checked for
     finiteness. If they are corrupted, the last good checkpoint is reloaded, the
     learning rates are cut 4x, and the epoch is retried (up to 3 times).
  4. compute_metrics sanitises non-finite probabilities and warns, so a numerical
     problem can never kill a multi-hour run inside sklearn again.
  5. macOS AppleDouble sidecar files ("._image1.png") are excluded during dataset
     parsing. The Kaggle mirror ships four of them; they are not images, and they were
     being fed to the model as blank placeholder tensors (image count 40,009 vs the
     official 40,005).
  6. The "channels identical" sanity check now tests the tensor before normalisation.
     It previously compared post-Normalize channels, which differ by design because
     ImageNet mean/std are per-channel, and so always printed False.

Windows note: MURADataset, _InferenceDataset, and seed_worker are defined at true
module level (before `main()`) rather than inline where the notebook had them,
because DataLoader worker processes on Windows ("spawn" start method) need to
pickle these by their import path. Everything else stays inside `main()`, guarded
by `if __name__ == "__main__":` at the bottom of the file.
"""

import os
# Reduce CUDA memory fragmentation -- helps most on 8 GB cards like the
# RTX 4060 / 4060 Ti / 4060 Laptop. Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")



# ==============================================================================
# Module-level definitions required for Windows-compatible multiprocessing
# ==============================================================================
# DataLoader workers on Windows use the multiprocessing "spawn" start method,
# which pickles the Dataset class (by import path) and worker_init_fn to send
# to each worker process. Objects defined *inside* a function (as they would be
# if left in-place in the notebook-derived `main()` below) are "local objects"
# and cannot be pickled, which raises:
#   _pickle.PicklingError: Can't pickle local object '...MURADataset'
# So MURADataset, _InferenceDataset, and seed_worker live here, at true module
# scope, importable by worker processes. Everything else in the pipeline stays
# inside main(), guarded by `if __name__ == "__main__":` at the bottom of this
# file, which is what makes spawning safe in the first place.

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True   # a few MURA files are mildly truncated
Image.MAX_IMAGE_PIXELS = None

IMAGE_SIZE = 300   # EfficientNet-B3's native training resolution (single source of truth)


# ----------------------------------------------------------------------------
# [Code cell 28]
class MURADataset(Dataset):
    """MURA radiographs -> (3xHxW float tensor, label float, index).

    Grayscale is replicated across 3 channels to match the ImageNet-pretrained stem
    (see the markdown above for why).
    """

    def __init__(self, df, transform=None, return_path=False):
        self.paths  = df["image_path"].values
        self.labels = df["label"].values.astype(np.float32)
        self.transform = transform
        self.return_path = return_path
        self._warned = 0

    def __len__(self):
        return len(self.paths)

    def _safe_load(self, path):
        try:
            with Image.open(path) as im:
                return im.convert("L").copy()        # force single-channel grayscale
        except Exception as e:
            if self._warned < 10:
                print(f"[MURADataset] unreadable image -> {path} ({e}); using blank placeholder")
                self._warned += 1
            return Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)

    def __getitem__(self, idx):
        img = self._safe_load(self.paths[idx])
        img = img.convert("RGB")                     # replicate L -> 3 identical channels
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.return_path:
            return img, label, self.paths[idx]
        return img, label, idx


# ----------------------------------------------------------------------------
# [Code cell 60]
class _InferenceDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        try:
            with Image.open(p) as im:
                img = im.convert("L").convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        return self.transform(img), i


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    """
    Entire pipeline (data prep, training, evaluation, Grad-CAM, inference demo).
    Wrapped in a function -- rather than left at module level like the notebook --
    so that `if __name__ == "__main__":` guard below works correctly with
    multiprocessing DataLoader workers (NUM_WORKERS > 0) on Windows, which is the
    default OS for RTX 4060 desktop/laptop setups. Without this guard, Windows
    workers using the "spawn" start method would re-execute the whole script,
    including model downloads and training, in every worker process.
    """

    # ============================================================================
    # [Markdown cell 0]
    # # MURA v1.1 — Musculoskeletal X-ray Abnormality Detection
    # 
    # **EfficientNet-B3 · Binary classification (Normal vs. Abnormal) · Patient-level splits · Grad-CAM explainability**
    # 
    # ---
    # 
    # ### What this notebook is
    # 
    # A complete, GPU-compatible research/educational deep-learning pipeline for the Stanford **MURA v1.1**
    # musculoskeletal radiograph dataset. It covers dataset parsing and analysis, leakage-free splitting,
    # preprocessing, two-stage fine-tuning of EfficientNet-B3, threshold selection, held-out test evaluation,
    # Grad-CAM explainability, model saving/loading, and single-image + batch inference.
    # 
    # ### Label semantics — read this first
    # 
    # MURA labels are provided at the **study** level as `positive` / `negative`, meaning
    # **abnormal** / **normal**. A "positive" study may contain fractures, hardware, degenerative changes,
    # lesions, or other abnormalities. Therefore:
    # 
    # * class `0` = **Normal**
    # * class `1` = **Abnormal**
    # 
    # Class 1 is **not** called "Fracture" anywhere in this notebook, because MURA does not provide a
    # fracture-specific label.
    # 
    # ### Medical safety statement
    # 
    # > This is a **research / educational** machine-learning system. It does **not** provide a medical
    # > diagnosis. All outputs are **model predictions**, not clinical findings. Grad-CAM output is a
    # > **model interpretability visualization** showing model-attributed importance — it is **not** a
    # > pixel-level lesion segmentation and must not be read as definitive localization of pathology.
    # 
    # ### Notebook sections
    # 
    # | # | Section |
    # |---|---------|
    # | 1 | Environment and GPU setup |
    # | 2 | Imports |
    # | 3 | Configuration |
    # | 4 | Dataset download and path resolution |
    # | 5 | Metadata creation (directory parsing) |
    # | 6 | Dataset analysis |
    # | 7 | Analysis plots |
    # | 8 | Patient/study-level splitting |
    # | 9 | Leakage verification |
    # | 10 | Dataset class |
    # | 11 | Data augmentation / transforms |
    # | 12 | DataLoaders |
    # | 13 | Class imbalance calculation |
    # | 14 | EfficientNet-B3 model |
    # | 15 | Training utilities |
    # | 16 | Stage 1 training (frozen backbone) |
    # | 17 | Stage 2 fine-tuning (partial unfreeze) |
    # | 18 | Training history plots |
    # | 19 | Threshold selection (validation only) |
    # | 20 | Test evaluation |
    # | 21 | Grad-CAM implementation |
    # | 22 | Model saving / loading |
    # | 23 | Single-image inference (`predict_xray`) |
    # | 24 | Batch inference (`predict_folder`) |
    # | 25 | Final end-to-end demonstration |
    # 
    # No frontend, web app, API, database, or deployment interface is built here — by design.

    # ============================================================================
    # [Markdown cell 1]
    # ---
    # ## 1. Environment and GPU setup
    # 
    # Detect CUDA, print the GPU name, and select the device. Everything downstream uses this single
    # `DEVICE` object, so the notebook runs unchanged on a T4, on an RTX card, or on CPU.

    # ----------------------------------------------------------------------------
    # [Code cell 2]
    # If a package is missing (e.g. on a fresh Colab runtime), uncomment:
    # !pip -q install torch torchvision scikit-learn pandas matplotlib seaborn pillow tqdm kagglehub opencv-python-headless

    import subprocess, sys

    try:
        print(subprocess.check_output(["nvidia-smi"], text=True))
    except Exception as e:
        print("nvidia-smi unavailable ->", e)
        print("Continuing; the notebook will fall back to CPU if CUDA is not present.")

    # ----------------------------------------------------------------------------
    # [Code cell 3]
    import torch

    # ---- Single source of truth for the compute device -------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("PyTorch version      :", torch.__version__)
    print("CUDA available       :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version         :", torch.version.cuda)
        print("GPU name             :", torch.cuda.get_device_name(0))
        _props = torch.cuda.get_device_properties(0)
        TOTAL_GPU_GB = _props.total_memory / 1024**3
        print("GPU total memory     : {:.2f} GB".format(TOTAL_GPU_GB))
        print("GPU count            :", torch.cuda.device_count())
    else:
        TOTAL_GPU_GB = 0.0
        print("No CUDA device detected -> running on CPU (training will be slow).")

    print("Selected device      :", DEVICE)

    # Mixed precision is only meaningful on CUDA; on CPU we disable it cleanly.
    AMP_ENABLED = DEVICE.type == "cuda"
    print("Mixed precision (AMP):", AMP_ENABLED)

    if DEVICE.type == "cuda":
        # TF32 gives a free speed-up on Ampere+ and Ada Lovelace (RTX 40-series, incl. the
        # 4060 / 4060 Ti / 4060 Laptop) and is a no-op on T4.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # ---- RTX 4060 tuning note --------------------------------------------------
        # The 4060 (8 GB desktop/laptop) and 4060 Ti (8 GB or 16 GB) are Ada Lovelace
        # GPUs: they support TF32, fp16/bf16 tensor cores and AMP just like the T4 this
        # notebook was originally tuned for, but with less VRAM on the 8 GB variants.
        # BATCH_SIZE and GRAD_ACCUM_STEPS below are chosen automatically from detected
        # VRAM so the 8 GB cards use a smaller physical batch (16) with 2-step gradient
        # accumulation (effective batch 32, matching the T4 default), while the 16 GB
        # 4060 Ti trains at physical batch 32 directly, same as a T4.
        _gpu_name_lower = torch.cuda.get_device_name(0).lower()
        IS_RTX_4060 = "4060" in _gpu_name_lower
        if IS_RTX_4060:
            print(f"Detected RTX 4060-family GPU ({torch.cuda.get_device_name(0)}) "
                  f"-> using VRAM-aware batch size / gradient accumulation below.")
    else:
        IS_RTX_4060 = False

    # ============================================================================
    # [Markdown cell 4]
    # ---
    # ## 2. Imports

    # ----------------------------------------------------------------------------
    # [Code cell 5]
    import os
    import io
    import gc
    import json
    import glob
    import time
    import math
    import random
    import hashlib
    import warnings
    from pathlib import Path
    from collections import Counter, defaultdict

    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image, ImageFile

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    import torchvision
    from torchvision import transforms
    from torchvision.models import efficientnet_b3

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, roc_curve,
        precision_recall_curve, confusion_matrix, classification_report,
    )
    from sklearn.model_selection import GroupShuffleSplit

    from tqdm.auto import tqdm

    warnings.filterwarnings("ignore")
    ImageFile.LOAD_TRUNCATED_IMAGES = True   # a few MURA files are mildly truncated
    Image.MAX_IMAGE_PIXELS = None

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 140

    print("torch       :", torch.__version__)
    print("torchvision :", torchvision.__version__)
    print("numpy       :", np.__version__)
    print("pandas      :", pd.__version__)

    # ============================================================================
    # [Markdown cell 6]
    # ---
    # ## 3. Configuration
    # 
    # Every knob that matters is here. Nothing important is hard-coded further down.
    # 
    # **Batch-size auto-configuration:** EfficientNet-B3 at 300×300 with AMP fits comfortably at batch 32
    # on a 16 GB T4. The helper below scales the batch size to the detected GPU memory and falls back to a
    # small CPU-safe value when no GPU is present.

    # ----------------------------------------------------------------------------
    # [Code cell 7]
    # ------------------------------ Reproducibility -----------------------------
    SEED = 42

    # ------------------------------ Data ----------------------------------------
    KAGGLE_DATASET   = "cjinny/mura-v11"
    DATA_ROOT        = None          # set to a local path to skip the download, e.g. "/content/MURA-v1.1"
    # IMAGE_SIZE is defined once at true module level (above `def main():`) since
    # the module-level MURADataset / _InferenceDataset classes need it too.

    # Split strategy:
    #   "official_valid_as_test" -> MURA's official validation set becomes our held-out TEST set,
    #                               and our validation set is carved out of the official train set
    #                               at the PATIENT level. This is the defensible default because the
    #                               official split is already patient-disjoint.
    #   "random_70_15_15"        -> pool everything and make a fresh patient-level 70/15/15 split.
    SPLIT_STRATEGY   = "official_valid_as_test"
    VAL_FRACTION     = 0.165         # fraction of official-train PATIENTS held out for validation
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15   # used by "random_70_15_15"

    # Restrict to specific body regions, or None for all seven.
    BODY_PARTS       = None          # e.g. ["XR_WRIST", "XR_HAND"]

    # ------------------------------ Training ------------------------------------
    NUM_EPOCHS_STAGE1 = 3            # head-only warm-up
    NUM_EPOCHS_STAGE2 = 12           # partial-unfreeze fine-tuning
    LEARNING_RATE_S1  = 1e-3
    LEARNING_RATE_S2  = 1e-4
    BACKBONE_LR_MULT  = 0.25         # backbone learns slower than the head in stage 2
    WEIGHT_DECAY      = 1e-4
    PATIENCE          = 4            # early stopping patience (epochs, stage 2)
    LABEL_SMOOTHING   = 0.0
    GRAD_CLIP_NORM    = 5.0
    UNFREEZE_FROM     = 5            # unfreeze backbone.features[UNFREEZE_FROM:] in stage 2
    DROPOUT_P         = 0.4
    MONITOR_METRIC    = "roc_auc"    # metric used for best-checkpoint selection ("roc_auc"/"f1"/"loss")

    # ------------------------------ Threshold -----------------------------------
    # "f1"      -> threshold maximising F1 on the validation set
    # "youden"  -> threshold maximising (sensitivity + specificity - 1)
    # "recall_at" -> lowest threshold achieving TARGET_RECALL (screening-oriented)
    THRESHOLD_CRITERION = "f1"
    TARGET_RECALL       = 0.90

    # ------------------------------ Runtime -------------------------------------
    def auto_batch_size():
        if DEVICE.type != "cuda":
            return 8
        if TOTAL_GPU_GB >= 30:   return 64
        if TOTAL_GPU_GB >= 14:   return 32     # T4 (16 GB) and RTX 4060 Ti 16 GB land here
        if TOTAL_GPU_GB >= 10:   return 24
        return 16                              # RTX 4060 / 4060 Ti 8 GB lands here


    def auto_grad_accum_steps():
        """Gradient accumulation to keep the *effective* batch size roughly constant
        (~32) across GPUs with less VRAM, without raising peak memory. Only kicks in
        below 10 GB, which covers the 8 GB RTX 4060 / 4060 Ti / 4060 Laptop GPUs."""
        if DEVICE.type != "cuda":
            return 1
        if TOTAL_GPU_GB < 10:
            return 2
        return 1

    # Both can be overridden manually, e.g. `MURA_BATCH_SIZE=8 python mura_...py`
    # if you still hit CUDA OOM on an 8 GB card with other programs also using the GPU.
    BATCH_SIZE       = int(os.environ.get("MURA_BATCH_SIZE", auto_batch_size()))
    GRAD_ACCUM_STEPS = int(os.environ.get("MURA_GRAD_ACCUM_STEPS", auto_grad_accum_steps()))

    # On Windows, DataLoader workers use the "spawn" start method, which re-imports
    # this script in each worker process. That only works safely because the whole
    # pipeline below is wrapped in `main()` and guarded by `if __name__ == "__main__"`
    # at the bottom of this file — do not move this code back to true module level.
    NUM_WORKERS = min(4, os.cpu_count() or 2) if DEVICE.type == "cuda" else 2
    PIN_MEMORY  = DEVICE.type == "cuda"
    PERSISTENT_WORKERS = NUM_WORKERS > 0
    PREFETCH_FACTOR    = 4 if NUM_WORKERS > 0 else None

    # ------------------------------ Normalisation -------------------------------
    # ImageNet statistics, because the backbone is an ImageNet-pretrained checkpoint.
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    # ------------------------------ Analysis / output ---------------------------
    ANALYSIS_SAMPLE      = 3000      # images sampled for dimension / integrity / duplicate stats
    FULL_INTEGRITY_CHECK = False     # True = verify every image (slow, ~40k files)
    OUTPUT_DIR     = Path("./mura_outputs");     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR = Path("./mura_checkpoints"); CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH = CHECKPOINT_DIR / "mura_efficientnet_b3_best.pth"
    HISTORY_PATH    = OUTPUT_DIR / "training_history.json"

    MODEL_NAME = "efficientnet_b3"
    CLASS_NAMES = {0: "Normal", 1: "Abnormal"}

    print("Batch size   :", BATCH_SIZE)
    print("Grad accum   :", GRAD_ACCUM_STEPS, f"(effective batch = {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print("Num workers  :", NUM_WORKERS)
    print("Pin memory   :", PIN_MEMORY)
    print("Image size   :", IMAGE_SIZE)
    print("Checkpoint   :", CHECKPOINT_PATH.resolve())

    # ============================================================================
    # [Markdown cell 8]
    # ### Reproducibility setup
    # 
    # Seeds are set for Python, NumPy and PyTorch. Full cuDNN determinism is *available* but costs
    # noticeable throughput on convolutional backbones, so it is behind a flag; by default we use the
    # benchmark autotuner and accept run-to-run jitter of a few tenths of a percent.

    # ----------------------------------------------------------------------------
    # [Code cell 9]
    DETERMINISTIC = False   # set True for bit-wise reproducibility at a speed cost

    def set_seed(seed=SEED, deterministic=DETERMINISTIC):
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
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    # NOTE (script conversion): seed_worker() is defined at true module level
    # (above `def main():`), not here, because it is passed as `worker_init_fn`
    # to DataLoader, which also must be pickled for Windows "spawn" workers.

    set_seed(SEED)
    GENERATOR = torch.Generator(); GENERATOR.manual_seed(SEED)

    print("Random seed        :", SEED)
    print("Deterministic mode :", DETERMINISTIC)
    print("cudnn.benchmark    :", torch.backends.cudnn.benchmark)

    # ============================================================================
    # [Markdown cell 10]
    # ---
    # ## 4. Dataset download and path resolution
    # 
    # The notebook tries, in order:
    # 
    # 1. a user-supplied `DATA_ROOT`,
    # 2. `kagglehub.dataset_download("cjinny/mura-v11")` (works on Colab after `kagglehub` login or with
    #    a `kaggle.json` present),
    # 3. a scan of common local locations (`/content`, `./`, `/kaggle/input`).
    # 
    # It then locates the directory that actually contains `train/` and `valid/` — the archive nests
    # things differently depending on how it was extracted, so we search rather than assume.

    # ----------------------------------------------------------------------------
    # [Code cell 11]
    def try_kagglehub_download(dataset=KAGGLE_DATASET):
        try:
            import kagglehub
        except ImportError:
            print("kagglehub not installed. Run:  pip install kagglehub")
            return None
        try:
            path = kagglehub.dataset_download(dataset)
            print("kagglehub downloaded to:", path)
            return path
        except Exception as e:
            print("kagglehub download failed ->", repr(e))
            return None


    def find_mura_root(search_roots):
        """Return the directory that directly contains MURA's train/ and valid/ folders."""
        candidates = []
        for root in search_roots:
            if root is None:
                continue
            root = Path(root)
            if not root.exists():
                continue
            # the root itself
            if (root / "train").is_dir() and (root / "valid").is_dir():
                candidates.append(root)
            # up to 3 levels down
            for depth in range(1, 4):
                pattern = "/".join(["*"] * depth)
                for p in root.glob(pattern):
                    if p.is_dir() and (p / "train").is_dir() and (p / "valid").is_dir():
                        candidates.append(p)
        # prefer a path whose train/ contains XR_* folders
        for c in candidates:
            if any(q.name.startswith("XR_") for q in (c / "train").iterdir() if q.is_dir()):
                return c
        return candidates[0] if candidates else None


    search_roots = [DATA_ROOT]
    if DATA_ROOT is None:
        kh = try_kagglehub_download()
        search_roots += [kh, "/content", "/kaggle/input", ".", "./data", "/content/drive/MyDrive"]

    MURA_ROOT = find_mura_root(search_roots)

    if MURA_ROOT is None:
        raise FileNotFoundError(
            "Could not locate the MURA v1.1 dataset.\n"
            "Expected a directory containing 'train/' and 'valid/' subfolders with XR_* body regions.\n"
            "Fix: set DATA_ROOT in the Configuration cell to the extracted dataset path, or authenticate "
            "kagglehub/Kaggle and re-run this cell."
        )

    MURA_ROOT = Path(MURA_ROOT)
    print("MURA root resolved to:", MURA_ROOT.resolve())
    print("\nTop-level contents:")
    for p in sorted(MURA_ROOT.iterdir())[:20]:
        print("  ", p.name + ("/" if p.is_dir() else ""))
    print("\nBody regions under train/:")
    print("  ", sorted(q.name for q in (MURA_ROOT / "train").iterdir() if q.is_dir()))

    # ============================================================================
    # [Markdown cell 12]
    # ---
    # ## 5. Metadata creation — parse the real directory tree
    # 
    # MURA's on-disk layout is:
    # 
    # ```
    # MURA-v1.1/
    #   train/
    #     XR_SHOULDER/
    #       patient00001/
    #         study1_positive/
    #           image1.png
    #           image2.png
    # ```
    # 
    # We walk the tree with `os.scandir` and build a dataframe rather than trusting the shipped CSVs, so
    # the metadata always reflects the files that are actually present. Every row records the identifiers
    # needed for leakage-free splitting.
    # 
    # **Patient-ID scope:** a patient folder name such as `patient00011` can recur under different body
    # regions. Because we cannot be certain those are different individuals, we use the **bare patient ID
    # as a global grouping key**. That is the conservative choice — it can only make the split stricter,
    # never leakier.

    # ----------------------------------------------------------------------------
    # [Code cell 13]
    VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def parse_label_from_study(study_dir_name):
        name = study_dir_name.lower()
        if "positive" in name:
            return 1          # abnormal
        if "negative" in name:
            return 0          # normal
        return None


    def build_metadata(mura_root, splits=("train", "valid"), body_parts=None):
        rows = []
        unlabeled, empty_studies = [], []

        for split in splits:
            split_dir = Path(mura_root) / split
            if not split_dir.is_dir():
                print("  [warn] missing split directory:", split_dir)
                continue

            for bp_entry in sorted(os.scandir(split_dir), key=lambda e: e.name):
                if not bp_entry.is_dir():
                    continue
                body_part = bp_entry.name
                if body_parts is not None and body_part not in body_parts:
                    continue

                for pat_entry in sorted(os.scandir(bp_entry.path), key=lambda e: e.name):
                    if not pat_entry.is_dir():
                        continue
                    patient_id = pat_entry.name            # e.g. "patient00011"

                    for study_entry in sorted(os.scandir(pat_entry.path), key=lambda e: e.name):
                        if not study_entry.is_dir():
                            continue
                        label = parse_label_from_study(study_entry.name)
                        if label is None:
                            unlabeled.append(study_entry.path)
                            continue

                        # Skip macOS AppleDouble sidecar files ("._image1.png"). They
                        # carry a .png extension but hold resource-fork metadata, not
                        # image data, so PIL cannot open them. The Kaggle mirror of
                        # MURA ships four of these; leaving them in inflates the image
                        # count to 40,009 (official: 40,005) and feeds blank
                        # placeholder tensors into training.
                        images = [f for f in os.scandir(study_entry.path)
                                  if f.is_file()
                                  and not f.name.startswith("._")
                                  and Path(f.name).suffix.lower() in VALID_EXT]
                        if not images:
                            empty_studies.append(study_entry.path)
                            continue

                        study_uid = f"{body_part}/{patient_id}/{study_entry.name}"
                        for img in sorted(images, key=lambda e: e.name):
                            rows.append({
                                "image_path":  img.path,
                                "image_name":  img.name,
                                "body_part":   body_part,
                                "patient_id":  patient_id,          # global grouping key
                                "patient_uid": f"{body_part}/{patient_id}",
                                "study_name":  study_entry.name,
                                "study_uid":   study_uid,
                                "label":       label,
                                "label_name":  "Abnormal" if label == 1 else "Normal",
                                "orig_split":  split,
                            })

        df = pd.DataFrame(rows)
        print(f"Parsed {len(df):,} images from {mura_root}")
        print(f"  unlabeled study folders skipped : {len(unlabeled)}")
        print(f"  empty study folders skipped     : {len(empty_studies)}")
        if unlabeled[:3]:
            print("  examples:", unlabeled[:3])
        return df


    t0 = time.time()
    meta_df = build_metadata(MURA_ROOT, splits=("train", "valid"), body_parts=BODY_PARTS)
    print(f"Directory walk took {time.time() - t0:.1f}s")

    assert len(meta_df) > 0, "No images were parsed — check MURA_ROOT and the directory layout."
    meta_df.head(8)

    # ============================================================================
    # [Markdown cell 14]
    # ---
    # ## 6. Dataset analysis
    # 
    # Counts, class balance, per-body-region breakdowns, patient/study statistics, and file-integrity
    # checks. Pixel-level statistics (dimensions, mode, format, hashes) are computed on a random sample of
    # `ANALYSIS_SAMPLE` images by default — opening all ~40k files is slow and the sample is more than
    # enough to characterise the distribution. Set `FULL_INTEGRITY_CHECK = True` to check everything.

    # ----------------------------------------------------------------------------
    # [Code cell 15]
    n_images   = len(meta_df)
    n_abnormal = int((meta_df.label == 1).sum())
    n_normal   = int((meta_df.label == 0).sum())

    print("=" * 70)
    print("MURA v1.1 — DATASET OVERVIEW")
    print("=" * 70)
    print(f"Total images            : {n_images:,}")
    print(f"Normal   (label 0)      : {n_normal:,}  ({100*n_normal/n_images:.2f}%)")
    print(f"Abnormal (label 1)      : {n_abnormal:,}  ({100*n_abnormal/n_images:.2f}%)")
    print(f"Imbalance ratio (N:A)   : {n_normal/max(n_abnormal,1):.3f} : 1")
    print()
    print(f"Body regions            : {meta_df.body_part.nunique()}")
    print(f"Unique patient IDs      : {meta_df.patient_id.nunique():,}")
    print(f"Unique patient x region : {meta_df.patient_uid.nunique():,}")
    print(f"Unique studies          : {meta_df.study_uid.nunique():,}")

    ipp = meta_df.groupby("study_uid").size()
    print(f"Images per study        : mean {ipp.mean():.2f} | median {ipp.median():.0f} "
          f"| min {ipp.min()} | max {ipp.max()}")
    spp = meta_df.groupby("patient_id")["study_uid"].nunique()
    print(f"Studies per patient     : mean {spp.mean():.2f} | max {spp.max()}")
    print()
    print("Original MURA split sizes (images):")
    print(meta_df.orig_split.value_counts().to_string())

    # ----------------------------------------------------------------------------
    # [Code cell 16]
    print("=" * 70)
    print("PER-BODY-REGION BREAKDOWN")
    print("=" * 70)

    region_stats = (
        meta_df.groupby("body_part")
        .agg(images=("image_path", "size"),
             normal=("label", lambda s: int((s == 0).sum())),
             abnormal=("label", lambda s: int((s == 1).sum())),
             patients=("patient_id", "nunique"),
             studies=("study_uid", "nunique"))
        .assign(abnormal_pct=lambda d: (100 * d.abnormal / d.images).round(2))
        .sort_values("images", ascending=False)
    )
    print(region_stats.to_string())
    print()
    print("Share of all images per region (%):")
    print((100 * region_stats.images / region_stats.images.sum()).round(2).to_string())

    # ----------------------------------------------------------------------------
    # [Code cell 17]
    # ---- Pixel-level statistics, integrity, duplicates -------------------------
    rng = np.random.default_rng(SEED)
    if FULL_INTEGRITY_CHECK:
        sample_idx = np.arange(len(meta_df))
    else:
        sample_idx = rng.choice(len(meta_df), size=min(ANALYSIS_SAMPLE, len(meta_df)), replace=False)

    sample_paths = meta_df.image_path.values[sample_idx]

    widths, heights, modes, formats = [], [], [], []
    corrupted, missing = [], []
    hashes = {}

    for p in tqdm(sample_paths, desc="Inspecting images"):
        if not os.path.exists(p):
            missing.append(p); continue
        try:
            with Image.open(p) as im:
                im.verify()                       # structural check
            with Image.open(p) as im:             # re-open: verify() invalidates the handle
                widths.append(im.width); heights.append(im.height)
                modes.append(im.mode);   formats.append(im.format or "UNKNOWN")
            h = hashlib.md5(open(p, "rb").read()).hexdigest()
            hashes.setdefault(h, []).append(p)
        except Exception as e:
            corrupted.append((p, repr(e)))

    dup_groups = {h: v for h, v in hashes.items() if len(v) > 1}
    n_dup_files = sum(len(v) - 1 for v in dup_groups.values())

    print("\n" + "=" * 70)
    print(f"IMAGE FILE ANALYSIS  (sample of {len(sample_paths):,} files"
          f"{' = FULL dataset' if FULL_INTEGRITY_CHECK else ''})")
    print("=" * 70)
    print(f"Missing files        : {len(missing)}")
    print(f"Corrupted/unreadable : {len(corrupted)}")
    if corrupted[:3]:
        for p, e in corrupted[:3]:
            print("   ", p, "->", e)
    print(f"Exact duplicates     : {n_dup_files} redundant file(s) in {len(dup_groups)} group(s)")
    if dup_groups:
        k = next(iter(dup_groups))
        print("   example group:", [os.path.basename(x) for x in dup_groups[k][:3]])
    print()
    print(f"Width  : min {min(widths)} | max {max(widths)} | mean {np.mean(widths):.1f} | median {np.median(widths):.0f}")
    print(f"Height : min {min(heights)} | max {max(heights)} | mean {np.mean(heights):.1f} | median {np.median(heights):.0f}")
    print(f"Aspect ratio (W/H)   : mean {np.mean(np.array(widths)/np.array(heights)):.3f}")
    print()
    print("Image modes/channels :", dict(Counter(modes)))
    print("Image formats        :", dict(Counter(formats)))
    print()
    print("Takeaway: image sizes vary widely, so a fixed resize to "
          f"{IMAGE_SIZE}x{IMAGE_SIZE} is required before batching.")

    dim_df = pd.DataFrame({"width": widths, "height": heights})

    # ============================================================================
    # [Markdown cell 18]
    # ---
    # ## 7. Analysis plots

    # ----------------------------------------------------------------------------
    # [Code cell 19]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # (1) Class distribution
    counts = meta_df.label_name.value_counts().reindex(["Normal", "Abnormal"])
    axes[0].bar(counts.index, counts.values, color=["#4C9F70", "#C1445A"])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}\n({100*v/len(meta_df):.1f}%)", ha="center", va="bottom", fontsize=10)
    axes[0].set_title("1. Class distribution (image level)")
    axes[0].set_ylabel("Images"); axes[0].set_ylim(0, counts.max() * 1.18)

    # (2) Body-region distribution
    rc = meta_df.body_part.value_counts()
    axes[1].barh(rc.index[::-1], rc.values[::-1], color="#3C6E9F")
    for i, v in enumerate(rc.values[::-1]):
        axes[1].text(v, i, f" {v:,}", va="center", fontsize=9)
    axes[1].set_title("2. Body-region distribution")
    axes[1].set_xlabel("Images"); axes[1].set_xlim(0, rc.max() * 1.15)

    plt.tight_layout(); plt.show()

    # ----------------------------------------------------------------------------
    # [Code cell 20]
    # (3) Body region vs class
    ct = pd.crosstab(meta_df.body_part, meta_df.label_name).reindex(columns=["Normal", "Abnormal"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ct.plot(kind="bar", stacked=False, ax=axes[0], color=["#4C9F70", "#C1445A"], width=0.8)
    axes[0].set_title("3a. Normal / Abnormal counts per body region")
    axes[0].set_ylabel("Images"); axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=45)

    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    pct.plot(kind="barh", stacked=True, ax=axes[1], color=["#4C9F70", "#C1445A"])
    axes[1].axvline(50, ls="--", c="k", lw=0.8)
    axes[1].set_title("3b. Class proportion per body region (%)")
    axes[1].set_xlabel("% of images"); axes[1].set_ylabel("")
    plt.tight_layout(); plt.show()

    print("Regions ranked by abnormal rate:")
    print((100 * ct.Abnormal / ct.sum(axis=1)).sort_values(ascending=False).round(2).to_string())

    # ----------------------------------------------------------------------------
    # [Code cell 21]
    # (4) Image dimension distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(dim_df.width, bins=50, color="#3C6E9F"); axes[0].set_title("4a. Width (px)")
    axes[0].axvline(IMAGE_SIZE, c="r", ls="--", label=f"target {IMAGE_SIZE}"); axes[0].legend()
    axes[1].hist(dim_df.height, bins=50, color="#7A5C9E"); axes[1].set_title("4b. Height (px)")
    axes[1].axvline(IMAGE_SIZE, c="r", ls="--")
    axes[2].scatter(dim_df.width, dim_df.height, s=4, alpha=0.25, color="#2A7F7F")
    axes[2].set_xlabel("width"); axes[2].set_ylabel("height"); axes[2].set_title("4c. Width vs height")
    for ax in axes[:2]:
        ax.set_ylabel("count")
    plt.tight_layout(); plt.show()

    # ----------------------------------------------------------------------------
    # [Code cell 22]
    # (5) & (6) Example Normal and Abnormal images
    def show_examples(df, label, n=6, title=""):
        sub = df[df.label == label].sample(n=min(n, (df.label == label).sum()), random_state=SEED)
        fig, axes = plt.subplots(1, len(sub), figsize=(3 * len(sub), 3.4))
        axes = np.atleast_1d(axes)
        for ax, (_, r) in zip(axes, sub.iterrows()):
            try:
                with Image.open(r.image_path) as im:
                    ax.imshow(np.array(im.convert("L")), cmap="gray")
            except Exception as e:
                ax.text(0.5, 0.5, "unreadable", ha="center")
            ax.set_title(f"{r.body_part.replace('XR_','')}\n{r.label_name}", fontsize=9)
            ax.axis("off")
        fig.suptitle(title, fontsize=13, y=1.03)
        plt.tight_layout(); plt.show()

    show_examples(meta_df, 0, 6, "5. Example NORMAL (negative) studies")
    show_examples(meta_df, 1, 6, "6. Example ABNORMAL (positive) studies")
    print("Note: 'Abnormal' covers fractures, hardware, degenerative change, lesions and more — "
          "MURA does not label a specific pathology.")

    # ============================================================================
    # [Markdown cell 23]
    # ---
    # ## 8. Patient / study-level splitting
    # 
    # **Why not a random image split?** MURA contains multiple radiographs per study and multiple studies
    # per patient, all sharing the same label and the same anatomy. A random image-level split would place
    # near-identical views of the same body part in both train and test, and the reported test score would
    # measure memorisation rather than generalisation. The inflation is large — typically several AUC
    # points.
    # 
    # We therefore split on **`patient_id`**, which is strictly coarser than `study_uid`; grouping by
    # patient automatically guarantees study-level disjointness as well.
    # 
    # **Chosen strategy (`official_valid_as_test`, the default):** MURA ships a patient-disjoint official
    # validation set. We promote it to our **held-out test set** (touched exactly once, at the very end)
    # and carve our own validation set out of the official train patients with a grouped split.
    # 
    # Be aware of the arithmetic: MURA's official validation set is only about **8%** of all images, so
    # this strategy yields roughly a **77 / 15 / 8** train/val/test partition rather than an exact
    # 70/15/15. That is the trade-off for inheriting Stanford's curated, patient-disjoint test boundary —
    # which is what makes results comparable to published MURA numbers. If you want an exact 70/15/15
    # instead, set `SPLIT_STRATEGY = "random_70_15_15"`; it pools everything and re-splits at the patient
    # level, at the cost of discarding the official boundary. Both are leakage-free; the actual realised
    # percentages are printed below either way.
    # 
    # `GroupShuffleSplit` is used with `groups=patient_id`, so no patient can straddle two splits.

    # ----------------------------------------------------------------------------
    # [Code cell 24]
    def grouped_split(df, group_col, test_size, seed=SEED):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx_a, idx_b = next(gss.split(df, groups=df[group_col]))
        return df.iloc[idx_a].copy(), df.iloc[idx_b].copy()


    set_seed(SEED)

    if SPLIT_STRATEGY == "official_valid_as_test":
        official_train = meta_df[meta_df.orig_split == "train"].reset_index(drop=True)
        official_valid = meta_df[meta_df.orig_split == "valid"].reset_index(drop=True)

        # guard: the official valid patients must not appear in official train
        overlap = set(official_train.patient_id) & set(official_valid.patient_id)
        print(f"Patient IDs shared between official train and official valid: {len(overlap)}")
        if overlap:
            print("  -> removing those patients from the TRAIN pool to keep the test set clean.")
            official_train = official_train[~official_train.patient_id.isin(overlap)].reset_index(drop=True)

        train_df, val_df = grouped_split(official_train, "patient_id", VAL_FRACTION, SEED)
        test_df = official_valid

    elif SPLIT_STRATEGY == "random_70_15_15":
        pool = meta_df.reset_index(drop=True)
        train_df, rest = grouped_split(pool, "patient_id", VAL_FRAC + TEST_FRAC, SEED)
        rel = TEST_FRAC / (VAL_FRAC + TEST_FRAC)
        val_df, test_df = grouped_split(rest, "patient_id", rel, SEED + 1)

    else:
        raise ValueError(f"Unknown SPLIT_STRATEGY: {SPLIT_STRATEGY}")

    for d in (train_df, val_df, test_df):
        d.reset_index(drop=True, inplace=True)

    splits = {"train": train_df, "val": val_df, "test": test_df}

    print("\n" + "=" * 78)
    print(f"SPLIT SUMMARY — strategy: {SPLIT_STRATEGY}")
    print("=" * 78)
    rows = []
    total_imgs = sum(len(d) for d in splits.values())
    for name, d in splits.items():
        rows.append({
            "split": name,
            "patients": d.patient_id.nunique(),
            "studies": d.study_uid.nunique(),
            "images": len(d),
            "% images": round(100 * len(d) / total_imgs, 2),
            "normal": int((d.label == 0).sum()),
            "abnormal": int((d.label == 1).sum()),
            "% abnormal": round(100 * d.label.mean(), 2),
        })
    split_table = pd.DataFrame(rows).set_index("split")
    print(split_table.to_string())

    combined = pd.concat([d.assign(split=n) for n, d in splits.items()], ignore_index=True)
    print("\nClass distribution per split (images):")
    print(pd.crosstab(combined.split, combined.label_name).to_string())

    print("\nBody-region distribution per split (%):")
    bp_dist = pd.DataFrame({n: 100 * d.body_part.value_counts(normalize=True) for n, d in splits.items()})
    print(bp_dist.round(2).to_string())

    # ============================================================================
    # [Markdown cell 25]
    # ---
    # ## 9. Leakage verification
    # 
    # An explicit, assertive check. If any patient or study appears in two splits, the notebook stops here
    # rather than producing an optimistic-but-meaningless test score.

    # ----------------------------------------------------------------------------
    # [Code cell 26]
    def leakage_report(splits, keys=("patient_id", "study_uid", "image_path")):
        print("=" * 78)
        print("DATA LEAKAGE CHECK")
        print("=" * 78)
        names = list(splits.keys())
        all_clean = True
        for key in keys:
            print(f"\n--- key: {key} ---")
            sets = {n: set(splits[n][key]) for n in names}
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    inter = sets[a] & sets[b]
                    status = "PASS  (no overlap)" if not inter else f"FAIL  ({len(inter)} shared)"
                    print(f"  {a:<5} vs {b:<5}: {status}")
                    if inter:
                        all_clean = False
                        print("      examples:", list(inter)[:5])
            print(f"  unique {key} per split:",
                  {n: len(sets[n]) for n in names},
                  "| union:", len(set().union(*sets.values())))
        print("\n" + "=" * 78)
        print("RESULT:", "NO LEAKAGE DETECTED — splits are patient- and study-disjoint."
              if all_clean else "LEAKAGE DETECTED — do not trust downstream metrics.")
        print("=" * 78)
        return all_clean

    clean = leakage_report(splits)
    assert clean, "Leakage detected between splits — fix the split before training."

    # ============================================================================
    # [Markdown cell 27]
    # ---
    # ## 10. Dataset class
    # 
    # Loading policy for each radiograph:
    # 
    # 1. **Safe load** — every read is wrapped; an unreadable file yields a black placeholder plus a
    #    warning instead of crashing a multi-hour training run.
    # 2. **Grayscale conversion** — `convert("L")` normalises the mix of `L`, `RGB` and `RGBA` files
    #    found in MURA to a single channel, so palette/alpha artefacts cannot leak into the pixel values.
    # 3. **Resize to 300×300** — EfficientNet-B3's native resolution. Radiographs vary from roughly
    #    130 px to over 1500 px on a side; 300 px keeps cortical margins and trabecular texture legible
    #    while fitting a T4's memory budget.
    # 4. **Tensor + ImageNet normalisation.**
    # 
    # ### Why channel replication is the right adaptation here
    # 
    # The backbone's stem is a `Conv2d(3, 40, k=3, s=2)` trained on RGB. Three options exist:
    # 
    # | Option | Consequence |
    # |---|---|
    # | Average the stem weights to 1 channel | Keeps pretrained information but halves-ish the effective filter diversity and forces custom surgery on the checkpoint |
    # | Random-init a 1-channel stem | Discards the most transferable layer in the network |
    # | **Replicate grayscale to 3 channels** | The stem computes `Σ_c W_c * x`, i.e. exactly the response of the summed-across-channels kernel — the pretrained filters stay intact and the edge/texture detectors that matter most for bone imaging transfer directly |
    # 
    # Replication is chosen deliberately, not as a default: it is mathematically equivalent to a
    # channel-summed 1-channel stem while requiring zero modification to the pretrained weights. The only
    # cost is ~3× arithmetic in the very first convolution, which is negligible.
    # 
    # Consistently, ImageNet mean/std are applied **per replicated channel**, which is what the pretrained
    # statistics expect.

    # ----------------------------------------------------------------------------
    # [Code cell 28]
    # NOTE (script conversion): MURADataset is defined at true module level (above
    # `def main():`), not here, because DataLoader worker processes on Windows
    # ("spawn" start method) must be able to pickle the Dataset class by its
    # qualified name. A class nested inside a function cannot be pickled
    # ("Can't pickle local object"), which is what caused the training-loader
    # crash. See the module-level `class MURADataset(Dataset):` near the top of
    # this file.

    # ============================================================================
    # [Markdown cell 29]
    # ---
    # ## 11. Data augmentation
    # 
    # Augmentation is deliberately conservative. Radiographic abnormality cues are often subtle
    # (a cortical step-off, a thin lucent line), and aggressive geometric or photometric distortion can
    # destroy or fabricate them.
    # 
    # | Transform | Setting | Rationale |
    # |---|---|---|
    # | Rotation | ±10° | Positioning varies between technicians; small rotations are realistic |
    # | Translation | ±7% | Anatomy is not always centred in the field of view |
    # | Scale | 0.92–1.08 | Source-to-detector distance varies mildly |
    # | Horizontal flip | p = 0.5 | **Clinically acceptable here.** MURA is limb radiography and the same abnormality presents on left and right extremities; a flipped wrist is a plausible wrist. It would *not* be acceptable for chest or abdominal imaging, where situs matters |
    # | Brightness/contrast | ±15% | Exposure and windowing differ across machines |
    # 
    # Explicitly avoided: vertical flips (produce anatomically impossible limbs), large shears, elastic
    # deformation, heavy blur, cutout over the region of interest.
    # 
    # **Validation and test transforms contain no randomness whatsoever** — deterministic resize,
    # tensor conversion, normalisation. That is what makes the reported metrics meaningful.

    # ----------------------------------------------------------------------------
    # [Code cell 30]
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.07, 0.07),
            scale=(0.92, 1.08),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Deterministic — used for validation, test, threshold selection and inference.
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print("TRAIN transform:\n", train_transform)
    print("\nEVAL transform (no randomness):\n", eval_transform)

    # Visual sanity check of the augmentation strength
    _row = train_df.iloc[0]
    with Image.open(_row.image_path) as _im:
        _base = _im.convert("L").convert("RGB")
    inv = lambda t: (t * torch.tensor(IMAGENET_STD).view(3,1,1) + torch.tensor(IMAGENET_MEAN).view(3,1,1)).clamp(0,1)

    fig, axes = plt.subplots(1, 6, figsize=(16, 3))
    axes[0].imshow(_base.convert("L"), cmap="gray"); axes[0].set_title("original"); axes[0].axis("off")
    for ax in axes[1:]:
        ax.imshow(inv(train_transform(_base)).permute(1, 2, 0).numpy()[:, :, 0], cmap="gray")
        ax.set_title("augmented", fontsize=9); ax.axis("off")
    fig.suptitle(f"Augmentation preview — {_row.body_part} / {_row.label_name}", y=1.05)
    plt.tight_layout(); plt.show()

    # ============================================================================
    # [Markdown cell 31]
    # ---
    # ## 12. DataLoaders
    # 
    # T4-friendly settings: `pin_memory=True` on CUDA for faster host→device copies,
    # `persistent_workers=True` so workers are not re-spawned every epoch, a prefetch factor of 4, and
    # `non_blocking=True` transfers in the training loop (which only help when memory is pinned).

    # ----------------------------------------------------------------------------
    # [Code cell 32]
    def make_loader(df, transform, shuffle, batch_size=BATCH_SIZE, drop_last=False):
        ds = MURADataset(df, transform=transform)
        kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            drop_last=drop_last,
            worker_init_fn=seed_worker,
        )
        if NUM_WORKERS > 0:
            kwargs.update(persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR)
        if shuffle:
            kwargs["generator"] = GENERATOR
        return DataLoader(ds, **kwargs)


    train_loader = make_loader(train_df, train_transform, shuffle=True,  drop_last=True)
    val_loader   = make_loader(val_df,   eval_transform,  shuffle=False)
    test_loader  = make_loader(test_df,  eval_transform,  shuffle=False)

    print(f"train: {len(train_loader.dataset):,} images / {len(train_loader):,} batches")
    print(f"val  : {len(val_loader.dataset):,} images / {len(val_loader):,} batches")
    print(f"test : {len(test_loader.dataset):,} images / {len(test_loader):,} batches")

    xb, yb, _ = next(iter(train_loader))
    print(f"\nBatch tensor  : {tuple(xb.shape)}  dtype={xb.dtype}")
    print(f"Value range   : [{xb.min():.3f}, {xb.max():.3f}]  (normalised)")
    # Check replication BEFORE normalisation. Normalize() applies a different mean/std
    # per channel, so comparing channels of the normalised tensor always reports False
    # even when replication worked correctly -- which is what the earlier run printed.
    _undo = (xb[:1] * torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
             + torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
    print(f"Channels identical pre-normalisation (grayscale replicated): "
          f"{torch.allclose(_undo[:, 0], _undo[:, 1], atol=1e-5) and torch.allclose(_undo[:, 1], _undo[:, 2], atol=1e-5)}")
    print("  (post-normalisation channels differ by design: ImageNet mean/std are per-channel.)")
    print(f"Labels sample : {yb[:8].tolist()}")

    # ============================================================================
    # [Markdown cell 33]
    # ---
    # ## 13. Class imbalance
    # 
    # MURA is mildly imbalanced (roughly 40% abnormal at the image level, varying a lot by body region).
    # We compute `pos_weight = N_normal / N_abnormal` **from the training split only** — computing it on
    # the full dataset would leak validation/test statistics into training.
    # 
    # `BCEWithLogitsLoss(pos_weight=w)` multiplies the positive-class term of the loss by `w`, raising the
    # cost of missing an abnormal study. In a screening-flavoured task, a false negative (missed
    # abnormality) is the more costly error, so this is the right lever.

    # ----------------------------------------------------------------------------
    # [Code cell 34]
    n_train_neg = int((train_df.label == 0).sum())
    n_train_pos = int((train_df.label == 1).sum())
    pos_weight_value = n_train_neg / max(n_train_pos, 1)

    IMBALANCE_TOL = 0.15   # apply weighting only if the split deviates >15% from balanced
    apply_pos_weight = abs(train_df.label.mean() - 0.5) > IMBALANCE_TOL / 2

    print("=" * 70)
    print("CLASS IMBALANCE (training split only)")
    print("=" * 70)
    print(f"Normal   : {n_train_neg:,}  ({100*n_train_neg/len(train_df):.2f}%)")
    print(f"Abnormal : {n_train_pos:,}  ({100*n_train_pos/len(train_df):.2f}%)")
    print(f"pos_weight = N_normal / N_abnormal = {pos_weight_value:.4f}")
    print(f"Imbalance meaningful? {apply_pos_weight}")

    if apply_pos_weight:
        POS_WEIGHT = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)
        print(f"-> Using BCEWithLogitsLoss(pos_weight={pos_weight_value:.4f})")
    else:
        POS_WEIGHT = None
        print("-> Dataset is near-balanced; using unweighted BCEWithLogitsLoss.")

    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    print("\nLoss:", criterion)

    # ============================================================================
    # [Markdown cell 35]
    # ---
    # ## 14. EfficientNet-B3 model
    # 
    # ```
    # Input X-ray (3 x 300 x 300, grayscale replicated)
    #         |
    # EfficientNet-B3 backbone (features, ImageNet-pretrained)
    #         |
    # Global average pooling  ->  1536-d feature vector
    #         |
    # Dropout(p = 0.4)
    #         |
    # Linear(1536 -> 1)
    #         |
    # single logit
    #         |
    # Sigmoid — applied at INFERENCE ONLY
    # ```
    # 
    # **No sigmoid inside the model.** `BCEWithLogitsLoss` fuses the sigmoid with the loss using the
    # log-sum-exp trick, which is numerically stable; adding a separate sigmoid would double-apply it and
    # destroy the gradient. Sigmoid is applied explicitly in the evaluation and inference helpers.
    # 
    # The weights loader falls back to a random-initialised backbone if the torchvision checkpoint cannot
    # be downloaded (offline runtime), with a clear warning — a random backbone will train far worse, so
    # you want to see that message if it happens.

    # ----------------------------------------------------------------------------
    # [Code cell 36]
    class MURAEfficientNetB3(nn.Module):
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
                print("!! Falling back to RANDOM initialisation. Expect much weaker results.")
                self.backbone = efficientnet_b3(weights=None)
                self.pretrained = False

            in_features = self.backbone.classifier[1].in_features    # 1536 for B3
            self.in_features = in_features

            # Replace the 1000-class ImageNet head with a single-logit binary head.
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p, inplace=True),
                nn.Linear(in_features, 1),
            )
            nn.init.zeros_(self.backbone.classifier[1].bias)
            nn.init.normal_(self.backbone.classifier[1].weight, std=0.01)

        def forward(self, x):
            return self.backbone(x).squeeze(1)      # -> (B,) raw logits, NO sigmoid

        # ---- convenience accessors used by Grad-CAM and the freezing logic ----
        @property
        def features(self):
            return self.backbone.features

        @property
        def gradcam_target_layer(self):
            # Last conv block of the feature extractor: Conv2dNormActivation producing
            # 1536 channels at 10x10 for a 300x300 input — the deepest layer that still
            # retains spatial structure, which is exactly what Grad-CAM needs.
            return self.backbone.features[-1]


    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable


    set_seed(SEED)
    model = MURAEfficientNetB3(pretrained=True).to(DEVICE)

    print("\n" + "=" * 78)
    print("MODEL ARCHITECTURE")
    print("=" * 78)
    print(model)

    total, trainable = count_parameters(model)
    print("\n" + "=" * 78)
    print(f"Backbone pretrained on ImageNet : {model.pretrained}")
    print(f"Feature dimension               : {model.in_features}")
    print(f"Total parameters                : {total:,}")
    print(f"Trainable parameters            : {trainable:,}")
    print(f"Model size (fp32)               : {total * 4 / 1024**2:.2f} MB")
    print(f"Grad-CAM target layer           : backbone.features[-1] "
          f"({type(model.gradcam_target_layer).__name__})")
    print("=" * 78)

    # Shape check through the network
    model.eval()
    with torch.no_grad():
        _probe = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        _feat = model.features(_probe)
        _out = model(_probe)
    print(f"\nFeature map from features[-1] : {tuple(_feat.shape)}  (B, C, H, W)")
    print(f"Model output                  : {tuple(_out.shape)}  -> one logit per image")
    print(f"Sigmoid(logits) preview       : {torch.sigmoid(_out).cpu().numpy().round(4)}")
    del _probe, _feat, _out
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ============================================================================
    # [Markdown cell 37]
    # ---
    # ## 15. Training utilities
    # 
    # Includes:
    # 
    # * A version-tolerant AMP shim. Current PyTorch wants `torch.amp.autocast(device_type=...)` and
    #   `torch.amp.GradScaler(device)`; older builds only expose `torch.cuda.amp.*`. The helpers below
    #   prefer the current API and fall back silently, and everything is disabled cleanly on CPU.
    # * `run_epoch` — one training or evaluation pass, returning loss, probabilities and labels.
    # * `compute_metrics` — accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC.
    # * `EarlyStopping` + best-checkpoint bookkeeping.

    # ----------------------------------------------------------------------------
    # [Code cell 38]
    # ---------- AMP shim (dtype-aware; bfloat16 preferred on Ampere/Ada) --------
    #
    # WHY THIS MATTERS (this was the cause of the "Input contains NaN" crash):
    # float16 has a maximum representable value of 65504. EfficientNet's MBConv
    # blocks multiply wide expanded channel tensors through SiLU activations, and
    # once the backbone is unfrozen in stage 2 those activations can exceed that
    # range and become +inf. GradScaler protects the *weights* (it skips a step
    # whose gradients are inf/NaN), but it cannot protect BatchNorm's running_mean
    # and running_var, which are updated in the forward pass, outside the autograd
    # graph. A single inf activation poisons those buffers permanently. Training
    # then keeps looking healthy (train mode uses batch statistics) while every
    # eval-mode forward pass returns NaN -- exactly the failure seen at
    # stage 2 epoch 1, where validation probabilities were all NaN.
    #
    # bfloat16 has the same exponent range as float32, so those activations simply
    # cannot overflow. It is supported on Ampere and newer, which includes the
    # RTX 4060 family, and it needs no GradScaler at all. float16 + GradScaler is
    # kept as the fallback for older cards (e.g. T4), where an extra safety net in
    # run_epoch and a per-epoch finiteness check guard the same failure mode.
    def _pick_amp_dtype():
        if DEVICE.type != "cuda":
            return None
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    AMP_DTYPE = _pick_amp_dtype()
    # GradScaler is required for float16 only; bfloat16 needs no loss scaling.
    SCALER_ENABLED = bool(AMP_ENABLED and AMP_DTYPE == torch.float16)

    def make_autocast(enabled=AMP_ENABLED, dtype=None):
        dtype = dtype if dtype is not None else AMP_DTYPE
        if not enabled or dtype is None:
            import contextlib
            return contextlib.nullcontext
        try:
            return lambda: torch.amp.autocast(device_type=DEVICE.type, dtype=dtype, enabled=True)
        except (AttributeError, TypeError):
            return lambda: torch.cuda.amp.autocast(enabled=True, dtype=dtype)

    def make_scaler(enabled=None):
        enabled = SCALER_ENABLED if enabled is None else enabled
        try:
            return torch.amp.GradScaler(DEVICE.type, enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=enabled)

    autocast_ctx = make_autocast()
    _dtype_name = "disabled" if AMP_DTYPE is None else str(AMP_DTYPE).replace("torch.", "")
    print(f"AMP autocast ready | enabled: {AMP_ENABLED} | dtype: {_dtype_name} | "
          f"GradScaler: {SCALER_ENABLED}")
    if AMP_DTYPE == torch.bfloat16:
        print("  bfloat16 selected -> float16 overflow (the NaN-BatchNorm failure mode) "
              "cannot occur.")
    elif AMP_DTYPE == torch.float16:
        print("  float16 selected (GPU has no bfloat16 support) -> overflow guards in "
              "run_epoch are active.")


    # ---------- non-finite guards ------------------------------------------------
    def model_is_finite(model):
        """True if every parameter AND buffer (incl. BatchNorm running stats) is finite.

        Buffers matter as much as parameters here: BatchNorm running_mean/running_var
        are updated outside the autograd graph, so GradScaler never inspects them.
        """
        for p in model.parameters():
            if not torch.isfinite(p).all():
                return False
        for b in model.buffers():
            if b.is_floating_point() and not torch.isfinite(b).all():
                return False
        return True


    def sanitize_probs(y_prob, context=""):
        """Replace any non-finite probability with 0.5 and report it loudly."""
        y_prob = np.asarray(y_prob, dtype=float)
        bad = ~np.isfinite(y_prob)
        if bad.any():
            print(f"  [warn] {bad.sum():,}/{len(y_prob):,} non-finite predictions"
                  f"{' in ' + context if context else ''}; substituting 0.5 so metrics "
                  f"can still be computed. This signals numerical instability upstream.")
            y_prob = np.where(bad, 0.5, y_prob)
        return y_prob


    # ---------------------------- metrics ---------------------------------------
    def compute_metrics(y_true, y_prob, threshold=0.5):
        y_true = np.asarray(y_true).astype(int)
        # Never let a NaN reach sklearn -- it raises "Input contains NaN" and kills
        # a long training run. Sanitise, warn, and carry on.
        y_prob = sanitize_probs(y_prob, "compute_metrics")
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out = {
            "accuracy":    accuracy_score(y_true, y_pred),
            "precision":   precision_score(y_true, y_pred, zero_division=0),
            "recall":      recall_score(y_true, y_pred, zero_division=0),      # sensitivity
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "f1":          f1_score(y_true, y_pred, zero_division=0),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "threshold": float(threshold),
        }
        # AUCs are threshold-free; they need both classes present
        if len(np.unique(y_true)) > 1:
            out["roc_auc"] = roc_auc_score(y_true, y_prob)
            out["pr_auc"]  = average_precision_score(y_true, y_prob)
        else:
            out["roc_auc"] = float("nan"); out["pr_auc"] = float("nan")
        return out


    def fmt_metrics(m):
        return (f"acc {m['accuracy']:.4f} | prec {m['precision']:.4f} | rec {m['recall']:.4f} | "
                f"spec {m['specificity']:.4f} | f1 {m['f1']:.4f} | auc {m['roc_auc']:.4f} | "
                f"pr-auc {m['pr_auc']:.4f}")


    # ---------------------------- epoch runner ----------------------------------
    def run_epoch(model, loader, criterion, optimizer=None, scaler=None,
                  scheduler=None, desc="", grad_clip=GRAD_CLIP_NORM, eval_modules=None):
        """One pass. optimizer=None -> evaluation mode (no grad).

        eval_modules: submodules forced back into eval() after model.train(True) — used in
        stage 1 so that frozen BatchNorm layers do not keep updating their running statistics.
        """
        training = optimizer is not None
        model.train(training)
        if training and eval_modules:
            for m in eval_modules:
                m.eval()

        total_loss, n_seen = 0.0, 0
        all_probs, all_labels = [], []

        # Gradient accumulation (see GRAD_ACCUM_STEPS in the Configuration section):
        # on an 8 GB RTX 4060 the physical batch is smaller than on a 16 GB T4, so we
        # accumulate gradients over a few small batches before each optimizer step to
        # reach the same *effective* batch size without raising peak memory. With
        # GRAD_ACCUM_STEPS=1 (the default on >=10 GB cards) this is a no-op and the
        # loop behaves exactly like the original single-step version.
        accum_steps = max(1, int(GRAD_ACCUM_STEPS)) if training else 1
        n_batches = len(loader)
        n_nonfinite = 0
        if training:
            optimizer.zero_grad(set_to_none=True)

        bar = tqdm(loader, desc=desc, leave=False)
        for step_i, (xb, yb, _) in enumerate(bar, start=1):
            xb = xb.to(DEVICE, non_blocking=PIN_MEMORY)
            yb = yb.to(DEVICE, non_blocking=PIN_MEMORY)

            with torch.set_grad_enabled(training):
                with autocast_ctx():
                    logits = model(xb)
                    loss = criterion(logits, yb)

                if training:
                    # Guard: a non-finite loss means the forward pass already
                    # overflowed. Backpropagating it would write NaN into every
                    # gradient, so drop this micro-batch entirely.
                    if not torch.isfinite(loss):
                        n_nonfinite += 1
                        if n_nonfinite <= 5:
                            print(f"\n  [warn] non-finite loss at step {step_i}; "
                                  f"skipping this micro-batch.")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    scaler.scale(loss / accum_steps).backward()
                    is_boundary = (step_i % accum_steps == 0) or (step_i == n_batches)
                    if is_boundary:
                        if grad_clip:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        if scheduler is not None and isinstance(
                                scheduler, torch.optim.lr_scheduler.OneCycleLR):
                            scheduler.step()

            bs = xb.size(0)
            loss_val = loss.item()
            if math.isfinite(loss_val):
                total_loss += loss_val * bs
                n_seen += bs
            # Cast to float32 before sigmoid so a bf16/fp16 logit cannot round oddly.
            all_probs.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())
            bar.set_postfix(loss=f"{total_loss / max(n_seen,1):.4f}")

        if n_nonfinite:
            print(f"  [warn] {n_nonfinite} non-finite micro-batch(es) skipped this epoch.")

        return (total_loss / max(n_seen, 1),
                np.concatenate(all_probs),
                np.concatenate(all_labels))


    # ---------------------------- early stopping --------------------------------
    class EarlyStopping:
        def __init__(self, patience=PATIENCE, mode="max", min_delta=1e-4):
            self.patience, self.mode, self.min_delta = patience, mode, min_delta
            self.best, self.counter, self.should_stop = None, 0, False

        def improved(self, value):
            if self.best is None:
                return True
            return (value > self.best + self.min_delta) if self.mode == "max" \
                   else (value < self.best - self.min_delta)

        def step(self, value):
            if self.improved(value):
                self.best, self.counter = value, 0
                return True
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


    def freeze_backbone(model, unfreeze_from=None):
        """unfreeze_from=None -> freeze the whole feature extractor (head-only training).
           unfreeze_from=k    -> train features[k:] and the classifier."""
        for p in model.backbone.features.parameters():
            p.requires_grad = False
        if unfreeze_from is not None:
            for block in model.backbone.features[unfreeze_from:]:
                for p in block.parameters():
                    p.requires_grad = True
        for p in model.backbone.classifier.parameters():
            p.requires_grad = True
        t, tr = count_parameters(model)
        print(f"Frozen config: unfreeze_from={unfreeze_from} -> trainable {tr:,} / {t:,} "
              f"({100*tr/t:.1f}%)")
        return model


    history = {"stage": [], "epoch": [], "train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [], "val_precision": [], "val_recall": [],
               "val_f1": [], "val_roc_auc": [], "val_pr_auc": [], "lr": [], "time": []}

    def log_epoch(stage, epoch, tr_loss, tr_m, va_loss, va_m, lr, secs):
        history["stage"].append(stage);            history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss);     history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_m["accuracy"]); history["val_acc"].append(va_m["accuracy"])
        history["val_precision"].append(va_m["precision"]); history["val_recall"].append(va_m["recall"])
        history["val_f1"].append(va_m["f1"]);      history["val_roc_auc"].append(va_m["roc_auc"])
        history["val_pr_auc"].append(va_m["pr_auc"]); history["lr"].append(lr)
        history["time"].append(secs)


    def save_checkpoint(model, path, extra=None):
        payload = {
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "architecture": "torchvision.models.efficientnet_b3",
            "num_classes": 1,
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "input_channels": 3,
            "grayscale_handling": "PIL convert('L') then replicate to 3 channels",
            "dropout_p": DROPOUT_P,
            "loss": "BCEWithLogitsLoss",
            "pos_weight": float(pos_weight_value) if POS_WEIGHT is not None else None,
            "seed": SEED,
            "torch_version": torch.__version__,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    print("Training utilities ready.")

    # ============================================================================
    # [Markdown cell 39]
    # ---
    # ## 16. Stage 1 — train the classification head on a frozen backbone
    # 
    # **Why two stages?** The new head is randomly initialised. If we unfreeze the backbone immediately,
    # the large, noisy gradients coming from that random head propagate into the pretrained convolutional
    # filters and wash out exactly the representations we are trying to reuse — "catastrophic forgetting"
    # in the first few hundred steps.
    # 
    # **Stage 1** freezes every backbone parameter (`requires_grad=False`) and trains only
    # `Dropout → Linear(1536→1)`. The backbone acts as a fixed feature extractor. With ~1.5k trainable
    # parameters this converges in a couple of epochs at a comparatively large learning rate (1e-3), and
    # it is fast because no gradients flow through the 12M-parameter feature extractor.
    # 
    # Note that BatchNorm layers inside the frozen backbone are still in `train()` mode and would update
    # their running statistics; we put the frozen feature extractor into `eval()` mode each epoch so those
    # statistics stay at their ImageNet values during stage 1.

    # ----------------------------------------------------------------------------
    # [Code cell 40]
    set_seed(SEED)
    model = freeze_backbone(model, unfreeze_from=None)

    optimizer_s1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE_S1, weight_decay=WEIGHT_DECAY,
    )
    scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s1, T_max=max(NUM_EPOCHS_STAGE1, 1), eta_min=LEARNING_RATE_S1 * 0.05,
    )
    scaler = make_scaler()

    best_metric_s1 = -np.inf
    print("\n" + "=" * 78)
    print(f"STAGE 1 — head-only training | epochs={NUM_EPOCHS_STAGE1} | lr={LEARNING_RATE_S1}")
    print("=" * 78)

    for epoch in range(1, NUM_EPOCHS_STAGE1 + 1):
        t0 = time.time()
        # eval_modules keeps the frozen backbone's BatchNorm running stats fixed at their
        # ImageNet values during stage 1.
        tr_loss, tr_p, tr_y = run_epoch(model, train_loader, criterion, optimizer_s1, scaler,
                                        desc=f"S1 train {epoch}/{NUM_EPOCHS_STAGE1}",
                                        eval_modules=[model.backbone.features])
        va_loss, va_p, va_y = run_epoch(model, val_loader, criterion,
                                        desc=f"S1 val   {epoch}/{NUM_EPOCHS_STAGE1}")
        scheduler_s1.step()

        tr_m, va_m = compute_metrics(tr_y, tr_p), compute_metrics(va_y, va_p)
        secs = time.time() - t0
        lr_now = optimizer_s1.param_groups[0]["lr"]
        log_epoch(1, epoch, tr_loss, tr_m, va_loss, va_m, lr_now, secs)

        print(f"[S1 {epoch}/{NUM_EPOCHS_STAGE1}] {secs:.0f}s | lr {lr_now:.2e}")
        print(f"    train loss {tr_loss:.4f} acc {tr_m['accuracy']:.4f}")
        print(f"    val   loss {va_loss:.4f} | {fmt_metrics(va_m)}")

        score = va_m[MONITOR_METRIC] if MONITOR_METRIC != "loss" else -va_loss
        if score > best_metric_s1:
            best_metric_s1 = score
            save_checkpoint(model, CHECKPOINT_PATH,
                            extra={"stage": 1, "epoch": epoch, "val_metrics": va_m,
                                   "monitor_metric": MONITOR_METRIC, "monitor_value": float(score)})
            print(f"    -> new best ({MONITOR_METRIC} = {va_m.get(MONITOR_METRIC, score):.4f}); checkpoint saved")

    print(f"\nStage 1 complete. Best val {MONITOR_METRIC}: {best_metric_s1:.4f}")

    # ============================================================================
    # [Markdown cell 41]
    # ---
    # ## 17. Stage 2 — fine-tune the later backbone blocks
    # 
    # Now the head is calibrated, so unfreezing is safe. We unfreeze `features[5:]` — the last three of
    # EfficientNet-B3's nine feature stages plus the final conv block. Rationale:
    # 
    # * **Early blocks** (`features[0:5]`) encode edges, corners and generic texture. Those transfer
    #   essentially unchanged from ImageNet to radiography and are the layers most likely to be damaged by
    #   fine-tuning on a comparatively small medical dataset. Keeping them frozen also cuts memory and
    #   keeps a T4 comfortable.
    # * **Later blocks** encode high-level, domain-specific composition. Those genuinely need to move from
    #   "dog vs. car parts" to "cortical continuity, joint spacing, hardware".
    # 
    # Two further choices:
    # 
    # * **Discriminative learning rates.** The backbone gets `LEARNING_RATE_S2 × BACKBONE_LR_MULT`
    #   (2.5e-5), the head gets the full `LEARNING_RATE_S2` (1e-4). Pretrained weights should move in
    #   small steps; the head can keep adapting.
    # * **Scheduler.** `ReduceLROnPlateau` on the monitored validation metric — it reacts to actual
    #   validation behaviour rather than a fixed schedule, which suits a run whose length is unknown
    #   because of early stopping.
    # 
    # Early stopping (patience `PATIENCE`) and best-checkpoint saving both key off `MONITOR_METRIC`
    # (ROC-AUC by default), never off accuracy alone.

    # ----------------------------------------------------------------------------
    # [Code cell 42]
    # Resume from the best stage-1 weights
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Resumed stage-1 best checkpoint (epoch {ckpt.get('epoch')}, "
          f"{MONITOR_METRIC}={ckpt.get('monitor_value', float('nan')):.4f})")

    model = freeze_backbone(model, unfreeze_from=UNFREEZE_FROM)

    backbone_params = [p for p in model.backbone.features.parameters() if p.requires_grad]
    head_params     = [p for p in model.backbone.classifier.parameters() if p.requires_grad]
    print(f"Backbone trainable tensors: {len(backbone_params)} | head tensors: {len(head_params)}")

    optimizer_s2 = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LEARNING_RATE_S2 * BACKBONE_LR_MULT},
            {"params": head_params,     "lr": LEARNING_RATE_S2},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler_s2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_s2,
        mode="min" if MONITOR_METRIC == "loss" else "max",
        factor=0.3, patience=max(PATIENCE // 2, 1), min_lr=1e-7,
    )
    scaler = make_scaler()
    # cmp_score below is always "higher is better" (loss is negated), so mode is always "max".
    stopper = EarlyStopping(patience=PATIENCE, mode="max")

    best_metric_s2 = -np.inf
    best_epoch_s2 = None

    print("\n" + "=" * 78)
    print(f"STAGE 2 — fine-tuning features[{UNFREEZE_FROM}:] | max epochs={NUM_EPOCHS_STAGE2} | "
          f"backbone lr={LEARNING_RATE_S2*BACKBONE_LR_MULT:.1e} head lr={LEARNING_RATE_S2:.1e}")
    print(f"Early stopping on val {MONITOR_METRIC}, patience={PATIENCE}")
    print("=" * 78)

    nan_recoveries = 0
    MAX_NAN_RECOVERIES = 3

    for epoch in range(1, NUM_EPOCHS_STAGE2 + 1):
        t0 = time.time()
        tr_loss, tr_p, tr_y = run_epoch(model, train_loader, criterion, optimizer_s2, scaler,
                                        desc=f"S2 train {epoch}/{NUM_EPOCHS_STAGE2}")

        # Safety net: if any weight or BatchNorm buffer went non-finite during this
        # epoch, every subsequent eval-mode forward pass returns NaN. Roll back to the
        # last good checkpoint, cut the learning rate, and retry rather than spending
        # hours training a dead model. Under bfloat16 this branch should never fire.
        if not model_is_finite(model):
            nan_recoveries += 1
            print(f"\n  [recovery {nan_recoveries}/{MAX_NAN_RECOVERIES}] non-finite weights or "
                  f"BatchNorm buffers detected after S2 epoch {epoch}.")
            if nan_recoveries > MAX_NAN_RECOVERIES:
                print("  Too many recoveries; stopping stage 2 and keeping the last good "
                      "checkpoint.")
                break
            _ck = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(_ck["model_state_dict"])
            model.to(DEVICE)
            for gparam in optimizer_s2.param_groups:
                gparam["lr"] *= 0.25
            scaler = make_scaler()
            print(f"  Restored last best checkpoint and cut all learning rates 4x "
                  f"(head lr now {optimizer_s2.param_groups[-1]['lr']:.2e}). Retrying.")
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            continue

        va_loss, va_p, va_y = run_epoch(model, val_loader, criterion,
                                        desc=f"S2 val   {epoch}/{NUM_EPOCHS_STAGE2}")

        tr_m, va_m = compute_metrics(tr_y, tr_p), compute_metrics(va_y, va_p)
        score = va_loss if MONITOR_METRIC == "loss" else va_m[MONITOR_METRIC]
        scheduler_s2.step(score)

        secs = time.time() - t0
        lr_now = optimizer_s2.param_groups[-1]["lr"]
        log_epoch(2, epoch, tr_loss, tr_m, va_loss, va_m, lr_now, secs)

        print(f"[S2 {epoch}/{NUM_EPOCHS_STAGE2}] {secs:.0f}s | head lr {lr_now:.2e}")
        print(f"    train loss {tr_loss:.4f} acc {tr_m['accuracy']:.4f}")
        print(f"    val   loss {va_loss:.4f} | {fmt_metrics(va_m)}")

        cmp_score = -score if MONITOR_METRIC == "loss" else score
        if cmp_score > best_metric_s2:
            best_metric_s2, best_epoch_s2 = cmp_score, epoch
            save_checkpoint(model, CHECKPOINT_PATH,
                            extra={"stage": 2, "epoch": epoch, "val_metrics": va_m,
                                   "monitor_metric": MONITOR_METRIC, "monitor_value": float(score)})
            print(f"    -> new best ({MONITOR_METRIC} = {score:.4f}); checkpoint saved")

        stopper.step(cmp_score)
        if stopper.should_stop:
            print(f"\nEarly stopping at epoch {epoch}: no improvement for {PATIENCE} epochs.")
            break

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print(f"Training complete. Best epoch: stage 2 / epoch {best_epoch_s2} "
          f"({MONITOR_METRIC} = {best_metric_s2:.4f})")
    print(f"Best checkpoint: {CHECKPOINT_PATH.resolve()}")
    print("=" * 78)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print("History saved ->", HISTORY_PATH.resolve())

    # ============================================================================
    # [Markdown cell 43]
    # ---
    # ## 18. Training history plots

    # ----------------------------------------------------------------------------
    # [Code cell 44]
    hist_df = pd.DataFrame(history)
    hist_df["global_epoch"] = np.arange(1, len(hist_df) + 1)
    s2_start = hist_df.loc[hist_df.stage == 2, "global_epoch"].min() if (hist_df.stage == 2).any() else None
    print(hist_df[["stage", "epoch", "train_loss", "val_loss", "train_acc", "val_acc",
                   "val_f1", "val_roc_auc", "val_pr_auc", "lr"]].round(4).to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    x = hist_df.global_epoch

    def mark_stage(ax):
        if s2_start is not None:
            ax.axvline(s2_start - 0.5, color="gray", ls="--", lw=1)
            ax.text(s2_start - 0.45, ax.get_ylim()[1], " stage 2", va="top", fontsize=8, color="gray")

    axes[0, 0].plot(x, hist_df.train_loss, "-o", ms=4, label="train")
    axes[0, 0].plot(x, hist_df.val_loss, "-o", ms=4, label="validation")
    axes[0, 0].set_title("1. Training vs validation loss"); axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("BCE loss"); axes[0, 0].legend(); mark_stage(axes[0, 0])

    axes[0, 1].plot(x, hist_df.train_acc, "-o", ms=4, label="train")
    axes[0, 1].plot(x, hist_df.val_acc, "-o", ms=4, label="validation")
    axes[0, 1].set_title("2. Training vs validation accuracy"); axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("accuracy"); axes[0, 1].legend(); mark_stage(axes[0, 1])

    axes[1, 0].plot(x, hist_df.val_roc_auc, "-o", ms=4, color="#7A5C9E", label="ROC-AUC")
    axes[1, 0].plot(x, hist_df.val_pr_auc, "-s", ms=4, color="#2A7F7F", label="PR-AUC")
    best_i = int(np.nanargmax(hist_df.val_roc_auc))
    axes[1, 0].scatter([x.iloc[best_i]], [hist_df.val_roc_auc.iloc[best_i]], s=120,
                       facecolors="none", edgecolors="red", lw=2, label="best ROC-AUC")
    axes[1, 0].set_title("3. Validation ROC-AUC / PR-AUC"); axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend(); mark_stage(axes[1, 0])

    axes[1, 1].plot(x, hist_df.val_f1, "-o", ms=4, color="#C1445A", label="F1")
    axes[1, 1].plot(x, hist_df.val_recall, "--", lw=1, color="#4C9F70", label="recall")
    axes[1, 1].plot(x, hist_df.val_precision, "--", lw=1, color="#3C6E9F", label="precision")
    axes[1, 1].set_title("4. Validation F1 (with precision / recall)"); axes[1, 1].set_xlabel("epoch")
    axes[1, 1].legend(); mark_stage(axes[1, 1])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_history.png", bbox_inches="tight")
    plt.show()
    print("Saved ->", (OUTPUT_DIR / 'training_history.png').resolve())

    # ============================================================================
    # [Markdown cell 45]
    # ---
    # ## 19. Threshold selection — validation set only
    # 
    # `0.5` is the default decision boundary for a sigmoid, but it is **not** a medically meaningful
    # threshold. It is optimal only when classes are balanced, the model is calibrated, and false
    # positives and false negatives cost the same — none of which holds here.
    # 
    # We sweep thresholds over the **validation** predictions and pick one by an explicit criterion:
    # 
    # | Criterion | Picks | Use when |
    # |---|---|---|
    # | `f1` (default) | Threshold maximising F1 | Balanced view of precision and recall |
    # | `youden` | Maximises `sensitivity + specificity − 1` | Equal weight on both error types, prevalence-independent |
    # | `recall_at` | Highest threshold that still reaches `TARGET_RECALL` | Screening/triage: missing an abnormality is worse than a false alarm |
    # 
    # The chosen threshold is then **locked** and reused unchanged for the test set and for all inference.
    # Tuning a threshold on the test set would turn the held-out set into a second validation set and
    # inflate every number that follows it — so the test set is loaded only *after* this cell.

    # ----------------------------------------------------------------------------
    # [Code cell 46]
    # ---- Load the BEST checkpoint (not the final epoch) ------------------------
    best_ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"Loaded BEST checkpoint: stage {best_ckpt.get('stage')} epoch {best_ckpt.get('epoch')} | "
          f"val {best_ckpt.get('monitor_metric')} = {best_ckpt.get('monitor_value', float('nan')):.4f}")

    @torch.no_grad()
    def predict_loader(model, loader, desc="predict"):
        model.eval()
        probs, labels = [], []
        for xb, yb, _ in tqdm(loader, desc=desc, leave=False):
            xb = xb.to(DEVICE, non_blocking=PIN_MEMORY)
            with autocast_ctx():
                logits = model(xb)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            labels.append(yb.numpy())
        return np.concatenate(probs), np.concatenate(labels)

    val_probs, val_labels = predict_loader(model, val_loader, "Validation inference")
    print(f"Validation predictions: {len(val_probs):,} | mean abnormal prob {val_probs.mean():.4f}")

    # ----------------------------------------------------------------------------
    # [Code cell 47]
    def select_threshold(y_true, y_prob, criterion=THRESHOLD_CRITERION, target_recall=TARGET_RECALL):
        grid = np.linspace(0.01, 0.99, 197)
        rows = []
        for t in grid:
            m = compute_metrics(y_true, y_prob, t)
            rows.append({"threshold": t, "f1": m["f1"], "precision": m["precision"],
                         "recall": m["recall"], "specificity": m["specificity"],
                         "accuracy": m["accuracy"], "youden": m["recall"] + m["specificity"] - 1})
        sweep = pd.DataFrame(rows)

        if criterion == "f1":
            best_t = float(sweep.loc[sweep.f1.idxmax(), "threshold"])
            reason = "maximises F1 on the validation set (balances precision and recall)"
        elif criterion == "youden":
            best_t = float(sweep.loc[sweep.youden.idxmax(), "threshold"])
            reason = "maximises Youden's J = sensitivity + specificity - 1 (prevalence-independent)"
        elif criterion == "recall_at":
            ok = sweep[sweep.recall >= target_recall]
            best_t = float(ok.threshold.max()) if len(ok) else float(sweep.threshold.min())
            reason = (f"highest threshold still achieving recall >= {target_recall:.0%} "
                      f"(screening-oriented: prioritises not missing abnormal studies)")
        else:
            raise ValueError(criterion)
        return best_t, reason, sweep


    SELECTED_THRESHOLD, THRESHOLD_REASON, sweep = select_threshold(val_labels, val_probs)

    m_default  = compute_metrics(val_labels, val_probs, 0.50)
    m_selected = compute_metrics(val_labels, val_probs, SELECTED_THRESHOLD)

    print("=" * 78)
    print("THRESHOLD SELECTION  (validation set only — test set untouched)")
    print("=" * 78)
    print(f"Criterion          : {THRESHOLD_CRITERION}")
    print(f"Rationale          : {THRESHOLD_REASON}")
    print(f"Selected threshold : {SELECTED_THRESHOLD:.4f}")
    print()
    print(f"  @ 0.5000 (default) : {fmt_metrics(m_default)}")
    print(f"  @ {SELECTED_THRESHOLD:.4f} (chosen)  : {fmt_metrics(m_selected)}")
    print()
    print(f"Delta F1     : {m_selected['f1'] - m_default['f1']:+.4f}")
    print(f"Delta recall : {m_selected['recall'] - m_default['recall']:+.4f}")
    print(f"Delta spec.  : {m_selected['specificity'] - m_default['specificity']:+.4f}")
    print("\nTHRESHOLD IS NOW LOCKED. It will not be re-tuned on the test set.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for col, c in [("f1", "#C1445A"), ("precision", "#3C6E9F"),
                   ("recall", "#4C9F70"), ("specificity", "#7A5C9E")]:
        axes[0].plot(sweep.threshold, sweep[col], label=col, color=c)
    axes[0].axvline(SELECTED_THRESHOLD, color="k", ls="--",
                    label=f"selected {SELECTED_THRESHOLD:.3f}")
    axes[0].axvline(0.5, color="gray", ls=":", label="default 0.5")
    axes[0].set_xlabel("threshold"); axes[0].set_title("Validation metrics vs threshold")
    axes[0].legend(fontsize=8)

    axes[1].hist(val_probs[val_labels == 0], bins=50, alpha=0.65, label="Normal", color="#4C9F70")
    axes[1].hist(val_probs[val_labels == 1], bins=50, alpha=0.65, label="Abnormal", color="#C1445A")
    axes[1].axvline(SELECTED_THRESHOLD, color="k", ls="--")
    axes[1].set_xlabel("predicted abnormal probability")
    axes[1].set_title("Validation probability distribution by true class"); axes[1].legend()
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "threshold_selection.png", bbox_inches="tight"); plt.show()

    # ============================================================================
    # [Markdown cell 48]
    # ---
    # ## 20. Test evaluation — held-out set, locked threshold
    # 
    # ### What each metric means for Normal vs. Abnormal
    # 
    # * **Accuracy** — fraction of radiographs classified correctly. Least informative metric here,
    #   because a model that always predicted "Normal" would still score around 60%.
    # * **Precision (PPV)** — of the studies flagged abnormal, how many really were. Low precision means a
    #   reader wastes time on false alarms.
    # * **Recall / Sensitivity** — of the truly abnormal studies, how many were caught. **The safety-
    #   critical metric**: a false negative is a missed abnormality.
    # * **Specificity (TNR)** — of the truly normal studies, how many were correctly cleared.
    # * **F1** — harmonic mean of precision and recall; a single number that penalises ignoring either.
    # * **ROC-AUC** — probability that a random abnormal study is scored higher than a random normal one.
    #   Threshold-free measure of ranking quality.
    # * **PR-AUC (average precision)** — area under the precision–recall curve. More informative than
    #   ROC-AUC when the positive class is the minority, because it ignores true negatives.
    # * **Confusion matrix** — the raw TP / TN / FP / FN counts behind every number above.

    # ----------------------------------------------------------------------------
    # [Code cell 49]
    test_probs, test_labels = predict_loader(model, test_loader, "Test inference")
    test_metrics = compute_metrics(test_labels, test_probs, SELECTED_THRESHOLD)
    test_metrics_05 = compute_metrics(test_labels, test_probs, 0.5)
    test_preds = (test_probs >= SELECTED_THRESHOLD).astype(int)

    print("=" * 78)
    print(f"TEST SET EVALUATION  |  n = {len(test_labels):,} images  |  "
          f"threshold = {SELECTED_THRESHOLD:.4f} (locked on validation)")
    print("=" * 78)
    print(f"Accuracy              : {test_metrics['accuracy']:.4f}")
    print(f"Precision (PPV)       : {test_metrics['precision']:.4f}")
    print(f"Recall / Sensitivity  : {test_metrics['recall']:.4f}")
    print(f"Specificity (TNR)     : {test_metrics['specificity']:.4f}")
    print(f"F1-score              : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC               : {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC (avg precision): {test_metrics['pr_auc']:.4f}")
    print()
    print("Confusion matrix counts:")
    print(f"  True  Positives (abnormal caught)     : {test_metrics['tp']:,}")
    print(f"  True  Negatives (normal cleared)      : {test_metrics['tn']:,}")
    print(f"  False Positives (normal flagged)      : {test_metrics['fp']:,}")
    print(f"  False Negatives (abnormal MISSED)     : {test_metrics['fn']:,}")
    print()
    npv = test_metrics['tn'] / max(test_metrics['tn'] + test_metrics['fn'], 1)
    print(f"  Negative predictive value             : {npv:.4f}")
    print(f"  Miss rate (FN / all abnormal)         : "
          f"{test_metrics['fn'] / max(test_metrics['tp'] + test_metrics['fn'], 1):.4f}")
    print()
    print(f"For reference, at the naive 0.5 threshold: {fmt_metrics(test_metrics_05)}")
    print()
    print("Per-class report:")
    print(classification_report(test_labels, test_preds, target_names=["Normal", "Abnormal"], digits=4))

    # Per-body-region breakdown
    tdf = test_df.copy()
    tdf["prob"] = test_probs
    tdf["pred"] = test_preds
    print("\nTest ROC-AUC per body region:")
    for bp, g in tdf.groupby("body_part"):
        if g.label.nunique() > 1:
            print(f"  {bp:<12} n={len(g):>5,}  auc={roc_auc_score(g.label, g.prob):.4f}  "
                  f"acc={accuracy_score(g.label, g.pred):.4f}")

    # ----------------------------------------------------------------------------
    # [Code cell 50]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (1) Confusion matrix
    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Pred Normal", "Pred Abnormal"],
                yticklabels=["True Normal", "True Abnormal"], annot_kws={"size": 13})
    axes[0].set_title(f"1. Confusion matrix @ {SELECTED_THRESHOLD:.3f}")
    for (i, j), lab in [((0,0),"TN"), ((0,1),"FP"), ((1,0),"FN"), ((1,1),"TP")]:
        axes[0].text(j + 0.5, i + 0.78, lab, ha="center", fontsize=9, color="gray")

    # (2) ROC curve
    fpr, tpr, roc_thr = roc_curve(test_labels, test_probs)
    axes[1].plot(fpr, tpr, lw=2, color="#3C6E9F", label=f"AUC = {test_metrics['roc_auc']:.4f}")
    axes[1].plot([0, 1], [0, 1], "--", c="gray", lw=1, label="chance")
    op = np.argmin(np.abs(roc_thr - SELECTED_THRESHOLD))
    axes[1].scatter(fpr[op], tpr[op], s=110, facecolors="none", edgecolors="red", lw=2,
                    label=f"operating point ({SELECTED_THRESHOLD:.3f})")
    axes[1].set_xlabel("False positive rate (1 - specificity)")
    axes[1].set_ylabel("True positive rate (sensitivity)")
    axes[1].set_title("2. ROC curve — test set"); axes[1].legend(loc="lower right", fontsize=9)

    # (3) Precision-Recall curve
    prec, rec, pr_thr = precision_recall_curve(test_labels, test_probs)
    baseline = test_labels.mean()
    axes[2].plot(rec, prec, lw=2, color="#C1445A", label=f"PR-AUC = {test_metrics['pr_auc']:.4f}")
    axes[2].axhline(baseline, ls="--", c="gray", lw=1, label=f"prevalence = {baseline:.3f}")
    op2 = np.argmin(np.abs(pr_thr - SELECTED_THRESHOLD))
    axes[2].scatter(rec[op2], prec[op2], s=110, facecolors="none", edgecolors="red", lw=2,
                    label="operating point")
    axes[2].set_xlabel("Recall (sensitivity)"); axes[2].set_ylabel("Precision (PPV)")
    axes[2].set_title("3. Precision-Recall curve — test set"); axes[2].legend(loc="lower left", fontsize=9)

    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "test_evaluation.png", bbox_inches="tight"); plt.show()

    with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump({"threshold": SELECTED_THRESHOLD,
                   "threshold_criterion": THRESHOLD_CRITERION,
                   "test_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                    for k, v in test_metrics.items()},
                   "n_test_images": int(len(test_labels))}, f, indent=2)
    print("Saved ->", (OUTPUT_DIR / "test_metrics.json").resolve())

    # ============================================================================
    # [Markdown cell 51]
    # ---
    # ## 21. Grad-CAM for EfficientNet-B3
    # 
    # **Target layer.** Grad-CAM needs the deepest layer that still carries spatial structure. For
    # torchvision's EfficientNet-B3 that is `backbone.features[-1]`, the final
    # `Conv2dNormActivation` (1536 channels). At a 300×300 input it outputs a **10×10** map, which is
    # after all nine MBConv stages — semantically rich — but before global average pooling, which would
    # destroy the spatial axes. Going shallower gives finer resolution but weaker semantics; going deeper
    # means no spatial axes exist at all.
    # 
    # **Method.** With `A^k` the k-th activation channel and `y` the abnormal logit:
    # 
    # ```
    # alpha_k = GAP( dy / dA^k )                # gradient-derived channel importance
    # L       = ReLU( sum_k alpha_k * A^k )     # weighted channel sum, negatives dropped
    # ```
    # 
    # ReLU keeps only evidence *supporting* the class. The map is then min-max normalised and bilinearly
    # upsampled to the original radiograph's dimensions.
    # 
    # **Implementation notes.** Grad-CAM needs gradients, so it runs with `autocast` disabled in fp32 —
    # half-precision gradients through the hook can underflow to zero and produce a blank map. Hooks are
    # registered and removed via a context manager so repeated calls do not accumulate handles or leak GPU
    # memory.
    # 
    # > **Interpretation limit.** Grad-CAM shows where the network's activations were most influential for
    # > its own output. It is a **model-interpretability tool, not a lesion segmentation**. A highlighted
    # > region is a model-attributed region of importance — it may fall on genuine pathology, on
    # > positioning artefacts, on hardware, on image borders, or on spurious correlations. It must never be
    # > read as a claim that pathology is present at that location.

    # ----------------------------------------------------------------------------
    # [Code cell 52]
    class GradCAM:
        """Grad-CAM for a single-logit binary model. Device-agnostic, fp32, hook-safe."""

        def __init__(self, model, target_layer=None):
            self.model = model
            self.target_layer = target_layer if target_layer is not None else model.gradcam_target_layer
            self.activations = None
            self.gradients = None
            self.handles = []

        def _fwd_hook(self, module, inp, out):
            self.activations = out.detach()

        def _bwd_hook(self, module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        def __enter__(self):
            self.handles.append(self.target_layer.register_forward_hook(self._fwd_hook))
            # full_backward_hook is the non-deprecated API
            try:
                self.handles.append(self.target_layer.register_full_backward_hook(self._bwd_hook))
            except AttributeError:
                self.handles.append(self.target_layer.register_backward_hook(self._bwd_hook))
            return self

        def __exit__(self, *exc):
            self.remove_hooks()
            return False

        def remove_hooks(self):
            for h in self.handles:
                h.remove()
            self.handles = []

        def generate(self, input_tensor, output_size=None):
            """input_tensor: (1,3,H,W) on any device. Returns (cam HxW in [0,1], logit, prob)."""
            self.model.eval()
            input_tensor = input_tensor.to(DEVICE).float().requires_grad_(True)

            # fp32 only: AMP gradients through the hooked layer can underflow.
            with torch.enable_grad():
                logit = self.model(input_tensor)              # (1,)
                score = logit.sum()
                self.model.zero_grad(set_to_none=True)
                score.backward()

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks captured nothing — check the target layer.")

            acts = self.activations[0]                        # (C, h, w)
            grads = self.gradients[0]                         # (C, h, w)
            alpha = grads.mean(dim=(1, 2), keepdim=True)      # (C,1,1) channel importance
            cam = F.relu((alpha * acts).sum(dim=0))           # (h, w)

            cam = cam - cam.min()
            denom = cam.max()
            cam = cam / denom if denom > 0 else torch.zeros_like(cam)

            size = output_size or (input_tensor.shape[-2], input_tensor.shape[-1])
            cam = F.interpolate(cam[None, None], size=size, mode="bilinear",
                                align_corners=False)[0, 0]

            prob = torch.sigmoid(logit.detach()).item()
            return cam.detach().cpu().numpy(), float(logit.detach().item()), prob


    def overlay_heatmap(original_gray, cam, alpha=0.42, cmap="jet"):
        """Blend a [0,1] CAM over a grayscale uint8 image. Returns float RGB in [0,1]."""
        gray = np.asarray(original_gray, dtype=np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
        rgb = np.stack([gray] * 3, axis=-1)
        heat = matplotlib.colormaps[cmap](np.clip(cam, 0, 1))[..., :3]
        return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)


    # ---- quick smoke test on one validation image ------------------------------
    _row = val_df.iloc[0]
    with Image.open(_row.image_path) as _im:
        _gray = _im.convert("L")
    _tensor = eval_transform(_gray.convert("RGB")).unsqueeze(0)

    with GradCAM(model) as cam_engine:
        _cam, _logit, _prob = cam_engine.generate(_tensor, output_size=(_gray.height, _gray.width))

    print("Grad-CAM smoke test")
    print(f"  target layer      : backbone.features[-1] ({type(model.gradcam_target_layer).__name__})")
    print(f"  raw CAM upsampled : {_cam.shape}  (matches original {(_gray.height, _gray.width)})")
    print(f"  CAM range         : [{_cam.min():.3f}, {_cam.max():.3f}]")
    print(f"  logit / prob      : {_logit:.4f} / {_prob:.4f}   true label = {_row.label_name}")

    # ============================================================================
    # [Markdown cell 53]
    # ---
    # ## 22. Model saving and loading
    # 
    # The checkpoint is self-contained: weights plus everything needed to reproduce inference without
    # retraining — class map, image size, normalisation constants, the locked decision threshold,
    # architecture identifier, and provenance metadata. `load_model_for_inference` rebuilds the model from
    # it and returns the configuration alongside.

    # ----------------------------------------------------------------------------
    # [Code cell 54]
    FINAL_CHECKPOINT = CHECKPOINT_DIR / "mura_efficientnet_b3_best.pth"

    save_checkpoint(
        model, FINAL_CHECKPOINT,
        extra={
            "stage": best_ckpt.get("stage"),
            "epoch": best_ckpt.get("epoch"),
            "threshold": float(SELECTED_THRESHOLD),
            "threshold_criterion": THRESHOLD_CRITERION,
            "threshold_rationale": THRESHOLD_REASON,
            "val_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in m_selected.items()},
            "test_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                             for k, v in test_metrics.items()},
            "split_strategy": SPLIT_STRATEGY,
            "n_train_images": int(len(train_df)),
            "n_val_images": int(len(val_df)),
            "n_test_images": int(len(test_df)),
            "body_parts": sorted(meta_df.body_part.unique().tolist()),
            "task": "binary classification: 0=Normal, 1=Abnormal (NOT fracture-specific)",
            "disclaimer": ("Research/educational model. Not a medical device. Outputs are model "
                           "predictions, not clinical diagnoses."),
        },
    )
    print("Checkpoint saved ->", FINAL_CHECKPOINT.resolve())
    print("Size: {:.1f} MB".format(FINAL_CHECKPOINT.stat().st_size / 1024**2))


    def load_model_for_inference(checkpoint_path=FINAL_CHECKPOINT, device=DEVICE):
        """Rebuild the trained model from a checkpoint. No retraining required."""
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        m = MURAEfficientNetB3(pretrained=False, dropout_p=ck.get("dropout_p", DROPOUT_P))
        m.load_state_dict(ck["model_state_dict"])
        m.to(device).eval()
        cfg = {
            "image_size": ck["image_size"],
            "mean": ck["normalization"]["mean"],
            "std": ck["normalization"]["std"],
            "threshold": ck.get("threshold", 0.5),
            "class_names": ck.get("class_names", CLASS_NAMES),
            "model_name": ck.get("model_name", MODEL_NAME),
        }
        return m, cfg


    # Verify the round-trip
    _m, _cfg = load_model_for_inference()
    print("\nReloaded model config:")
    for k, v in _cfg.items():
        print(f"  {k:<12}: {v}")
    with torch.no_grad():
        _a = model(_tensor.to(DEVICE)).item()
        _b = _m(_tensor.to(DEVICE)).item()
    print(f"\nRound-trip check | original logit {_a:.6f} vs reloaded {_b:.6f} | "
          f"match: {abs(_a - _b) < 1e-4}")
    del _m

    # ============================================================================
    # [Markdown cell 55]
    # ---
    # ## 23. Single-image inference — `predict_xray`
    # 
    # Takes a path to any X-ray, preserves the original image untouched, preprocesses a **copy** for the
    # model, runs on GPU when available, applies the **locked validation threshold**, and renders the
    # required panel:
    # 
    # * **Normal prediction** → original + probabilities (Grad-CAM optional, clearly labelled as model
    #   attention).
    # * **Abnormal prediction** → original + probabilities + Grad-CAM + overlay.
    # 
    # The original input image is always displayed and never overwritten on disk.

    # ----------------------------------------------------------------------------
    # [Code cell 56]
    # NOTE (script conversion): in the original notebook this cell relied on
    # `globals()["model"]` to fall back to the trained model when `model=None`.
    # That only works at true module/notebook top level. Running as a plain .py
    # script, the trained model lives as a local inside main(), so we capture an
    # explicit reference here for the default-argument fallback below to close over.
    _ACTIVE_MODEL = model

    def predict_xray(
        image_path,
        model=None,
        threshold=None,
        generate_gradcam=True,
        show=True,
        save_dir=None,
        save_prefix=None,
        cam_alpha=0.42,
        verbose=True,
    ):
        """Run the full inference + explainability pipeline on one X-ray.

        Returns a dict with the prediction, both class probabilities, confidence,
        the threshold used, and paths to any saved artefacts.
        """
        model = model if model is not None else _ACTIVE_MODEL
        threshold = float(threshold if threshold is not None else SELECTED_THRESHOLD)
        image_path = str(image_path)

        # 1) Load the user's X-ray and PRESERVE the original (never modified, never overwritten)
        with Image.open(image_path) as im:
            original_pil = im.convert("L").copy()
        original_np = np.array(original_pil)
        orig_h, orig_w = original_np.shape

        # 2) Preprocess a COPY for the model (deterministic eval transform)
        input_tensor = eval_transform(original_pil.convert("RGB")).unsqueeze(0)

        # 3) Forward pass on the selected device
        model.eval()
        with torch.no_grad():
            logit = model(input_tensor.to(DEVICE, non_blocking=PIN_MEMORY)).float()
            p_abnormal = torch.sigmoid(logit).item()
        p_normal = 1.0 - p_abnormal

        # 4) Apply the LOCKED validation threshold
        pred_idx = int(p_abnormal >= threshold)
        pred_label = CLASS_NAMES[pred_idx]
        confidence = p_abnormal if pred_idx == 1 else p_normal

        result = {
            "image_path": os.path.abspath(image_path),
            "prediction": pred_label,
            "predicted_class": pred_idx,
            "normal_probability": round(p_normal * 100, 2),
            "abnormal_probability": round(p_abnormal * 100, 2),
            "confidence": round(confidence * 100, 2),
            "raw_logit": round(float(logit.item()), 6),
            "threshold": round(threshold, 6),
            "model_name": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "original_dimensions": [int(orig_w), int(orig_h)],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "disclaimer": ("Research/educational model output. Not a clinical diagnosis. "
                           "Grad-CAM shows model-attributed importance, not lesion segmentation."),
        }

        if verbose:
            print("=" * 62)
            print(f"Image                : {os.path.basename(image_path)}  ({orig_w}x{orig_h})")
            print(f"Model prediction     : {pred_label.upper()}")
            print(f"Normal probability   : {result['normal_probability']:.2f}%")
            print(f"Abnormal probability : {result['abnormal_probability']:.2f}%")
            print(f"Confidence           : {result['confidence']:.2f}%")
            print(f"Decision threshold   : {threshold:.4f}  ({THRESHOLD_CRITERION}, from validation)")
            print("=" * 62)

        # 5) Grad-CAM (at the ORIGINAL resolution)
        cam, overlay = None, None
        if generate_gradcam:
            with GradCAM(model) as engine:
                cam, _, _ = engine.generate(input_tensor, output_size=(orig_h, orig_w))
            overlay = overlay_heatmap(original_np, cam, alpha=cam_alpha)

        # 6) Save artefacts (the original input file itself is never touched)
        if save_dir is not None:
            save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
            stem = save_prefix or Path(image_path).stem
            p_orig = save_dir / f"{stem}_original_xray.png"
            plt.imsave(p_orig, original_np, cmap="gray")
            result["saved_original"] = str(p_orig)
            if cam is not None:
                p_cam = save_dir / f"{stem}_gradcam.png"
                p_ov  = save_dir / f"{stem}_gradcam_overlay.png"
                plt.imsave(p_cam, cam, cmap="jet")
                plt.imsave(p_ov, overlay)
                result["saved_gradcam"] = str(p_cam)
                result["saved_overlay"] = str(p_ov)
            p_json = save_dir / f"{stem}_prediction.json"
            with open(p_json, "w") as f:
                json.dump(result, f, indent=2)
            result["saved_json"] = str(p_json)
            if verbose:
                print("Artefacts saved to:", save_dir.resolve())

        # 7) Visualise
        if show:
            is_abnormal = pred_idx == 1
            n_panels = 3 if (generate_gradcam and cam is not None) else 1
            fig, axes = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 6.2))
            axes = np.atleast_1d(axes)

            # Panel 1 — THE ORIGINAL INPUT X-RAY (always shown, never replaced)
            axes[0].imshow(original_np, cmap="gray")
            axes[0].set_title("ORIGINAL X-RAY (model input)", fontsize=12, fontweight="bold")
            axes[0].axis("off")

            colour = "#C1445A" if is_abnormal else "#2E7D4F"
            if is_abnormal:
                info = (f"Model Prediction: ABNORMAL\n\n"
                        f"Abnormal probability : {result['abnormal_probability']:.2f}%\n"
                        f"Normal probability   : {result['normal_probability']:.2f}%\n"
                        f"Confidence           : {result['confidence']:.2f}%\n"
                        f"Threshold            : {threshold:.4f}")
            else:
                info = (f"Model Prediction: NORMAL\n\n"
                        f"Normal probability   : {result['normal_probability']:.2f}%\n"
                        f"Abnormal probability : {result['abnormal_probability']:.2f}%\n"
                        f"Confidence           : {result['confidence']:.2f}%\n"
                        f"Threshold            : {threshold:.4f}")
            axes[0].text(0.5, -0.015, info, transform=axes[0].transAxes, ha="center", va="top",
                         fontsize=10.5, family="monospace", color="white",
                         bbox=dict(boxstyle="round,pad=0.6", facecolor=colour, alpha=0.92))

            if n_panels == 3:
                axes[1].imshow(cam, cmap="jet")
                axes[1].set_title("GRAD-CAM\n(model-attributed importance)", fontsize=12,
                                  fontweight="bold")
                axes[1].axis("off")

                axes[2].imshow(overlay)
                axes[2].set_title("ORIGINAL + GRAD-CAM OVERLAY\n(model attention, NOT lesion "
                                  "segmentation)", fontsize=12, fontweight="bold")
                axes[2].axis("off")

                note = ("Warm colours = regions that contributed most strongly to the model's "
                        f"{'ABNORMAL' if is_abnormal else 'NORMAL'} output. "
                        "Research visualisation only — not a clinical diagnosis and not "
                        "definitive lesion localisation.")
                fig.text(0.5, -0.035, note, ha="center", fontsize=9.5, style="italic", wrap=True)

            fig.suptitle(f"MURA EfficientNet-B3 — model prediction: {pred_label.upper()}",
                         fontsize=14, fontweight="bold", y=1.0)
            plt.tight_layout()
            plt.show()

        return result


    print("predict_xray() ready.")

    # ----------------------------------------------------------------------------
    # [Code cell 57]
    # ---- Demonstrate both branches: a NORMAL case and an ABNORMAL case ---------
    norm_row = test_df[test_df.label == 0].sample(1, random_state=SEED).iloc[0]
    print(f"[Ground truth: {norm_row.label_name} | {norm_row.body_part}]")
    res_norm = predict_xray(norm_row.image_path, generate_gradcam=True,
                            save_dir=OUTPUT_DIR / "inference")

    # ----------------------------------------------------------------------------
    # [Code cell 58]
    abn_row = test_df[test_df.label == 1].sample(1, random_state=SEED).iloc[0]
    print(f"[Ground truth: {abn_row.label_name} | {abn_row.body_part}]")
    res_abn = predict_xray(abn_row.image_path, generate_gradcam=True,
                           save_dir=OUTPUT_DIR / "inference")
    print("\nReturned dictionary:")
    print(json.dumps({k: v for k, v in res_abn.items() if not k.startswith("saved")}, indent=2))

    # ============================================================================
    # [Markdown cell 59]
    # ---
    # ## 24. Batch inference — `predict_folder`
    # 
    # Recursively finds valid images in a folder, batches them through the model, and writes a CSV with
    # `image_path, prediction, normal_probability, abnormal_probability, confidence`.
    # 
    # Grad-CAM is **off by default** (`generate_gradcam=False`): it requires a backward pass per image and
    # cannot be batched efficiently, so generating it for thousands of files would be slow and memory
    # hungry. When enabled, `gradcam_limit` caps how many images get one.

    # ----------------------------------------------------------------------------
    # [Code cell 60]
    # NOTE (script conversion): _InferenceDataset is defined at true module level
    # (above `def main():`) for the same pickling reason as MURADataset above.



    def predict_folder(
        folder_path,
        model=None,
        threshold=None,
        output_csv=None,
        generate_gradcam=False,
        gradcam_limit=20,
        gradcam_dir=None,
        batch_size=None,
        recursive=True,
        verbose=True,
    ):
        """Run inference over every valid X-ray in a folder and write results to CSV."""
        model = model if model is not None else _ACTIVE_MODEL
        threshold = float(threshold if threshold is not None else SELECTED_THRESHOLD)
        batch_size = batch_size or BATCH_SIZE
        folder_path = Path(folder_path)

        pattern = "**/*" if recursive else "*"
        paths = sorted(str(p) for p in folder_path.glob(pattern)
                       if p.is_file() and not p.name.startswith("._")
                       and p.suffix.lower() in VALID_EXT)
        if not paths:
            print(f"No images with extensions {sorted(VALID_EXT)} found under {folder_path}")
            return pd.DataFrame()

        if verbose:
            print(f"Found {len(paths):,} image(s) under {folder_path}")

        loader = DataLoader(
            _InferenceDataset(paths, eval_transform),
            batch_size=batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            **({"persistent_workers": False} if NUM_WORKERS > 0 else {}),
        )

        model.eval()
        probs = np.zeros(len(paths), dtype=np.float32)
        with torch.no_grad():
            for xb, idx in tqdm(loader, desc="Batch inference", disable=not verbose):
                xb = xb.to(DEVICE, non_blocking=PIN_MEMORY)
                with autocast_ctx():
                    logits = model(xb)
                probs[idx.numpy()] = torch.sigmoid(logits.float()).cpu().numpy()

        preds = (probs >= threshold).astype(int)
        out = pd.DataFrame({
            "image_path": paths,
            "prediction": [CLASS_NAMES[p] for p in preds],
            "normal_probability": np.round((1 - probs) * 100, 2),
            "abnormal_probability": np.round(probs * 100, 2),
            "confidence": np.round(np.where(preds == 1, probs, 1 - probs) * 100, 2),
        })

        output_csv = Path(output_csv) if output_csv else OUTPUT_DIR / "batch_predictions.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)

        if verbose:
            print(f"\nResults -> {output_csv.resolve()}")
            print(out.prediction.value_counts().to_string())
            print(f"Mean abnormal probability: {out.abnormal_probability.mean():.2f}%")

        if generate_gradcam:
            gradcam_dir = Path(gradcam_dir or (OUTPUT_DIR / "batch_gradcam"))
            gradcam_dir.mkdir(parents=True, exist_ok=True)
            # Prioritise the most confident abnormal predictions
            order = out[out.prediction == "Abnormal"].sort_values(
                "abnormal_probability", ascending=False).index[:gradcam_limit]
            if verbose:
                print(f"\nGenerating Grad-CAM for {len(order)} image(s) "
                      f"(limit={gradcam_limit}) -> {gradcam_dir.resolve()}")
            for i in tqdm(order, desc="Grad-CAM", disable=not verbose):
                predict_xray(out.image_path[i], model=model, threshold=threshold,
                             generate_gradcam=True, show=False,
                             save_dir=gradcam_dir, verbose=False)

        return out


    # Demo on a small slice of the test set
    demo_folder = Path(test_df.iloc[0].image_path).parent
    batch_results = predict_folder(demo_folder, generate_gradcam=False,
                                   output_csv=OUTPUT_DIR / "batch_predictions.csv")
    batch_results.head(10)

    # ============================================================================
    # [Markdown cell 61]
    # ---
    # ## 25. Final end-to-end demonstration
    # 
    # One X-ray taken from the dataset, pushed through the complete pipeline. The output answers, at a
    # glance:
    # 
    # 1. **What image was given to the model?** — the original radiograph, panel 1.
    # 2. **What did the model predict?** — the label banner under panel 1.
    # 3. **How confident was the model?** — both class probabilities and the confidence figure.
    # 4. **Which region contributed most?** — the Grad-CAM heatmap (panel 2) and the overlay (panel 3).

    # ----------------------------------------------------------------------------
    # [Code cell 62]
    print("=" * 78)
    print("RUN CONFIGURATION SUMMARY")
    print("=" * 78)
    print(f"Random seed              : {SEED}")
    print(f"PyTorch version          : {torch.__version__}")
    print(f"CUDA available           : {torch.cuda.is_available()}")
    print(f"GPU name                 : "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A (CPU)'}")
    print(f"Device in use            : {DEVICE}")
    print(f"Dataset path             : {MURA_ROOT.resolve()}")
    print(f"Split strategy           : {SPLIT_STRATEGY}")
    print(f"Training images          : {len(train_df):,}  "
          f"({train_df.patient_id.nunique():,} patients, {train_df.study_uid.nunique():,} studies)")
    print(f"Validation images        : {len(val_df):,}  "
          f"({val_df.patient_id.nunique():,} patients, {val_df.study_uid.nunique():,} studies)")
    print(f"Test images              : {len(test_df):,}  "
          f"({test_df.patient_id.nunique():,} patients, {test_df.study_uid.nunique():,} studies)")
    print(f"Model                    : EfficientNet-B3 (pretrained={model.pretrained})")
    print(f"Input size               : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Decision threshold       : {SELECTED_THRESHOLD:.4f}  ({THRESHOLD_CRITERION}, validation-derived)")
    print(f"Test ROC-AUC / F1        : {test_metrics['roc_auc']:.4f} / {test_metrics['f1']:.4f}")
    print(f"Checkpoint               : {FINAL_CHECKPOINT.resolve()}")
    print("=" * 78)

    # ----------------------------------------------------------------------------
    # [Code cell 63]
    # ---------- FINAL DEMONSTRATION -------------------------------------------
    # Pick the test image the model is most confident is abnormal, so the Grad-CAM
    # panel is informative. Change `demo_path` to any X-ray file to try your own.
    tdf_sorted = tdf.sort_values("prob", ascending=False)
    demo_row = tdf_sorted.iloc[0]
    demo_path = demo_row.image_path

    print(f"Demo image   : {demo_path}")
    print(f"Body region  : {demo_row.body_part}")
    print(f"Ground truth : {demo_row.label_name}   (shown for reference only; "
          f"the model never sees it)\n")

    final_result = predict_xray(
        demo_path,
        generate_gradcam=True,
        show=True,
        save_dir=OUTPUT_DIR / "final_demo",
        save_prefix="final_demo",
    )

    print("\nSaved artefacts:")
    for k in ["saved_original", "saved_gradcam", "saved_overlay", "saved_json"]:
        if k in final_result:
            print(f"  {k:<16}: {final_result[k]}")

    print("\n" + "=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)
    print("Reminder: this is a research/educational system. Every output above is a MODEL "
          "PREDICTION,\nnot a clinical diagnosis, and the Grad-CAM visualisation shows "
          "model-attributed importance,\nnot verified lesion localisation. MURA labels are "
          "Normal/Abnormal at the study level and\ndo not identify any specific pathology.")

    # ============================================================================
    # [Markdown cell 64]
    # ---
    # ## Appendix — practical notes
    # 
    # **Runtime on a Colab T4.** With the full dataset (~37k train images), one epoch takes roughly
    # 12–18 minutes at batch 32 with AMP. Stage 1 (3 epochs) plus stage 2 (up to 12, usually stopping
    # around 7–9) lands in the 2–3 hour range. To iterate faster, set `BODY_PARTS = ["XR_WRIST"]` in the
    # configuration cell — wrist is the largest single region and trains in a fraction of the time.
    # 
    # **If you hit CUDA OOM.** Lower `BATCH_SIZE` to 16, keep `IMAGE_SIZE = 300`, and raise
    # `UNFREEZE_FROM` to 6 or 7 so fewer blocks store activations for the backward pass. Reducing image
    # size below 288 starts to cost real signal on thin cortical lines.
    # 
    # **Study-level prediction.** MURA's official metric aggregates per study (mean probability over a
    # study's images, compared against the radiologist label). This notebook reports image-level metrics
    # throughout; to report the official-style number, group `tdf` by `study_uid`, average `prob`, and
    # compare against the study label — a short addition given the metadata already present.
    # 
    # **Known limitations.**
    # 
    # * Image-level labels are inherited from the study label, so an individual view in an abnormal study
    #   may not itself show the abnormality. This puts a ceiling on achievable image-level accuracy.
    # * Grad-CAM at 10×10 native resolution is coarse; upsampling to full radiograph size produces smooth
    #   blobs, not precise boundaries.
    # * No calibration step is applied — the sigmoid outputs are ranking scores and should not be read as
    #   calibrated probabilities of disease without a Platt/isotonic recalibration on held-out data.
    # * The model has been evaluated only on MURA. Performance on radiographs from other scanners,
    #   protocols, or populations is unknown.


if __name__ == "__main__":
    # multiprocessing.freeze_support() is a no-op on Linux/Mac and is required
    # for DataLoader multiprocessing to work correctly if this script is ever
    # frozen into a Windows .exe (e.g. with PyInstaller).
    import multiprocessing
    multiprocessing.freeze_support()
    main()
