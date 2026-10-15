"""Configuration, paths, reproducibility, environment information and small helpers."""
import copy
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = ("Research prototype. Predicts dataset-defined dementia categories from preprocessed JPG brain MRI "
              "slices. It is NOT a clinically validated diagnostic system; model confidence is not clinical certainty.")
SEVERITY = {"Non Demented": 0, "Very mild Dementia": 1, "Mild Dementia": 2, "Moderate Dementia": 3}


# --------------------------------------------------------------------------------------------- config
def _deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, overrides=None):
    """Load configs/config.json, then apply DEMENTIA_CFG (JSON env var) and explicit overrides."""
    path = Path(path) if path else PROJECT_ROOT / "configs" / "config.json"
    with open(path) as f:
        cfg = json.load(f)
    env = os.environ.get("DEMENTIA_CFG")
    if env:
        _deep_update(cfg, json.loads(env))
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    cfg["_config_path"] = str(path)
    return cfg


def parse_cli_overrides(items):
    """['a.b=1', 'model="x"'] -> nested dict (values parsed as JSON when possible)."""
    out = {}
    for it in items or []:
        key, val = it.split("=", 1)
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return out


def on_kaggle():
    return Path("/kaggle/working").exists()


def get_paths(cfg):
    if cfg.get("output_root"):
        root = Path(cfg["output_root"])
    elif on_kaggle():
        root = Path("/kaggle/working/dementia_mri_project")
    else:
        root = PROJECT_ROOT
    p = {
        "root": root,
        "data": root / "data",
        "audit": root / "data" / "audit",
        "checkpoints": root / "checkpoints",
        "results": root / "results",
        "gradcam": root / "gradcam",
        "reports": root / "reports",
        "configs": root / "configs",
    }
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------------------ json helpers
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path, set, tuple)):
        return str(o) if isinstance(o, Path) else list(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _clean_nans(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean_nans(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nans(v) for v in o]
    return o


def _stringify_keys(o):
    if isinstance(o, dict):
        return {(" -> ".join(map(str, k)) if isinstance(k, tuple) else k if isinstance(k, (str, int, float, bool)) or k is None
                 else str(k)): _stringify_keys(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_stringify_keys(v) for v in o]
    return o


def save_json(obj, path):
    obj = _stringify_keys(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(_clean_nans(json.loads(json.dumps(obj, default=_json_default))), f, indent=2)
    os.replace(tmp, path)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------ reproducibility
def seed_everything(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    except ImportError:
        pass


def seed_worker(worker_id):
    import torch
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_settings(cfg):
    """(enabled, dtype, use_scaler). bf16 on Ampere/Ada (e.g. RTX 4060), fp16 + GradScaler on Turing (T4)."""
    import torch
    if not (cfg.get("amp", True) and torch.cuda.is_available()):
        return False, None, False
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, False
    return True, torch.float16, True


def env_info():
    import torch
    import torchvision
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": None,
        "gpu_memory_gb": None,
        "cpu_count": os.cpu_count(),
        "kaggle": on_kaggle(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
        info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    try:
        import pandas
        info["pandas"] = pandas.__version__
    except ImportError:
        pass
    return info


def print_env():
    info = env_info()
    print_header("ENVIRONMENT")
    for k in ("python", "torch", "torchvision", "cuda_available", "cuda_version", "gpu", "gpu_memory_gb", "cpu_count", "kaggle"):
        print(f"  {k:<16}: {info.get(k)}")
    if not info["cuda_available"]:
        print("  !! CUDA not available -- training EfficientNetV2-S on CPU is impractically slow.")
    return info


def print_header(title, ch="="):
    print("\n" + ch * 72)
    print(title)
    print(ch * 72)


class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0
        if self.name:
            print(f"[time] {self.name}: {self.elapsed:.1f}s")


def resolve_num_workers(cfg):
    n = int(cfg.get("num_workers", 2))
    return max(0, min(n, os.cpu_count() or 1))
