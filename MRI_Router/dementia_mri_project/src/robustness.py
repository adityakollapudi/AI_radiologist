"""Robustness testing on transformed COPIES of the test images (created in memory; the official test set
and its files are never modified). Post-hoc analysis only -- never used for model selection."""
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from .calibration import softmax_np
from .dataset import load_split_frames
from .evaluate import load_model, predict_logits
from .metrics import multiclass_metrics
from .preprocessing import load_rgb, model_input_image, normalize_tensor
from .utils import amp_settings, get_device, print_header, resolve_num_workers, save_json


class PerturbedDataset(Dataset):
    def __init__(self, df, spec, kind, amount, seed=0):
        self.paths = df.filepath.tolist(); self.labels = df.class_index.astype(int).tolist()
        self.spec, self.kind, self.amount, self.seed = spec, kind, amount, seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = model_input_image(load_rgb(self.paths[i]), self.spec)
        sign = 1 if i % 2 == 0 else -1                           # deterministic +/- alternation
        if self.kind == "brightness":
            img = ImageEnhance.Brightness(img).enhance(1 + sign * self.amount)
        elif self.kind == "contrast":
            img = ImageEnhance.Contrast(img).enhance(1 + sign * self.amount)
        elif self.kind == "rotation":
            img = img.rotate(sign * self.amount, resample=Image.BILINEAR, fillcolor=(self.spec["pad_value"],) * 3)
        t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        if self.kind == "noise":
            g = torch.Generator().manual_seed(self.seed * 100003 + i)
            t = (t + torch.randn(1, *t.shape[1:], generator=g) * self.amount).clamp(0, 1)
        return normalize_tensor(t, self.spec), self.labels[i], i


def run_robustness(cfg, paths, run_name):
    print_header(f"ROBUSTNESS TESTING -- {run_name} (transformed copies of the test images)")
    dev = get_device()
    model, ck = load_model(paths["checkpoints"] / run_name / "best_model.pth", dev)
    classes, spec = ck["class_names"], ck["preprocessing"]
    test = load_split_frames(paths)["test"]
    rb = cfg["robustness"]
    settings = [("original", None, 0), ("brightness", "brightness", rb["brightness"]), ("contrast", "contrast", rb["contrast"]),
                ("noise", "noise", rb["noise_std"]), ("rotation", "rotation", rb["rotation_deg"])]
    rows = []
    for name, kind, amt in settings:
        ds = PerturbedDataset(test, spec, kind, amt, cfg["seed"])
        dl = DataLoader(ds, batch_size=cfg["eval_batch_size"], num_workers=resolve_num_workers(cfg), shuffle=False)
        p = predict_logits(model, dl, dev, amp_settings(cfg))
        m = multiclass_metrics(p["y"], softmax_np(p["logits"]), classes)
        rows.append({"condition": name, "amount": amt, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "macro_auc": m["macro_roc_auc_ovr"], "n": m["n"]})
        print(f"  {name:<11} ({amt}) acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | macro-AUC {m['macro_roc_auc_ovr']:.4f}")
    df = pd.DataFrame(rows)
    base = df.iloc[0]
    for k in ("accuracy", "macro_f1", "macro_auc"):
        df[f"{k}_drop"] = base[k] - df[k]
    out = paths["results"] / "runs" / run_name
    df.to_csv(out / "robustness_results.csv", index=False)
    df.to_csv(paths["results"] / "robustness_results.csv", index=False)
    save_json({"settings": rb, "results": df.to_dict(orient="records"),
               "note": "brightness/contrast factors 1 +/- amount and rotation +/- degrees alternate per image; noise is "
                       "Gaussian (std on the 0-1 scale) shared across channels."}, out / "robustness.json")
    return df
