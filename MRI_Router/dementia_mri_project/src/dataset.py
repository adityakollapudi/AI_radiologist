"""Lazy PyTorch dataset and DataLoaders (nothing is copied or preloaded)."""
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import build_transforms, load_rgb
from .utils import resolve_num_workers, seed_worker


class MRIDataset(Dataset):
    def __init__(self, df, spec, train=False, aug=None):
        self.df = df.reset_index(drop=True)
        self.paths = self.df.filepath.tolist()
        self.labels = self.df.class_index.astype(int).tolist()
        self.tf = build_transforms(spec, train=train, aug=aug)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = load_rgb(self.paths[i])
        except Exception as e:
            raise RuntimeError(f"Could not read {self.paths[i]}: {type(e).__name__}: {e}") from e
        return self.tf(img), self.labels[i], i


def load_split_frames(paths):
    return {s: pd.read_csv(paths["data"] / f"{s}.csv") for s in ("train", "val", "test")}


def make_loader(df, spec, cfg, train=False, aug=None, batch_size=None, shuffle=None):
    ds = MRIDataset(df, spec, train=train, aug=aug)
    nw = resolve_num_workers(cfg)
    g = torch.Generator()
    g.manual_seed(int(cfg["seed"]))
    bs = int(batch_size or (cfg["batch_size"] if train else cfg["eval_batch_size"]))
    return DataLoader(ds, batch_size=bs,
                      shuffle=train if shuffle is None else shuffle, num_workers=nw,
                      pin_memory=torch.cuda.is_available(), drop_last=bool(train and len(ds) % bs == 1),
                      persistent_workers=bool(cfg.get("persistent_workers", True)) and nw > 0,
                      worker_init_fn=seed_worker, generator=g)
