"""Small display helpers for the notebooks."""
from pathlib import Path

import pandas as pd


def show(path, width=None):
    from IPython.display import Image, display
    p = Path(path)
    if p.exists():
        display(Image(filename=str(p), width=width))
    else:
        print(f"(not found: {p})")


def show_md(path):
    from IPython.display import Markdown, display
    p = Path(path)
    display(Markdown(p.read_text()) if p.exists() else Markdown(f"_not found: {p}_"))


def show_table(path, n=20):
    from IPython.display import display
    p = Path(path)
    if p.exists():
        display(pd.read_csv(p).head(n))
    else:
        print(f"(not found: {p})")


def sample_grid(df, classes, n_per_class=6, path_col="filepath", title=""):
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(2.2 * n_per_class, 2.4 * len(classes)), squeeze=False)
    for r, c in enumerate(classes):
        sub = df[df["class"] == c]
        sub = sub.sample(min(n_per_class, len(sub)), random_state=0) if len(sub) else sub
        for k in range(n_per_class):
            ax = axes[r, k]; ax.axis("off")
            if k < len(sub):
                ax.imshow(Image.open(sub.iloc[k][path_col]).convert("RGB"))
                if k == 0:
                    ax.set_title(c, fontsize=9, loc="left")
    fig.suptitle(title); plt.tight_layout(); plt.show()
