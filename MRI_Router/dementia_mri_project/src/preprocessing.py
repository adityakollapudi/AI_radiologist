"""Image preprocessing: validation -> RGB -> optional grayscale replication -> optional conservative brain crop
-> aspect-ratio-preserving resize -> padding -> tensor -> ImageNet normalisation.

All transform components are module-level classes (picklable), so DataLoader workers also work on Windows
(spawn start method), e.g. on an RTX 4060 laptop.
"""
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_rgb(path):
    with Image.open(path) as im:
        im.load()
        return im.convert("RGB")


class ColorMode:
    def __init__(self, mode):
        assert mode in ("rgb", "gray3"), mode
        self.mode = mode

    def __call__(self, img):
        if self.mode == "gray3":
            g = img.convert("L")
            return Image.merge("RGB", (g, g, g))
        return img.convert("RGB")


class BrainCrop:
    """Conservative crop to the foreground bounding box plus a margin (never cuts into the brain)."""
    def __init__(self, enabled=True, threshold=15, margin=0.08):
        self.enabled, self.threshold, self.margin = enabled, threshold, margin

    def __call__(self, img):
        if not self.enabled:
            return img
        g = np.asarray(img.convert("L"))
        fg = g > self.threshold
        rows, cols = np.where(fg.any(1))[0], np.where(fg.any(0))[0]
        if len(rows) < 8 or len(cols) < 8:
            return img
        h, w = g.shape
        my, mx = int(self.margin * h), int(self.margin * w)
        box = (max(0, cols[0] - mx), max(0, rows[0] - my), min(w, cols[-1] + 1 + mx), min(h, rows[-1] + 1 + my))
        return img.crop(box)


class Letterbox:
    """Aspect-ratio-preserving resize to fit size x size, then centred constant padding (no stretching)."""
    def __init__(self, size, fill=0):
        self.size, self.fill = int(size), int(fill)

    def __call__(self, img):
        return letterbox(img, self.size, self.fill)[0]


def letterbox(img, size, fill=0):
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = img if (nw, nh) == (w, h) else img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas.paste(im, (left, top))
    return canvas, {"scale": s, "left": left, "top": top, "w": nw, "h": nh, "orig_size": (w, h)}


class SharedGaussianNoise:
    """Very mild Gaussian noise, identical across channels (keeps grayscale replication exact)."""
    def __init__(self, std=0.01, p=0.3):
        self.std, self.p = std, p

    def __call__(self, t):
        if self.std <= 0 or torch.rand(1).item() >= self.p:
            return t
        return (t + torch.randn(1, *t.shape[1:]) * self.std).clamp(0, 1)


def preprocessing_spec(cfg, decision=None):
    d = decision or {}
    color = cfg.get("color_mode", "auto")
    crop = cfg.get("brain_crop", "auto")
    return {
        "image_size": int(cfg["image_size"]),
        "color_mode": d.get("color_mode", "rgb") if color == "auto" else color,
        "brain_crop": bool(d.get("brain_crop", False)) if crop == "auto" else bool(crop),
        "pad_value": int(cfg.get("pad_value", 0)),
        "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
    }


def deterministic_steps(spec):
    return [ColorMode(spec["color_mode"]), BrainCrop(spec["brain_crop"]), Letterbox(spec["image_size"], spec["pad_value"])]


def build_transforms(spec, train=False, aug=None):
    steps = deterministic_steps(spec)
    if train and aug and aug.get("enabled", True):
        steps += [T.RandomAffine(degrees=aug["rotation_deg"], translate=(aug["translate"], aug["translate"]),
                                 scale=tuple(aug["scale"]), fill=spec["pad_value"],
                                 interpolation=T.InterpolationMode.BILINEAR),
                  T.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"])]
        if aug.get("hflip_p", 0) > 0:
            steps.append(T.RandomHorizontalFlip(aug["hflip_p"]))
        steps += [T.ToTensor(), SharedGaussianNoise(aug["noise_std"], aug["noise_p"])]
    else:
        steps.append(T.ToTensor())
    steps.append(T.Normalize(spec["mean"], spec["std"]))
    return T.Compose(steps)


def model_input_image(img, spec):
    """The exact (un-normalised) image the network sees, as PIL -- used for overlays and robustness copies."""
    for step in deterministic_steps(spec):
        img = step(img)
    return img


def normalize_tensor(t, spec):
    return T.functional.normalize(t, spec["mean"], spec["std"])


def denormalize(t, spec):
    mean = torch.tensor(spec["mean"]).view(3, 1, 1)
    std = torch.tensor(spec["std"]).view(3, 1, 1)
    return (t.detach().cpu() * std + mean).clamp(0, 1)


def quality_check(img):
    """Basic input checks for single-image inference. Returns (ok, list of warnings)."""
    warn = []
    w, h = img.size
    if min(w, h) < 64:
        warn.append(f"very small image {w}x{h}")
    g = np.asarray(img.convert("L"), dtype=np.float32)
    if g.std() < 2:
        warn.append("image is almost uniform (blank?)")
    if (g < 10).mean() > 0.97:
        warn.append("image is almost entirely black")
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    if np.percentile(a.max(-1) - a.min(-1), 99) > 30:
        warn.append("strong colour content -- training images are grayscale MRI slices")
    if not (0.5 <= w / h <= 2.0):
        warn.append(f"unusual aspect ratio {w / h:.2f}")
    return len([x for x in warn if "blank" in x or "black" in x]) == 0, warn
