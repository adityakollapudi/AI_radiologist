"""Standard and multi-layer Grad-CAM with explicit validity checks and sanity tests.

Grad-CAM is an attribution technique. It is NOT a lesion segmentation, a biomarker, proof of pathology or a diagnosis.

Fusion method (multi-layer): for each selected layer, CAM = ReLU(sum_c mean_xy(dY/dA_c) * A_c), upsampled
bilinearly to the input size and min-max normalised to [0, 1] INDEPENDENTLY; the fused map is the arithmetic
mean of these normalised maps, renormalised to [0, 1]. Raw activations are never averaged across layers.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import get_module, select_cam_layers
from .preprocessing import denormalize, load_rgb, model_input_image, normalize_tensor

FUSION_METHOD = ("mean of independently min-max-normalised, ReLU-ed, bilinearly upsampled Grad-CAM maps "
                 "(early, middle, late, final), renormalised to [0,1]")


class MultiLayerGradCAM:
    def __init__(self, model, layers):
        """layers: dict name -> dotted module path. Use {'final': path} for standard Grad-CAM."""
        self.model, self.layers = model, dict(layers)
        self.acts, self.grads, self.handles = {}, {}, []
        for name, path in self.layers.items():
            mod = get_module(model, path)
            self.handles.append(mod.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(module, inp, out):
            self.acts[name] = out
            if out.requires_grad:
                out.register_hook(lambda g: self.grads.__setitem__(name, g))
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __call__(self, x, target=None):
        """x: (1,3,H,W) normalised tensor. Returns dict with per-layer raw/normalised CAMs, probs, target, checks."""
        self.model.eval()
        self.acts.clear(); self.grads.clear()
        dev = next(self.model.parameters()).device
        x = x.to(dev).float().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits.float(), 1)[0].detach().cpu().numpy()
            tgt = int(probs.argmax()) if target is None else int(target)
            self.model.zero_grad(set_to_none=True)
            logits[0, tgt].backward()
        H, W = x.shape[-2:]
        out = {"probs": probs, "target": tgt, "predicted": int(probs.argmax()), "layers": {}, "checks": {}}
        norm_maps = []
        for name, path in self.layers.items():
            A, G = self.acts.get(name), self.grads.get(name)
            chk = {"layer_exists": True, "activation_exists": A is not None, "gradient_exists": G is not None}
            chk["activation_non_empty"] = bool(A is not None and A.numel() > 0 and A.detach().abs().sum().item() > 0)
            chk["gradient_non_empty"] = bool(G is not None and G.numel() > 0 and G.abs().sum().item() > 0)
            if not (chk["activation_exists"] and chk["gradient_exists"]):
                raise RuntimeError(f"Grad-CAM: missing activation/gradient for {name} ({path}): {chk}")
            A, G = A.detach().float(), G.float()
            w = G.mean(dim=(2, 3), keepdim=True)
            raw = F.relu((w * A).sum(1, keepdim=True))
            up = F.interpolate(raw, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            rmin, rmax = up.min(), up.max()
            norm = ((up - rmin) / (rmax - rmin)) if rmax > rmin else torch.zeros_like(up)
            norm = norm.cpu().numpy()
            chk["no_nan"] = bool(np.isfinite(norm).all())
            chk["not_all_zero"] = bool(raw.max().item() > 0)
            chk["not_all_one"] = bool(not np.allclose(norm, 1.0))
            chk["dimensions_match_input"] = norm.shape == (H, W)
            out["layers"][name] = {"module": path, "raw": raw[0, 0].cpu().numpy(), "cam": norm,
                                   "spatial": list(A.shape[-2:]), "channels": int(A.shape[1]), "checks": chk}
            norm_maps.append(norm)
        fused = np.mean(norm_maps, 0)
        fused = (fused - fused.min()) / (fused.max() - fused.min()) if fused.max() > fused.min() else np.zeros_like(fused)
        out["fused"] = fused
        with torch.no_grad():
            ref = torch.softmax(self.model(x.detach()).float(), 1)[0].cpu().numpy()
        hard = ("layer_exists", "activation_exists", "gradient_exists", "activation_non_empty", "gradient_non_empty",
                "no_nan", "dimensions_match_input")
        out["checks"] = {"target_class_used": tgt == (out["predicted"] if target is None else int(target)),
                         "model_output_consistent": bool(np.allclose(ref, probs, atol=1e-4)),
                         "all_layer_checks_passed": all(all(L["checks"][k] for k in hard) for L in out["layers"].values()),
                         # an all-zero map is a valid Grad-CAM outcome (no positive evidence for the target class);
                         # it is flagged, not hidden
                         "empty_maps": [n for n, L in out["layers"].items() if not L["checks"]["not_all_zero"]],
                         "saturated_maps": [n for n, L in out["layers"].items() if not L["checks"]["not_all_one"]]}
        return out


def cam_layers_for(model, image_size):
    sel, shapes = select_cam_layers(model, image_size, next(model.parameters()).device)
    return {k: v["module"] for k, v in sel.items()}, sel, shapes


def overlay(img_rgb01, cam, alpha=0.45):
    import matplotlib.cm as cm
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * img_rgb01 + alpha * heat, 0, 1)


def explain_image(model, path, spec, class_names, target=None, out_dir=None, name=None, true_label=None,
                  layers=None, multi=True, extra=None):
    """Grad-CAM for one image file. Saves original / model-input / heatmap / overlay / multi-layer panel / json."""
    import matplotlib
    import matplotlib.pyplot as plt
    img = load_rgb(path)
    inp = model_input_image(img, spec)
    x = normalize_tensor(torch.from_numpy(np.asarray(inp, dtype=np.float32) / 255.0).permute(2, 0, 1), spec)[None]
    if layers is None:
        layers, _, _ = cam_layers_for(model, spec["image_size"])
    use = layers if multi else {"final": layers["final"]}
    cam = MultiLayerGradCAM(model, use)
    try:
        r = cam(x, target)
    finally:
        cam.remove()
    base = np.asarray(inp, dtype=np.float32) / 255.0
    info = {"image": str(path), "predicted_class": class_names[r["predicted"]],
            "probabilities": {c: float(p) for c, p in zip(class_names, r["probs"])},
            "target_class": class_names[r["target"]], "true_class": class_names[true_label] if true_label is not None else None,
            "layers": {k: {"module": v["module"], "spatial": v["spatial"], "channels": v["channels"], "checks": v["checks"]}
                       for k, v in r["layers"].items()},
            "fusion_method": FUSION_METHOD if multi else None, "checks": r["checks"],
            "note": "Grad-CAM is an attribution map, not a lesion segmentation or diagnosis."}
    if extra:
        info.update(extra)
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        name = name or Path(path).stem
        img.save(out_dir / f"{name}_original.png")
        final = r["layers"]["final"]["cam"]
        Image.fromarray((plt.cm.jet(final)[..., :3] * 255).astype(np.uint8)).save(out_dir / f"{name}_heatmap.png")
        Image.fromarray((overlay(base, final) * 255).astype(np.uint8)).save(out_dir / f"{name}_overlay.png")
        cols = ["input"] + list(r["layers"]) + (["fused"] if multi else [])
        fig, ax = plt.subplots(1, len(cols) + 1, figsize=(3.1 * (len(cols) + 1), 3.6))
        ax[0].imshow(base); ax[0].set_title("model input")
        for a, k in zip(ax[1:], cols[1:]):
            m = r["fused"] if k == "fused" else r["layers"][k]["cam"]
            a.imshow(overlay(base, m))
            sub = "" if k == "fused" else f"\n{r['layers'][k]['module']} {r['layers'][k]['spatial'][0]}x{r['layers'][k]['spatial'][1]}"
            a.set_title(f"{k}{sub}", fontsize=9)
        ax[-1].imshow(final, cmap="jet"); ax[-1].set_title("final-layer heatmap", fontsize=9)
        for a in ax:
            a.axis("off")
        t = f"pred {info['predicted_class']} ({r['probs'].max():.1%}) | target {info['target_class']}"
        if true_label is not None:
            t += f" | actual {info['true_class']}"
        fig.suptitle(t, fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.88]); fig.savefig(out_dir / f"{name}_panel.png", dpi=100); plt.close(fig)
        if multi:
            Image.fromarray((overlay(base, r["fused"]) * 255).astype(np.uint8)).save(out_dir / f"{name}_fused_overlay.png")
        with open(out_dir / f"{name}_info.json", "w") as f:
            json.dump(info, f, indent=2, default=str)
        info["files"] = {"panel": str(out_dir / f"{name}_panel.png"), "overlay": str(out_dir / f"{name}_overlay.png")}
    return info, r, x


# ------------------------------------------------------------------------------------------- sanity tests
def _spearman(a, b):
    from scipy.stats import spearmanr
    a = F.interpolate(torch.tensor(a)[None, None], size=(32, 32), mode="area")[0, 0].numpy().ravel()
    b = F.interpolate(torch.tensor(b)[None, None], size=(32, 32), mode="area")[0, 0].numpy().ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(spearmanr(a, b).correlation)


def randomize_top(model, n_units=3, seed=0):
    """Cascading randomisation: re-initialise the head and the last n backbone units."""
    from .model import backbone_units
    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    mods = [m.head] + [u for _, u in backbone_units(m)[-n_units:]]
    for mod in mods:
        for sub in mod.modules():
            if hasattr(sub, "reset_parameters"):
                sub.reset_parameters()        # Conv / Linear weights and BatchNorm affine + running statistics
    return m


def sanity_checks(model, xs, layers, occlusion_fraction=0.2, seed=0):
    """Parameter-randomisation test (cascading: head + last 3 backbone units, then the whole network) and a
    deletion test (top-attributed pixels vs random pixels replaced by a blurred copy of the image)."""
    import pandas as pd
    import torchvision.transforms.functional as TF
    from .model import backbone_units
    rng = np.random.RandomState(seed)
    n_all = len(backbone_units(model))
    rnd = {"top3": randomize_top(model, 3, seed).eval(), "full": randomize_top(model, n_all, seed).eval()}
    final = {"final": layers["final"]}
    dev = next(model.parameters()).device
    rows = []
    for x in xs:
        c1 = MultiLayerGradCAM(model, final); r1 = c1(x); c1.remove()
        cam = r1["layers"]["final"]["cam"]
        row = {}
        for k, m in rnd.items():
            c2 = MultiLayerGradCAM(m, final); r2 = c2(x, target=r1["target"]); c2.remove()
            row[f"spearman_vs_randomized_{k}"] = _spearman(cam, r2["layers"]["final"]["cam"])
        H, W = cam.shape
        n = int(occlusion_fraction * H * W)
        x0 = x.to(dev)
        fill = TF.gaussian_blur(x0, kernel_size=[31, 31], sigma=[8.0, 8.0])
        with torch.no_grad():
            p0 = torch.softmax(model(x0).float(), 1)[0, r1["target"]].item()
            m_top = torch.zeros(H * W, dtype=torch.bool); m_top[torch.as_tensor(np.argsort(-cam.ravel())[:n])] = True
            pc = torch.softmax(model(torch.where(m_top.view(1, 1, H, W).to(dev), fill, x0)).float(), 1)[0, r1["target"]].item()
            drops = []
            for _ in range(5):
                rm = torch.zeros(H * W, dtype=torch.bool); rm[torch.as_tensor(rng.choice(H * W, n, replace=False))] = True
                drops.append(p0 - torch.softmax(model(torch.where(rm.view(1, 1, H, W).to(dev), fill, x0)).float(), 1)[0, r1["target"]].item())
        row.update({"prob_drop_cam_occlusion": p0 - pc, "prob_drop_random_occlusion": float(np.mean(drops)), "p_target": p0})
        rows.append(row)
    df = pd.DataFrame(rows)
    summ = {"n_images": int(len(df)),
            "mean_spearman_vs_randomized_top3": float(df.spearman_vs_randomized_top3.mean()),
            "mean_spearman_vs_randomized_full": float(df.spearman_vs_randomized_full.mean()),
            "randomization_test_passed": bool(df.spearman_vs_randomized_full.mean() < 0.5),
            "mean_prob_drop_cam_occlusion": float(df.prob_drop_cam_occlusion.mean()),
            "mean_prob_drop_random_occlusion": float(df.prob_drop_random_occlusion.mean()),
            "occlusion_test_passed": bool(df.prob_drop_cam_occlusion.mean() > df.prob_drop_random_occlusion.mean()),
            "occlusion_fraction": occlusion_fraction, "occlusion_fill": "Gaussian-blurred copy of the image (sigma 8 px)",
            "interpretation": ("Randomisation: maps from re-initialised networks should differ from the trained model's maps "
                               "(low rank correlation); a high correlation means the map mostly reflects image structure. "
                               "Deletion: replacing the most-attributed pixels should lower the target probability more than "
                               "replacing random pixels. These are sanity checks of the attribution method, not clinical validation.")}
    return summ, df
