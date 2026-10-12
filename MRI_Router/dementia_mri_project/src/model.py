"""EfficientNet backbones (torchvision, ImageNet weights) with the specified classification head."""
import io

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.ops.misc import Conv2dNormActivation

ARCHS = {
    "efficientnet_b0": (tvm.efficientnet_b0, "EfficientNet_B0_Weights"),
    "efficientnet_b1": (tvm.efficientnet_b1, "EfficientNet_B1_Weights"),
    "efficientnet_b2": (tvm.efficientnet_b2, "EfficientNet_B2_Weights"),
    "efficientnet_b3": (tvm.efficientnet_b3, "EfficientNet_B3_Weights"),
    "efficientnet_v2_s": (tvm.efficientnet_v2_s, "EfficientNet_V2_S_Weights"),
    "efficientnet_v2_m": (tvm.efficientnet_v2_m, "EfficientNet_V2_M_Weights"),
}
DISPLAY = {"efficientnet_b0": "EfficientNet-B0", "efficientnet_v2_s": "EfficientNetV2-S", "efficientnet_v2_m": "EfficientNetV2-M",
           "efficientnet_b1": "EfficientNet-B1", "efficientnet_b2": "EfficientNet-B2", "efficientnet_b3": "EfficientNet-B3"}


class DementiaNet(nn.Module):
    """backbone -> global average pooling -> Dropout -> Linear(F, hidden) -> SiLU -> Dropout -> Linear(hidden, K)"""
    def __init__(self, arch="efficientnet_v2_s", num_classes=4, pretrained=True, dropout=0.3, dropout2=0.2, hidden=256):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"Unknown architecture {arch}; choose from {sorted(ARCHS)}")
        builder, wname = ARCHS[arch]
        weights = getattr(tvm, wname).IMAGENET1K_V1 if pretrained else None
        try:
            base = builder(weights=weights)
        except Exception as e:
            raise RuntimeError(f"Could not load ImageNet weights for {arch} ({e}). On Kaggle enable Internet; "
                               "or set pretrained=false explicitly (results will be much worse).") from e
        self.arch = arch
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = next(m for m in base.classifier if isinstance(m, nn.Linear)).in_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, hidden), nn.SiLU(),
                                  nn.Dropout(dropout2), nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.head(torch.flatten(self.avgpool(self.features(x)), 1))


def build_model(arch, cfg, pretrained=None):
    return DementiaNet(arch, cfg["num_classes"], cfg["pretrained"] if pretrained is None else pretrained,
                       cfg["dropout"], cfg.get("dropout2", 0.2), cfg.get("hidden_dim", 256))


# ------------------------------------------------------------------------------------- freezing control
def backbone_units(model):
    """Backbone split into fine-grained units (individual MBConv blocks) in forward order."""
    units = []
    for i, child in enumerate(model.features):
        if isinstance(child, nn.Sequential) and not isinstance(child, Conv2dNormActivation) and len(child) > 1:
            units += [(f"features.{i}.{j}", b) for j, b in enumerate(child)]
        else:
            units.append((f"features.{i}", child))
    return units


def set_trainable(model, stage, unfreeze_fraction=0.35):
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    info = {"stage": stage, "unfrozen_units": [], "unfrozen_backbone_fraction": 0.0}
    if stage == 2:
        units = backbone_units(model)
        total = sum(p.numel() for p in model.features.parameters())
        acc = 0
        for name, m in reversed(units):
            if acc >= unfreeze_fraction * total:
                break
            for p in m.parameters():
                p.requires_grad = True
            acc += sum(p.numel() for p in m.parameters())
            info["unfrozen_units"].append(name)
        info["unfrozen_backbone_fraction"] = acc / total
        info["unfrozen_units"] = info["unfrozen_units"][::-1]
    return info


def hold_frozen_bn(model):
    """Fully frozen backbone units stay in eval mode (BatchNorm statistics and stochastic depth fixed)."""
    for _, m in backbone_units(model):
        if not any(p.requires_grad for p in m.parameters()):
            m.eval()


def param_groups(model, stage, cfg):
    head = [p for p in model.head.parameters() if p.requires_grad]
    if stage == 1:
        return [{"params": head, "lr": cfg["head_lr"], "name": "head"}]
    bb = [p for p in model.features.parameters() if p.requires_grad]
    return [{"params": bb, "lr": cfg["backbone_lr"], "name": "backbone"},
            {"params": head, "lr": cfg["finetune_head_lr"], "name": "head"}]


# ---------------------------------------------------------------------------------------- reporting
def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024**2


@torch.no_grad()
def probe_feature_shapes(model, image_size, device="cpu"):
    shapes, hooks = [], []
    for i, child in enumerate(model.features):
        hooks.append(child.register_forward_hook(lambda m, inp, out, i=i: shapes.append((f"features.{i}", tuple(out.shape[1:])))))
    was = model.training
    model.eval()
    model(torch.zeros(1, 3, image_size, image_size, device=device))
    model.train(was)
    for h in hooks:
        h.remove()
    return shapes


def select_cam_layers(model, image_size, device="cpu"):
    """Choose Grad-CAM layers from measured feature-map shapes (not hard-coded names).

    final  = last feature module (last convolutional output, spatial > 1)
    late   = last module before `final` at the same (lowest) resolution
    middle = last module at stride 16
    early  = first module at stride 8
    """
    shapes = probe_feature_shapes(model, image_size, device)
    has_conv = {f"features.{i}": any(isinstance(m, nn.Conv2d) for m in child.modules()) for i, child in enumerate(model.features)}
    conv_shapes = [(n, s) for n, s in shapes if has_conv[n] and s[1] > 1]
    if not conv_shapes:
        raise RuntimeError("No convolutional feature map with spatial size > 1")
    final = conv_shapes[-1]
    fh = final[1][1]
    late = [x for x in conv_shapes[:-1] if x[1][1] == fh]
    mid = [x for x in conv_shapes if x[1][1] == max(fh, image_size // 16)]
    early = [x for x in conv_shapes if x[1][1] == image_size // 8] or [conv_shapes[len(conv_shapes) // 3]]
    sel = {"early": early[0], "middle": (mid[-1] if mid else conv_shapes[len(conv_shapes) // 2]),
           "late": (late[-1] if late else conv_shapes[-2]), "final": final}
    return {k: {"module": v[0], "channels": v[1][0], "spatial": [v[1][1], v[1][2]]} for k, v in sel.items()}, shapes


def get_module(model, dotted):
    m = model
    for part in dotted.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m
