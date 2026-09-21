# ================================================================
# PNEUMONIA DETECTION - DYNAMIC MODEL MODULE
# EfficientNet-B3 + Multi-Layer LayerCAM
#
# Used by pipeline.py
#
# REQUIRED PUBLIC API:
#   load_model(model_path, device)
#   build_transform(cfg)
#   GradCAM
#
# Image paths are NOT hardcoded.
# The main pipeline supplies the image dynamically.
# ================================================================

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from PIL import Image

from torchvision import models, transforms


# ================================================================
# DEFAULT CONFIG
# ================================================================

DEFAULT_CFG = {
    "model_name": "efficientnet_b3",
    "image_size": 300,
    "letterbox": True,
    "mean": [
        0.485,
        0.456,
        0.406
    ],
    "std": [
        0.229,
        0.224,
        0.225
    ],
    "class_names": {
        0: "NORMAL",
        1: "PNEUMONIA"
    },
    "threshold": 0.75
}


# ================================================================
# LETTERBOX
# ================================================================

class LetterboxResize:

    def __init__(
        self,
        size
    ):
        self.size = int(size)

    def __call__(
        self,
        img
    ):

        width, height = img.size

        scale = min(
            self.size / width,
            self.size / height
        )

        new_width = max(
            1,
            int(round(width * scale))
        )

        new_height = max(
            1,
            int(round(height * scale))
        )

        resized = img.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.BILINEAR
        )

        canvas = Image.new(
            img.mode,
            (
                self.size,
                self.size
            ),
            0
        )

        left = (
            self.size - new_width
        ) // 2

        top = (
            self.size - new_height
        ) // 2

        canvas.paste(
            resized,
            (
                left,
                top
            )
        )

        return canvas


# ================================================================
# TRANSFORM
# ================================================================

def build_transform(
    cfg
):

    cfg = {
        **DEFAULT_CFG,
        **(cfg or {})
    }

    image_size = int(
        cfg["image_size"]
    )

    if cfg.get(
        "letterbox",
        True
    ):

        resize = LetterboxResize(
            image_size
        )

    else:

        resize = transforms.Resize(
            (
                image_size,
                image_size
            )
        )

    return transforms.Compose(
        [
            resize,

            transforms.ToTensor(),

            transforms.Normalize(
                mean=cfg["mean"],
                std=cfg["std"]
            )
        ]
    )


# ================================================================
# MODEL
# ================================================================

class PneumoniaEfficientNetB3(
    nn.Module
):

    def __init__(
        self,
        dropout_p=0.4
    ):

        super().__init__()

        self.backbone = (
            models.efficientnet_b3(
                weights=None
            )
        )

        in_features = (
            self.backbone
            .classifier[1]
            .in_features
        )

        self.backbone.classifier = (
            nn.Sequential(
                nn.Dropout(
                    p=dropout_p
                ),
                nn.Linear(
                    in_features,
                    1
                )
            )
        )

    def forward(
        self,
        x
    ):

        return self.backbone(
            x
        ).reshape(-1)


# ================================================================
# STATE DICT EXTRACTION
# ================================================================

def _get_state_dict(
    checkpoint
):

    if not isinstance(
        checkpoint,
        dict
    ):

        return checkpoint

    if (
        "model_state_dict"
        in checkpoint
    ):

        return checkpoint[
            "model_state_dict"
        ]

    if (
        "state_dict"
        in checkpoint
    ):

        return checkpoint[
            "state_dict"
        ]

    if (
        "model"
        in checkpoint
        and
        isinstance(
            checkpoint["model"],
            dict
        )
    ):

        return checkpoint[
            "model"
        ]

    if all(
        isinstance(
            v,
            torch.Tensor
        )
        for v in checkpoint.values()
    ):

        return checkpoint

    raise RuntimeError(
        "Could not find model state_dict "
        "in checkpoint."
    )


# ================================================================
# CLEAN STATE DICT
# ================================================================

def _clean_state_dict(
    state_dict
):

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        # Most of your checkpoint keys are:
        # backbone.features...
        #
        # Keep them exactly like that.

        cleaned[key] = value

    return cleaned


# ================================================================
# LOAD MODEL
# ================================================================

def load_model(
    model_path,
    device
):
    """
    Called by pipeline.py.

    Returns:
        model
        cfg
        checkpoint
    """

    model_path = str(
        model_path
    )

    if not os.path.isfile(
        model_path
    ):

        raise FileNotFoundError(
            f"\nPneumonia checkpoint not found:\n"
            f"{model_path}"
        )

    print()
    print("=" * 76)
    print("PNEUMONIA CHECKPOINT")
    print("=" * 76)

    print(
        "File:",
        model_path
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    cfg = dict(
        DEFAULT_CFG
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "image_size" in checkpoint:

            cfg[
                "image_size"
            ] = int(
                checkpoint[
                    "image_size"
                ]
            )

        if "letterbox" in checkpoint:

            cfg[
                "letterbox"
            ] = bool(
                checkpoint[
                    "letterbox"
                ]
            )

        if "normalization" in checkpoint:

            normalization = (
                checkpoint[
                    "normalization"
                ]
            )

            cfg[
                "mean"
            ] = normalization.get(
                "mean",
                cfg["mean"]
            )

            cfg[
                "std"
            ] = normalization.get(
                "std",
                cfg["std"]
            )

        if "class_names" in checkpoint:

            cfg[
                "class_names"
            ] = {
                int(k): str(v)
                for k, v
                in checkpoint[
                    "class_names"
                ].items()
            }

        if "threshold" in checkpoint:

            cfg[
                "threshold"
            ] = float(
                checkpoint[
                    "threshold"
                ]
            )

        if "model_name" in checkpoint:

            cfg[
                "model_name"
            ] = str(
                checkpoint[
                    "model_name"
                ]
            )

        if "dropout_p" in checkpoint:

            cfg[
                "dropout_p"
            ] = float(
                checkpoint[
                    "dropout_p"
                ]
            )

    print(
        "Classes   :",
        cfg["class_names"]
    )

    print(
        "Input     :",
        cfg["image_size"]
    )

    print(
        "Threshold :",
        cfg["threshold"]
    )

    # ------------------------------------------------------------
    # Build architecture
    # ------------------------------------------------------------

    model = PneumoniaEfficientNetB3(
        dropout_p=cfg.get(
            "dropout_p",
            0.4
        )
    )

    state_dict = _clean_state_dict(
        _get_state_dict(
            checkpoint
        )
    )

    # ------------------------------------------------------------
    # Key compatibility
    # ------------------------------------------------------------

    model_keys = set(
        model.state_dict().keys()
    )

    # If checkpoint lacks backbone prefix,
    # try adding it.
    if not (
        set(state_dict.keys())
        &
        model_keys
    ):

        if all(
            (
                "backbone." + key
            ) in model_keys
            for key in list(
                state_dict.keys()
            )[:10]
        ):

            state_dict = {
                "backbone." + key: value
                for key, value
                in state_dict.items()
            }

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    result = model.load_state_dict(
        state_dict,
        strict=False
    )

    if result.missing_keys:

        print(
            "\nMissing keys:"
        )

        for key in result.missing_keys[
            :10
        ]:

            print(
                " ",
                key
            )

        raise RuntimeError(
            "Pneumonia checkpoint could not "
            "be loaded completely."
        )

    if result.unexpected_keys:

        print(
            "\nUnexpected keys:"
        )

        for key in result.unexpected_keys[
            :10
        ]:

            print(
                " ",
                key
            )

    model = model.to(
        device
    )

    model.eval()

    print(
        "\nModel loaded successfully."
    )

    return (
        model,
        cfg,
        checkpoint
    )


# ================================================================
# MULTI-LAYER LAYERCAM
# ================================================================

class GradCAM:
    """
    Pipeline-compatible GradCAM class.

    Despite the class name, this implementation uses
    multi-layer LayerCAM-style spatial weighting.

    Selected EfficientNet-B3 layers:

        features[-4]
        features[-3]
        features[-2]
        features[-1]

    The maps are resized and fused.

    The target is the RAW signed logit.

    For pneumonia:
        signed = +1

    For normal:
        signed = -1
    """

    def __init__(
        self,
        model,
        target_layer=None
    ):

        self.model = model

        features = (
            model
            .backbone
            .features
        )

        n = len(
            features
        )

        # More layers = better spatial information.
        self.layer_indices = [
            max(0, n - 4),
            max(0, n - 3),
            max(0, n - 2),
            max(0, n - 1)
        ]

        # Remove duplicates
        self.layer_indices = list(
            dict.fromkeys(
                self.layer_indices
            )
        )

        # If pipeline provides a target layer,
        # include it too.
        if target_layer is not None:

            self.target_layer = (
                target_layer
            )

        else:

            self.target_layer = (
                features[
                    self.layer_indices[-1]
                ]
            )

        self.layers = [
            features[i]
            for i in self.layer_indices
        ]

        self.activations = {}

        self.gradients = {}

        self.handles = []

        self._register_hooks()

    # ============================================================
    # HOOKS
    # ============================================================

    def _register_hooks(
        self
    ):

        for idx, layer in zip(
            self.layer_indices,
            self.layers
        ):

            self.handles.append(
                layer.register_forward_hook(
                    self._make_forward_hook(
                        idx
                    )
                )
            )

            self.handles.append(
                layer.register_full_backward_hook(
                    self._make_backward_hook(
                        idx
                    )
                )
            )

    def _make_forward_hook(
        self,
        idx
    ):

        def hook(
            module,
            inputs,
            output
        ):

            self.activations[
                idx
            ] = output

        return hook

    def _make_backward_hook(
        self,
        idx
    ):

        def hook(
            module,
            grad_input,
            grad_output
        ):

            if (
                grad_output
                and
                grad_output[0]
                is not None
            ):

                self.gradients[
                    idx
                ] = grad_output[0]

        return hook

    # ============================================================
    # CONTEXT MANAGER
    # ============================================================

    def __enter__(
        self
    ):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):

        self.remove_hooks()

    # ============================================================
    # REMOVE HOOKS
    # ============================================================

    def remove_hooks(
        self
    ):

        for handle in self.handles:

            try:

                handle.remove()

            except Exception:

                pass

        self.handles = []

    # ============================================================
    # LAYERCAM
    # ============================================================

    def generate(
        self,
        input_tensor,
        device,
        signed=1.0,
        output_size=None
    ):
        """
        Signature intentionally matches your pipeline.py.

        Returns:

            cam
            raw_logit
            feature_shapes
        """

        self.model.eval()

        self.activations.clear()

        self.gradients.clear()

        # --------------------------------------------------------
        # Fresh tensor requiring gradients
        # --------------------------------------------------------

        x = (
            input_tensor
            .to(device)
            .float()
            .detach()
            .requires_grad_(True)
        )

        self.model.zero_grad(
            set_to_none=True
        )

        # --------------------------------------------------------
        # RAW LOGIT
        # --------------------------------------------------------

        with torch.enable_grad():

            output = self.model(
                x
            )

            raw_logit = (
                output.reshape(-1)[0]
            )

            # IMPORTANT:
            # Explain raw logit, NOT sigmoid probability.
            target_score = (
                signed *
                raw_logit
            )

            target_score.backward()

        # --------------------------------------------------------
        # Output size
        # --------------------------------------------------------

        if output_size is None:

            output_size = (
                x.shape[-2],
                x.shape[-1]
            )

        # --------------------------------------------------------
        # Layer CAMs
        # --------------------------------------------------------

        cams = []

        feature_shapes = []

        for idx in self.layer_indices:

            activation = (
                self.activations.get(
                    idx
                )
            )

            gradient = (
                self.gradients.get(
                    idx
                )
            )

            if (
                activation is None
                or
                gradient is None
            ):

                continue

            feature_shapes.append(
                tuple(
                    activation.shape
                )
            )

            # Remove batch dimension
            activation = activation[
                0
            ]

            gradient = gradient[
                0
            ]

            # ----------------------------------------------------
            # LayerCAM:
            #
            # positive spatial gradients
            # weighted element-wise with activations.
            # ----------------------------------------------------

            positive_gradient = F.relu(
                gradient
            )

            cam = (
                positive_gradient
                *
                activation
            ).sum(
                dim=0
            )

            cam = F.relu(
                cam
            )

            # ----------------------------------------------------
            # Normalize layer CAM
            # ----------------------------------------------------

            cam_min = cam.min()

            cam_max = cam.max()

            if (
                cam_max -
                cam_min
                >
                1e-8
            ):

                cam = (
                    cam -
                    cam_min
                ) / (
                    cam_max -
                    cam_min
                )

            else:

                cam = torch.zeros_like(
                    cam
                )

            # ----------------------------------------------------
            # Resize
            # ----------------------------------------------------

            cam = F.interpolate(
                cam[
                    None,
                    None
                ],
                size=output_size,
                mode="bilinear",
                align_corners=False
            )[0, 0]

            cams.append(
                cam
            )

        if not cams:

            raise RuntimeError(
                "No Grad-CAM feature maps "
                "were captured."
            )

        # --------------------------------------------------------
        # Fuse multiple layers
        # --------------------------------------------------------

        fused = torch.stack(
            cams,
            dim=0
        ).mean(
            dim=0
        )

        # --------------------------------------------------------
        # Final normalization
        # --------------------------------------------------------

        fused = (
            fused -
            fused.min()
        )

        maximum = fused.max()

        if maximum > 1e-8:

            fused = (
                fused /
                maximum
            )

        else:

            fused = torch.zeros_like(
                fused
            )

        return (
            fused.detach().cpu().numpy(),
            float(
                raw_logit.detach().cpu().item()
            ),
            feature_shapes
        )


# ================================================================
# OPTIONAL DIRECT TEST
# ================================================================

def predict_image(
    image_path,
    model_path,
    device=None
):
    """
    Optional standalone helper.

    Your main pipeline does NOT need to use this.
    """

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model, cfg, _ = load_model(
        model_path,
        device
    )

    transform = build_transform(
        cfg
    )

    with Image.open(
        image_path
    ) as image:

        gray = image.convert(
            "L"
        )

    x = transform(
        gray.convert(
            "RGB"
        )
    ).unsqueeze(
        0
    ).to(
        device
    )

    with torch.no_grad():

        logit = (
            model(x)
            .reshape(-1)[0]
        )

        pneumonia_probability = (
            torch.sigmoid(
                logit
            ).item()
        )

    normal_probability = (
        1.0
        -
        pneumonia_probability
    )

    threshold = float(
        cfg["threshold"]
    )

    if (
        pneumonia_probability
        >=
        threshold
    ):

        predicted_index = 1

    else:

        predicted_index = 0

    prediction = (
        cfg[
            "class_names"
        ][
            predicted_index
        ]
    )

    confidence = (
        pneumonia_probability
        if predicted_index == 1
        else
        normal_probability
    )

    return {

        "prediction":
            prediction,

        "predicted_index":
            predicted_index,

        "confidence":
            confidence * 100,

        "normal_probability":
            normal_probability * 100,

        "pneumonia_probability":
            pneumonia_probability * 100
    }


# ================================================================
# MAIN
# ================================================================
#
# Do NOT use a hardcoded image path.
#
# The normal use is:
#
#     pipeline.py
#         |
#         +--> loads this module
#         |
#         +--> load_model(checkpoint, device)
#         |
#         +--> build_transform(cfg)
#         |
#         +--> GradCAM(...)
#
# ================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Dynamic Pneumonia EfficientNet-B3 "
            "testing"
        )
    )

    parser.add_argument(
        "image",
        nargs="?",
        default=None
    )

    parser.add_argument(
        "--model",
        required=True
    )

    args = parser.parse_args()

    image_path = (
        args.image
    )

    if not image_path:

        image_path = input(
            "Enter X-ray image path: "
        ).strip().strip('"')

    result = predict_image(
        image_path,
        args.model
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        "Prediction :",
        result["prediction"]
    )

    print(
        "Confidence :",
        f"{result['confidence']:.2f}%"
    )

    print(
        "Normal     :",
        f"{result['normal_probability']:.2f}%"
    )

    print(
        "Pneumonia  :",
        f"{result['pneumonia_probability']:.2f}%"
    )

    print("=" * 70)