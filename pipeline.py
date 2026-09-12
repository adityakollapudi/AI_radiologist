# ================================================================
# AI RADIOLOGIST
# X-RAY ROUTING + SPECIALIST CLASSIFICATION + GRAD-CAM
# ================================================================
#
# INPUT X-RAY
#      |
#      v
# X-RAY ROUTER
#      |
#      +---------------------------+
#      |                           |
#      v                           v
#    CHEST                 MUSCULOSKELETAL
#      |                           |
#      v                           v
# PNEUMONIA MODEL             MURA MODEL
# EfficientNet-B3             EfficientNet-B3
#      |                           |
#      v                           v
# NORMAL/PNEUMONIA            NORMAL/ABNORMAL
#      |                           |
#      +-------------+-------------+
#                    |
#                    v
#                 GRAD-CAM
#                    |
#                    v
#       ORIGINAL + GRAD-CAM IMAGE
#
# ================================================================

import os
import sys
import json
import traceback
import subprocess
import importlib.util
from pathlib import Path

# ================================================================
# PYTORCH / CUDA
# ================================================================

import torch
import numpy as np

# ================================================================
# MATPLOTLIB
# ================================================================

import matplotlib

# Use a non-interactive backend for saving.
# Windows image viewer will be used to POP the final image.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ================================================================
# PIL
# ================================================================

from PIL import Image


# ================================================================
# PROJECT ROOT
# ================================================================

ROOT = Path(__file__).resolve().parent


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# ROUTER PATHS
# ================================================================

ROUTER_DIR = ROOT / "router"

ROUTER_SCRIPT = (
    ROUTER_DIR / "router_model.py"
)

ROUTER_CHECKPOINT = (
    ROUTER_DIR / "xray_router_best.pth"
)


# ================================================================
# CHEST / PNEUMONIA PATHS
# ================================================================

PNEUMONIA_DIR = (
    ROOT / "Pneumonia_Detection"
)

PNEUMONIA_SCRIPT = (
    PNEUMONIA_DIR / "test_pneumonia.py"
)

PNEUMONIA_CHECKPOINT = (
    PNEUMONIA_DIR
    / "pneumonia_checkpoints"
    / "pneumonia_efficientnet_b3_best.pth"
)


# ================================================================
# MURA PATHS
# ================================================================

MURA_DIR = (
    ROOT / "Foreign_Object_Detection"
)

MURA_SCRIPT = (
    MURA_DIR / "test.py"
)

MURA_CHECKPOINT = (
    MURA_DIR
    / "mura_checkpoints"
    / "mura_efficientnet_b3_best.pth"
)


# ================================================================
# OUTPUT DIRECTORY
# ================================================================

OUTPUT_DIR = (
    ROOT / "pipeline_outputs"
)


# ================================================================
# SETTINGS
# ================================================================

ROUTER_THRESHOLD = 0.80

GRADCAM_ALPHA = 0.45

GRADCAM_CMAP = "jet"


# ================================================================
# SUPPORTED IMAGE TYPES
# ================================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


# ================================================================
# PRINT HEADER
# ================================================================

def print_header(title):

    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


# ================================================================
# LOAD PYTHON MODULE FROM FILE
# ================================================================

def load_module(
    module_name,
    script_path
):

    script_path = Path(
        script_path
    ).resolve()

    if not script_path.exists():

        raise FileNotFoundError(
            "\nPython script not found:\n"
            f"{script_path}\n"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(script_path)
    )

    if spec is None:

        raise ImportError(
            f"Could not create module spec for:\n"
            f"{script_path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


# ================================================================
# OPEN RESULT IMAGE
# ================================================================

def open_result_image(
    image_path
):

    image_path = Path(
        image_path
    ).resolve()

    if not image_path.exists():

        print(
            "[viewer] Result image does not exist:"
        )

        print(
            image_path
        )

        return

    try:

        if sys.platform.startswith("win"):

            # Windows
            os.startfile(
                str(image_path)
            )

        elif sys.platform == "darwin":

            subprocess.Popen(
                [
                    "open",
                    str(image_path)
                ]
            )

        else:

            subprocess.Popen(
                [
                    "xdg-open",
                    str(image_path)
                ]
            )

        print()
        print(
            "[viewer] Result image opened."
        )

    except Exception as e:

        print()
        print(
            "[viewer] Could not automatically "
            "open the result image."
        )

        print(
            f"Reason: {e}"
        )

        print()
        print(
            "Open this file manually:"
        )

        print(
            image_path
        )


# ================================================================
# LOAD INPUT IMAGE
# ================================================================

def load_input_image(
    image_path
):

    image_path = Path(
        image_path
    ).expanduser().resolve()

    if not image_path.exists():

        raise FileNotFoundError(
            "\nInput X-ray image not found:\n"
            f"{image_path}\n\n"
            "Please check the image path."
        )

    if (
        image_path.suffix.lower()
        not in VALID_EXTENSIONS
    ):

        raise ValueError(
            "\nUnsupported image extension:\n"
            f"{image_path.suffix}\n"
        )

    image = Image.open(
        image_path
    )

    # X-rays are treated as grayscale.
    gray = image.convert("L")

    return (
        gray,
        image,
        image_path
    )


# ================================================================
# ROUTER
# ================================================================

class XRayRouter:

    def __init__(
        self
    ):

        print()
        print(
            "Router checkpoint:"
        )

        print(
            ROUTER_CHECKPOINT
        )

        if not ROUTER_CHECKPOINT.exists():

            raise FileNotFoundError(
                "\nRouter checkpoint not found:\n"
                f"{ROUTER_CHECKPOINT}"
            )

        # --------------------------------------------------------
        # Load router implementation
        # --------------------------------------------------------

        self.module = load_module(
            "xray_router_runtime",
            ROUTER_SCRIPT
        )

        # --------------------------------------------------------
        # Load model
        # --------------------------------------------------------

        if not hasattr(
            self.module,
            "load_router"
        ):

            raise AttributeError(
                "\nrouter_model.py does not contain "
                "load_router()."
            )

        loaded = (
            self.module.load_router(
                ROUTER_CHECKPOINT,
                DEVICE,
                verbose=True
            )
        )

        if isinstance(
            loaded,
            tuple
        ):

            self.model = loaded[0]

            self.cfg = (
                loaded[1]
                if len(loaded) > 1
                else {}
            )

        else:

            self.model = loaded

            self.cfg = {}

        # --------------------------------------------------------
        # Build transform
        # --------------------------------------------------------

        if hasattr(
            self.module,
            "build_router_transform"
        ):

            image_size = (
                self.cfg.get(
                    "image_size",
                    224
                )
            )

            mean = (
                self.cfg.get(
                    "mean",
                    [0.485, 0.456, 0.406]
                )
            )

            std = (
                self.cfg.get(
                    "std",
                    [0.229, 0.224, 0.225]
                )
            )

            self.transform = (
                self.module
                .build_router_transform(
                    image_size,
                    mean,
                    std,
                    train=False
                )
            )

        else:

            self.transform = None

        self.model.eval()

    # ============================================================
    # PREDICT
    # ============================================================

    def predict(
        self,
        image
    ):

        if hasattr(
            self.module,
            "predict_route"
        ):

            result = (
                self.module.predict_route(
                    self.model,
                    self.cfg,
                    image,
                    DEVICE,
                    self.transform
                )
            )

            return self.normalize_result(
                result
            )

        # --------------------------------------------------------
        # Generic fallback router
        # --------------------------------------------------------

        if self.transform is None:

            raise RuntimeError(
                "Router does not expose "
                "predict_route() or a transform."
            )

        rgb = image.convert(
            "RGB"
        )

        tensor = (
            self.transform(
                rgb
            )
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            output = (
                self.model(
                    tensor
                )
            )

        if output.ndim == 1:

            output = output.unsqueeze(0)

        probabilities = (
            torch.softmax(
                output,
                dim=1
            )[0]
            .detach()
            .cpu()
            .numpy()
        )

        classes = {
            0: "CHEST",
            1: "MUSCULOSKELETAL"
        }

        index = int(
            np.argmax(
                probabilities
            )
        )

        route = classes.get(
            index,
            str(index)
        )

        confidence = (
            float(
                probabilities[index]
            )
            * 100.0
        )

        return {

            "route":
                route,

            "confidence":
                confidence,

            "confident":
                confidence >=
                ROUTER_THRESHOLD * 100,

            "probabilities": {

                "CHEST":
                    float(
                        probabilities[0]
                    ) * 100.0,

                "MUSCULOSKELETAL":
                    float(
                        probabilities[1]
                    ) * 100.0
            }
        }

    # ============================================================
    # NORMALIZE ROUTER RESULT
    # ============================================================

    def normalize_result(
        self,
        result
    ):

        if result is None:

            raise RuntimeError(
                "Router returned None."
            )

        route = (
            result.get(
                "route",
                result.get(
                    "class",
                    "UNKNOWN"
                )
            )
        )

        route = str(
            route
        ).upper()

        confidence = float(
            result.get(
                "confidence",
                0.0
            )
        )

        # Handle confidence returned as 0-1.
        if confidence <= 1.0:

            confidence *= 100.0

        probabilities = (
            result.get(
                "probabilities",
                {}
            )
        )

        normalized_probabilities = {}

        for key, value in (
            probabilities.items()
        ):

            value = float(
                value
            )

            if value <= 1.0:

                value *= 100.0

            normalized_probabilities[
                str(key).upper()
            ] = value

        if not normalized_probabilities:

            normalized_probabilities = {

                "CHEST":
                    confidence
                    if route == "CHEST"
                    else
                    100.0 - confidence,

                "MUSCULOSKELETAL":
                    confidence
                    if route ==
                    "MUSCULOSKELETAL"
                    else
                    100.0 - confidence
            }

        confident = (
            confidence >=
            ROUTER_THRESHOLD * 100.0
        )

        return {

            "route":
                route,

            "confidence":
                confidence,

            "confident":
                confident,

            "probabilities":
                normalized_probabilities
        }


# ================================================================
# DOWNSTREAM MODEL
# ================================================================

class SpecialistModel:

    def __init__(
        self,
        branch
    ):

        self.branch = (
            str(branch)
            .upper()
        )

        # --------------------------------------------------------
        # Select branch
        # --------------------------------------------------------

        if self.branch == "CHEST":

            self.script = (
                PNEUMONIA_SCRIPT
            )

            self.checkpoint = (
                PNEUMONIA_CHECKPOINT
            )

            self.module_name = (
                "pneumonia_runtime"
            )

        elif (
            self.branch ==
            "MUSCULOSKELETAL"
        ):

            self.script = (
                MURA_SCRIPT
            )

            self.checkpoint = (
                MURA_CHECKPOINT
            )

            self.module_name = (
                "mura_runtime"
            )

        else:

            raise ValueError(
                f"Unknown branch: "
                f"{self.branch}"
            )

        print()
        print(
            f"Loading {self.branch} specialist model"
        )

        print(
            f"Script     : {self.script}"
        )

        print(
            f"Checkpoint : {self.checkpoint}"
        )

        if not self.checkpoint.exists():

            raise FileNotFoundError(
                "\nModel checkpoint not found:\n"
                f"{self.checkpoint}"
            )

        # --------------------------------------------------------
        # Load branch script
        # --------------------------------------------------------

        self.module = load_module(
            self.module_name,
            self.script
        )

        # --------------------------------------------------------
        # Load model
        # --------------------------------------------------------

        if not hasattr(
            self.module,
            "load_model"
        ):

            raise AttributeError(
                f"\n{self.script} does not contain "
                "load_model()."
            )

        loaded = (
            self.module.load_model(
                str(
                    self.checkpoint
                ),
                DEVICE
            )
        )

        if not isinstance(
            loaded,
            tuple
        ):

            self.model = loaded
            self.cfg = {}

        else:

            self.model = loaded[0]

            self.cfg = (
                loaded[1]
                if len(loaded) > 1
                else {}
            )

        self.model.eval()

        # --------------------------------------------------------
        # Build transform
        # --------------------------------------------------------

        if hasattr(
            self.module,
            "build_transform"
        ):

            self.transform = (
                self.module
                .build_transform(
                    self.cfg
                )
            )

        else:

            raise AttributeError(
                f"\n{self.script} does not contain "
                "build_transform()."
            )

    # ============================================================
    # PREDICT
    # ============================================================

    def predict(
        self,
        image
    ):

        rgb = image.convert(
            "RGB"
        )

        x = (
            self.transform(
                rgb
            )
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            output = (
                self.model(
                    x
                )
            )

        logit = float(
            output
            .reshape(-1)[0]
            .item()
        )

        positive_probability = float(
            torch.sigmoid(
                output.reshape(-1)[0]
            )
            .item()
        )

        negative_probability = (
            1.0 -
            positive_probability
        )

        # --------------------------------------------------------
        # Threshold
        # --------------------------------------------------------

        threshold = float(
            self.cfg.get(
                "threshold",
                0.5
            )
        )

        # --------------------------------------------------------
        # Class names
        # --------------------------------------------------------

        raw_classes = (
            self.cfg.get(
                "class_names",
                {
                    0: "NORMAL",
                    1: "ABNORMAL"
                }
            )
        )

        class_names = {}

        for key, value in (
            raw_classes.items()
        ):

            class_names[
                int(key)
            ] = str(value).upper()

        negative_class = (
            class_names.get(
                0,
                "NORMAL"
            )
        )

        positive_class = (
            class_names.get(
                1,
                "ABNORMAL"
            )
        )

        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        predicted_index = int(
            positive_probability
            >= threshold
        )

        prediction = (
            positive_class
            if predicted_index == 1
            else
            negative_class
        )

        confidence_probability = (
            positive_probability
            if predicted_index == 1
            else
            negative_probability
        )

        return {

            "prediction":
                prediction,

            "predicted_index":
                predicted_index,

            "confidence":
                confidence_probability * 100.0,

            "negative_class":
                negative_class,

            "positive_class":
                positive_class,

            "negative_probability":
                negative_probability * 100.0,

            "positive_probability":
                positive_probability * 100.0,

            "threshold":
                threshold,

            "logit":
                logit,

            "input_tensor":
                x
        }


# ================================================================
# GRAD-CAM
# ================================================================

def generate_gradcam(
    specialist,
    prediction,
    original_image
):

    module = specialist.module

    # ------------------------------------------------------------
    # Verify GradCAM
    # ------------------------------------------------------------

    if not hasattr(
        module,
        "GradCAM"
    ):

        raise RuntimeError(
            "\nGradCAM class was not found in:\n"
            f"{specialist.script}\n\n"
            "Add/use the GradCAM implementation "
            "from your model testing script."
        )

    GradCAMClass = (
        module.GradCAM
    )

    model = specialist.model

    x = (
        prediction[
            "input_tensor"
        ]
    )

    predicted_index = (
        prediction[
            "predicted_index"
        ]
    )

    # ------------------------------------------------------------
    # Target
    # ------------------------------------------------------------
    #
    # Binary classifier:
    #
    # Positive class:
    #       +1
    #
    # Negative class:
    #       -1
    #
    # This allows the Grad-CAM to explain
    # the predicted class.
    # ------------------------------------------------------------

    if predicted_index == 1:

        target = 1.0

    else:

        target = -1.0

    original_np = (
        np.asarray(
            original_image,
            dtype=np.float32
        )
        /
        255.0
    )

    # ------------------------------------------------------------
    # Generate CAM
    # ------------------------------------------------------------

    cam_engine = None

    try:

        cam_engine = (
            GradCAMClass(
                model
            )
        )

        # --------------------------------------------------------
        # Context manager support
        # --------------------------------------------------------

        if hasattr(
            cam_engine,
            "__enter__"
        ):

            with cam_engine as engine:

                cam = engine.generate(
                    x,
                    DEVICE,
                    target,
                    original_np.shape
                )

        else:

            cam = cam_engine.generate(
                x,
                DEVICE,
                target,
                original_np.shape
            )

    finally:

        if (
            cam_engine is not None
            and
            hasattr(
                cam_engine,
                "remove_hooks"
            )
        ):

            try:

                cam_engine.remove_hooks()

            except Exception:

                pass

    # ------------------------------------------------------------
    # Some implementations return
    # (cam, prediction) or similar.
    # ------------------------------------------------------------

    if isinstance(
        cam,
        tuple
    ):

        cam = cam[0]

    cam = np.asarray(
        cam,
        dtype=np.float32
    )

    # ------------------------------------------------------------
    # Remove unnecessary dimensions
    # ------------------------------------------------------------

    cam = np.squeeze(
        cam
    )

    if cam.ndim != 2:

        raise RuntimeError(
            "Grad-CAM returned an unexpected "
            f"shape: {cam.shape}"
        )

    # ------------------------------------------------------------
    # Clean values
    # ------------------------------------------------------------

    cam = np.nan_to_num(
        cam,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # ------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------

    cam_min = float(
        cam.min()
    )

    cam_max = float(
        cam.max()
    )

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

        cam = np.zeros_like(
            cam
        )

    return cam


# ================================================================
# RESIZE CAM
# ================================================================

def resize_cam(
    cam,
    width,
    height
):

    cam_uint8 = (
        np.clip(
            cam,
            0,
            1
        )
        *
        255.0
    ).astype(
        np.uint8
    )

    cam_image = Image.fromarray(
        cam_uint8
    )

    cam_image = (
        cam_image.resize(
            (
                width,
                height
            ),
            Image.Resampling.BILINEAR
        )
    )

    return (
        np.asarray(
            cam_image,
            dtype=np.float32
        )
        /
        255.0
    )


# ================================================================
# CREATE FINAL VISUALIZATION
# ================================================================

def create_result_figure(
    image,
    cam,
    router_result,
    prediction,
    branch,
    output_path
):

    # ------------------------------------------------------------
    # Original image
    # ------------------------------------------------------------

    original = (
        np.asarray(
            image,
            dtype=np.float32
        )
        /
        255.0
    )

    height, width = (
        original.shape
    )

    # ------------------------------------------------------------
    # Resize CAM
    # ------------------------------------------------------------

    cam = resize_cam(
        cam,
        width,
        height
    )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 10)
    )

    # ============================================================
    # ORIGINAL IMAGE
    # ============================================================

    axes[0].imshow(
        original,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0].set_title(
        "ORIGINAL X-RAY",
        fontsize=20,
        fontweight="bold"
    )

    axes[0].axis(
        "off"
    )

    # ============================================================
    # GRAD-CAM
    # ============================================================

    axes[1].imshow(
        original,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    heatmap = axes[1].imshow(
        cam,
        cmap=GRADCAM_CMAP,
        alpha=GRADCAM_ALPHA,
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "GRAD-CAM",
        fontsize=20,
        fontweight="bold"
    )

    axes[1].axis(
        "off"
    )

    # ============================================================
    # COLOR BAR
    # ============================================================

    colorbar = fig.colorbar(
        heatmap,
        ax=axes[1],
        fraction=0.046,
        pad=0.04
    )

    colorbar.set_label(
        "Model Attention",
        fontsize=11
    )

    # ============================================================
    # TEXT INFORMATION
    # ============================================================

    if branch == "CHEST":

        image_type = (
            "CHEST X-RAY"
        )

    else:

        image_type = (
            "MUSCULOSKELETAL X-RAY"
        )

    prediction_name = (
        prediction["prediction"]
    )

    confidence = (
        prediction["confidence"]
    )

    # ------------------------------------------------------------
    # Main title
    # ------------------------------------------------------------

    fig.suptitle(
        "AI RADIOLOGIST",
        fontsize=27,
        fontweight="bold",
        y=0.985
    )

    # ------------------------------------------------------------
    # Image type
    # ------------------------------------------------------------

    fig.text(
        0.5,
        0.945,
        f"IMAGE TYPE : {image_type}",
        ha="center",
        fontsize=17,
        fontweight="bold"
    )

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    fig.text(
        0.5,
        0.910,
        f"PREDICTION : {prediction_name}",
        ha="center",
        fontsize=19,
        fontweight="bold"
    )

    # ------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------

    fig.text(
        0.5,
        0.878,
        f"CONFIDENCE : {confidence:.2f}%",
        ha="center",
        fontsize=15
    )

    # ------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------

    probability_text = (
        f"{prediction['negative_class']} : "
        f"{prediction['negative_probability']:.2f}%"
        "          |          "
        f"{prediction['positive_class']} : "
        f"{prediction['positive_probability']:.2f}%"
    )

    fig.text(
        0.5,
        0.845,
        probability_text,
        ha="center",
        fontsize=12,
        family="monospace"
    )

    # ------------------------------------------------------------
    # Router
    # ------------------------------------------------------------

    fig.text(
        0.5,
        0.815,
        (
            f"ROUTER : {branch} "
            f"({router_result['confidence']:.2f}%)"
        ),
        ha="center",
        fontsize=12
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    fig.text(
        0.5,
        0.055,
        (
            "Model : EfficientNet-B3"
            "        |        "
            f"Branch : {branch}"
        ),
        ha="center",
        fontsize=11
    )

    # ------------------------------------------------------------
    # Grad-CAM explanation
    # ------------------------------------------------------------

    fig.text(
        0.5,
        0.028,
        (
            "Grad-CAM highlights image regions "
            "that influenced the model prediction."
        ),
        ha="center",
        fontsize=9,
        style="italic"
    )

    # ============================================================
    # LAYOUT
    # ============================================================

    plt.subplots_adjust(
        left=0.03,
        right=0.96,
        top=0.79,
        bottom=0.09,
        wspace=0.04
    )

    # ============================================================
    # SAVE
    # ============================================================

    output_path = (
        Path(
            output_path
        )
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print()
    print(
        "[figure] saved ->"
    )

    print(
        output_path
    )

    return output_path


# ================================================================
# SAVE JSON
# ================================================================

def save_result_json(
    result,
    image_path
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    json_path = (
        OUTPUT_DIR /
        f"{Path(image_path).stem}_pipeline.json"
    )

    # IMPORTANT:
    # json.dump requires an OPEN FILE HANDLE.
    #
    # Previous error:
    #
    # AttributeError:
    # 'WindowsPath' object has no attribute 'write'
    #
    # This implementation fixes it.

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "[json] saved ->"
    )

    print(
        json_path
    )

    return json_path


# ================================================================
# MAIN PROCESS
# ================================================================

def process_xray(
    image_path,
    force_route=None
):

    print_header(
        "AI RADIOLOGIST - X-RAY ANALYSIS"
    )

    # ============================================================
    # DEVICE
    # ============================================================

    print(
        f"Device : {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU    : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA   : "
            f"{torch.version.cuda}"
        )

    else:

        print(
            "GPU    : Not available"
        )

    # ============================================================
    # IMAGE
    # ============================================================

    (
        gray_image,
        original_image,
        image_path
    ) = load_input_image(
        image_path
    )

    print()
    print(
        f"IMAGE : {image_path.name}"
    )

    print(
        f"SIZE  : "
        f"{original_image.width} x "
        f"{original_image.height}"
    )

    # ============================================================
    # ROUTER
    # ============================================================

    print_header(
        "STEP 1 - X-RAY ROUTER"
    )

    router = XRayRouter()

    router_result = (
        router.predict(
            gray_image
        )
    )

    print()
    print(
        f"ROUTER  : "
        f"{router_result['route']}"
    )

    print(
        f"CONFIDENCE : "
        f"{router_result['confidence']:.2f}%"
    )

    print()

    for (
        class_name,
        probability
    ) in (
        router_result[
            "probabilities"
        ].items()
    ):

        print(
            f"{class_name:<18} : "
            f"{probability:.2f}%"
        )

    # ============================================================
    # ROUTE SELECTION
    # ============================================================

    if force_route:

        branch = (
            force_route.upper()
        )

        print()
        print(
            f"[override] Forced route: "
            f"{branch}"
        )

    else:

        if not router_result[
            "confident"
        ]:

            print()
            print(
                "=" * 76
            )

            print(
                "LOW ROUTER CONFIDENCE"
            )

            print(
                "=" * 76
            )

            print(
                f"Router confidence "
                f"{router_result['confidence']:.2f}% "
                f"is below "
                f"{ROUTER_THRESHOLD * 100:.0f}%."
            )

            print()
            print(
                "No specialist model was executed."
            )

            print()
            print(
                "For testing you can force a branch:"
            )

            print()
            print(
                "python pipeline.py "
                "\"IMAGE_PATH\" "
                "--force-route CHEST"
            )

            print()

            print(
                "or"
            )

            print()

            print(
                "python pipeline.py "
                "\"IMAGE_PATH\" "
                "--force-route MUSCULOSKELETAL"
            )

            result = {

                "status":
                    "low_router_confidence",

                "image":
                    str(image_path),

                "router":
                    router_result
            }

            save_result_json(
                result,
                image_path
            )

            return result

        branch = (
            router_result[
                "route"
            ]
        )

    # ============================================================
    # SPECIALIST MODEL
    # ============================================================

    print_header(
        f"STEP 2 - {branch} SPECIALIST MODEL"
    )

    specialist = SpecialistModel(
        branch
    )

    # ============================================================
    # PREDICTION
    # ============================================================

    prediction = (
        specialist.predict(
            gray_image
        )
    )

    # ============================================================
    # PRINT FINAL RESULT
    # ============================================================

    print()
    print(
        "=" * 76
    )

    print(
        "FINAL PREDICTION"
    )

    print(
        "=" * 76
    )

    if branch == "CHEST":

        print(
            "IMAGE TYPE : CHEST X-RAY"
        )

    else:

        print(
            "IMAGE TYPE : MUSCULOSKELETAL X-RAY"
        )

    print(
        f"PREDICTION : "
        f"{prediction['prediction']}"
    )

    print(
        f"CONFIDENCE : "
        f"{prediction['confidence']:.2f}%"
    )

    print()

    print(
        f"{prediction['negative_class']:<15}"
        f": "
        f"{prediction['negative_probability']:.2f}%"
    )

    print(
        f"{prediction['positive_class']:<15}"
        f": "
        f"{prediction['positive_probability']:.2f}%"
    )

    print()

    print(
        f"THRESHOLD  : "
        f"{prediction['threshold']:.4f}"
    )

    # ============================================================
    # GRAD-CAM
    # ============================================================

    print_header(
        "STEP 3 - GRAD-CAM"
    )

    print(
        "Generating Grad-CAM..."
    )

    cam = generate_gradcam(
        specialist,
        prediction,
        gray_image
    )

    print(
        "Grad-CAM generated successfully."
    )

    # ============================================================
    # RESULT IMAGE
    # ============================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result_image_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_pipeline.png"
    )

    create_result_figure(
        image=gray_image,
        cam=cam,
        router_result=router_result,
        prediction=prediction,
        branch=branch,
        output_path=result_image_path
    )

    # ============================================================
    # JSON
    # ============================================================

    result = {

        "status":
            "success",

        "image":
            str(
                image_path.resolve()
            ),

        "image_name":
            image_path.name,

        "image_type":
            (
                "CHEST X-RAY"
                if branch == "CHEST"
                else
                "MUSCULOSKELETAL X-RAY"
            ),

        "selected_branch":
            branch,

        "router":
            router_result,

        "prediction": {

            "class":
                prediction[
                    "prediction"
                ],

            "confidence":
                round(
                    prediction[
                        "confidence"
                    ],
                    4
                ),

            "negative_class":
                prediction[
                    "negative_class"
                ],

            "negative_probability":
                round(
                    prediction[
                        "negative_probability"
                    ],
                    4
                ),

            "positive_class":
                prediction[
                    "positive_class"
                ],

            "positive_probability":
                round(
                    prediction[
                        "positive_probability"
                    ],
                    4
                ),

            "threshold":
                prediction[
                    "threshold"
                ]
        },

        "model":
            "EfficientNet-B3",

        "device":
            str(
                DEVICE
            ),

        "gradcam":
            True,

        "result_image":
            str(
                result_image_path.resolve()
            ),

        "disclaimer":
            (
                "Research/educational model. "
                "This output is not a clinical diagnosis. "
                "Grad-CAM highlights model-attributed "
                "regions and is not lesion segmentation."
            )
    }

    save_result_json(
        result,
        image_path
    )

    # ============================================================
    # AUTOMATIC POP-UP
    # ============================================================

    print_header(
        "STEP 4 - DISPLAY RESULT"
    )

    print(
        "Opening result image..."
    )

    open_result_image(
        result_image_path
    )

    # ============================================================
    # FINAL CONSOLE OUTPUT
    # ============================================================

    print()
    print(
        "=" * 76
    )

    print(
        "ANALYSIS COMPLETED"
    )

    print(
        "=" * 76
    )

    print(
        f"IMAGE TYPE : "
        f"{'CHEST X-RAY' if branch == 'CHEST' else 'MUSCULOSKELETAL X-RAY'}"
    )

    print(
        f"PREDICTION : "
        f"{prediction['prediction']}"
    )

    print(
        f"CONFIDENCE : "
        f"{prediction['confidence']:.2f}%"
    )

    print()
    print(
        "RESULT IMAGE:"
    )

    print(
        result_image_path
    )

    print()
    print(
        "The result image contains:"
    )

    print(
        "  LEFT  -> Original X-ray"
    )

    print(
        "  RIGHT -> Original X-ray + Grad-CAM"
    )

    print(
        "=" * 76
    )

    return result


# ================================================================
# COMMAND LINE
# ================================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "AI Radiologist X-ray router "
            "with specialist classification "
            "and Grad-CAM visualization."
        )
    )

    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help=(
            "Full path to X-ray image"
        )
    )

    parser.add_argument(
        "--force-route",
        choices=[
            "CHEST",
            "MUSCULOSKELETAL"
        ],
        default=None,
        help=(
            "Force the image through "
            "a particular specialist model."
        )
    )

    args = parser.parse_args()

    # ============================================================
    # GET IMAGE PATH
    # ============================================================

    image_path = args.image

    if image_path is None:

        image_path = input(
            "\nEnter the full path of the X-ray image: "
        ).strip()

    # Remove accidental quotes
    image_path = (
        image_path
        .strip()
        .strip('"')
        .strip("'")
    )

    # ============================================================
    # RUN
    # ============================================================

    try:

        process_xray(
            image_path=image_path,
            force_route=args.force_route
        )

    except FileNotFoundError as e:

        print()
        print(
            "=" * 76
        )

        print(
            "FILE ERROR"
        )

        print(
            "=" * 76
        )

        print(
            e
        )

        sys.exit(1)

    except AttributeError as e:

        print()
        print(
            "=" * 76
        )

        print(
            "MODEL INTERFACE ERROR"
        )

        print(
            "=" * 76
        )

        print(
            e
        )

        print()
        print(
            "The pipeline expects the router/specialist "
            "scripts to expose the model-loading and "
            "Grad-CAM functions used above."
        )

        sys.exit(2)

    except RuntimeError as e:

        print()
        print(
            "=" * 76
        )

        print(
            "RUNTIME ERROR"
        )

        print(
            "=" * 76
        )

        print(
            e
        )

        sys.exit(3)

    except KeyboardInterrupt:

        print()
        print(
            "Process interrupted."
        )

        sys.exit(130)

    except Exception:

        print()
        print(
            "=" * 76
        )

        print(
            "UNEXPECTED ERROR"
        )

        print(
            "=" * 76
        )

        traceback.print_exc()

        sys.exit(4)


# ================================================================
# START
# ================================================================

if __name__ == "__main__":

    main()