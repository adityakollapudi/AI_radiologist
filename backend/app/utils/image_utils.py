import io
import base64
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Strictly enforce non-interactive backend for web server
import matplotlib.pyplot as plt

def bytes_to_pil_image(contents: bytes, filename: str = "") -> Image.Image:
    """Converts raw byte payload to PIL Image, handling DICOM if applicable."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".dcm":
        import pydicom
        dcm = pydicom.dcmread(io.BytesIO(contents))
        arr = dcm.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    
    img = Image.open(io.BytesIO(contents))
    return img.convert("RGB")

def pil_to_base64_data_url(img: Image.Image, format: str = "PNG") -> str:
    """Encodes a PIL Image into a base64 data URL string."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"

def numpy_overlay_to_base64(orig_img: Image.Image, cam_map: np.ndarray, alpha: float = 0.45, cmap: str = "jet") -> str:
    """Blends a 2D float CAM array with an original PIL image and returns a base64 data URL."""
    orig_rgb = np.array(orig_img.convert("RGB")).astype(np.float32) / 255.0
    h, w = orig_rgb.shape[:2]

    # Resize CAM to match image
    cam_resized = Image.fromarray(cam_map.astype(np.float32)).resize((w, h), resample=Image.Resampling.BILINEAR)
    cam_np = np.array(cam_resized)
    
    # Normalize 0..1
    cam_min, cam_max = cam_np.min(), cam_np.max()
    if cam_max > cam_min:
        cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = np.zeros_like(cam_np)

    colormap = plt.get_cmap(cmap)
    heatmap_rgba = colormap(cam_norm)[:, :, :3]  # RGB float 0..1

    overlay = (1.0 - alpha) * orig_rgb + alpha * heatmap_rgba
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    overlay_pil = Image.fromarray(overlay)
    return pil_to_base64_data_url(overlay_pil, format="PNG")
