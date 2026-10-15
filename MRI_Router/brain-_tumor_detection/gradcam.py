import os
import sys

# Auto-redirect to .venv if running under external Python without tensorflow
_venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe"))
if os.path.exists(_venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv_python).lower():
    try:
        import tensorflow
    except ImportError:
        import subprocess
        sys.exit(subprocess.call([_venv_python] + sys.argv))

import argparse
import numpy as np
import cv2
import tensorflow as tf

try:
    import tf_keras as keras
except Exception:
    from tensorflow import keras

from predict import load_brain_tumor_model, preprocess_image, crop_brain_contour

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def make_gradcam_heatmap(input_tensor, model, cropped_img=None, layer_names=None):
    """
    Computes a Multi-Layer Gradient Projection (Layer-Fused Grad-CAM) heatmap.
    Collects gradients and activations from all intermediate convolutional and pooling
    layers across the network to localize the brain tumor without border artifacts
    or scattered pixel noise.
    """
    if "tf_keras" in model.__class__.__module__:
        import tf_keras as k_backend
    else:
        try:
            import tf_keras as k_backend
        except Exception:
            from tensorflow import keras as k_backend

    if layer_names is None:
        layer_names = ['conv0', 'bn0', 'activation', 'max_pool0', 'max_pool1']

    # Resolve target layer objects
    target_layers = [model.get_layer(name) for name in layer_names]
    submodel = k_backend.models.Model(
        inputs=model.inputs,
        outputs=[l.output for l in target_layers] + [model.output]
    )

    # Pre-sigmoid logit weights from the dense classifier layer
    fc = model.get_layer('fc')
    w, b = fc.get_weights()

    with tf.GradientTape(persistent=True) as tape:
        t_input = tf.convert_to_tensor(input_tensor)
        outputs = submodel(t_input)
        layer_acts = outputs[:-1]
        predictions = outputs[-1]
        if isinstance(predictions, list):
            predictions = predictions[0]

        # Use pre-sigmoid logit to prevent sigmoid gradient saturation at p ~ 0.999
        flat = model.get_layer('flatten')(layer_acts[-1])
        logit = tf.matmul(flat, w) + b
        loss = logit[:, 0]

    prob = float(predictions[0, 0].numpy())
    has_tumor = prob > 0.5

    # Determine dimensions and build brain parenchyma mask
    if cropped_img is not None:
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        target_h, target_w = cropped_img.shape[:2]
    else:
        gray = (input_tensor[0, :, :, 0] * 255).astype(np.uint8)
        target_h, target_w = 240, 240

    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    # Morphological erosion removes outer skull border rim and zero-padding artifacts
    k_erode = max(11, int(min(target_h, target_w) * 0.05) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode))
    brain_mask = cv2.erode(thresh, kernel, iterations=1).astype(np.float32) / 255.0

    # Hierarchical semantic weights for each layer:
    # Deep layers (max_pool1, max_pool0) provide coarse receptive field and lesion localization
    # Shallower layers (activation, bn0, conv0) provide fine structural boundaries
    layer_weights = {
        'max_pool1': 0.35,
        'max_pool0': 0.30,
        'activation': 0.20,
        'bn0': 0.10,
        'conv0': 0.05
    }

    fused_heatmap = np.zeros((target_h, target_w), dtype=np.float32)

    for name, act in zip(layer_names, layer_acts):
        grads = tape.gradient(loss, act)[0].numpy()
        a = act[0].numpy()
        h_l, w_l = a.shape[:2]

        # Resize brain mask to layer resolution
        m_l = cv2.resize(brain_mask, (w_l, h_l), interpolation=cv2.INTER_NEAREST)
        m_g = grads * m_l[..., np.newaxis]
        m_a = a * m_l[..., np.newaxis]

        # Positive gradients: features that increase tumor probability
        pos_g = np.maximum(m_g, 0.0)

        # 1. Channel-wise Grad-CAM weight: GAP over positive gradients
        alpha = np.mean(pos_g, axis=(0, 1))
        cam = np.zeros((h_l, w_l), dtype=np.float32)
        for k in range(a.shape[-1]):
            if alpha[k] > 0:
                cam += alpha[k] * np.maximum(m_a[:, :, k], 0.0)

        # 2. Element-wise spatial gradient projection
        proj = np.sum(pos_g * np.maximum(m_a, 0.0), axis=-1)

        if cam.max() > 0:
            cam = cam / cam.max()
        if proj.max() > 0:
            proj = proj / proj.max()

        combined_layer = 0.5 * cam + 0.5 * proj

        # Bicubic resize to target dimensions
        res = cv2.resize(combined_layer, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Spatial Gaussian blur scaled to layer resolution to prevent pixel-level noise dots
        blur_k = max(9, int(min(target_h, target_w) * 0.06) | 1)
        res = cv2.GaussianBlur(res, (blur_k, blur_k), 0)
        if res.max() > 0:
            res = res / res.max()

        w_layer = layer_weights.get(name, 1.0 / len(layer_names))
        fused_heatmap += w_layer * res

    # Mask outer background & skull boundary
    fused_heatmap = fused_heatmap * brain_mask

    # If tumor detected, modulate with lesion intensity prior to concentrate focus on the tumor core
    if has_tumor:
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        bp = blurred[brain_mask > 0]
        if len(bp) > 0:
            med = np.median(bp)
            std = np.std(bp) + 1e-5
            z = np.maximum((blurred.astype(np.float32) - med) / std, 0.0) * brain_mask
            if z.max() > 0:
                z = z / z.max()
                fused_heatmap = fused_heatmap * (0.3 + 1.2 * z)
                fused_heatmap = fused_heatmap * brain_mask

    # Final smooth of the fused multi-layer heatmap
    final_blur = max(7, int(min(target_h, target_w) * 0.04) | 1)
    fused_heatmap = cv2.GaussianBlur(fused_heatmap, (final_blur, final_blur), 0)

    # Normalize heatmap
    if fused_heatmap.max() > 0:
        if has_tumor:
            p99 = np.percentile(fused_heatmap[fused_heatmap > 0], 99.0)
            fused_heatmap = np.clip(fused_heatmap, 0.0, p99) / (p99 + 1e-10)
        else:
            # Healthy brain scans remain cool (dark blue)
            fused_heatmap = (fused_heatmap / fused_heatmap.max()) * 0.20
    else:
        fused_heatmap = np.zeros_like(fused_heatmap)

    return fused_heatmap

def overlay_gradcam(image, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET, threshold=0.15):
    """
    Overlays the Grad-CAM heatmap onto an image with continuous soft thresholding.
    image: uint8 BGR or RGB image
    heatmap: 2D float numpy array in [0, 1]
    alpha: transparency factor for heatmap blending
    colormap: OpenCV colormap
    threshold: noise floor cutoff (soft continuous ramp to avoid pixel specks)
    """
    h, w = image.shape[:2]
    resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Soft continuous thresholding to eliminate salt-and-pepper noise specks
    if threshold > 0:
        enhanced = np.maximum(resized - threshold, 0.0) / (1.0 - threshold + 1e-10)
        enhanced = np.power(enhanced, 0.75)
    else:
        enhanced = np.clip(resized, 0.0, 1.0)

    heatmap_uint8 = np.uint8(255 * np.nan_to_num(enhanced, nan=0.0))
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    superimposed = cv2.addWeighted(colored_heatmap, alpha, image, 1.0 - alpha, 0)
    return superimposed, colored_heatmap

def generate_gradcam_panel(model, image_path, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Generates a full diagnostic panel:
    [Original Scan] | [Cropped Brain] | [Pure Heatmap] | [Grad-CAM Overlay]
    """
    input_tensor, orig_img, cropped = preprocess_image(image_path)
    pred_prob = float(model.predict(input_tensor, verbose=0)[0][0])
    has_tumor = pred_prob > 0.5
    confidence = pred_prob if has_tumor else (1.0 - pred_prob)

    heatmap = make_gradcam_heatmap(input_tensor, model, cropped_img=cropped)
    overlay, colored_heatmap = overlay_gradcam(cropped, heatmap, alpha=alpha, colormap=colormap)

    panel_h = 240
    def fit_h(img):
        ratio = panel_h / img.shape[0]
        new_w = max(1, int(img.shape[1] * ratio))
        return cv2.resize(img, (new_w, panel_h))

    p1 = fit_h(orig_img)
    p2 = fit_h(cropped)
    p3 = fit_h(colored_heatmap)
    p4 = fit_h(overlay)

    panel = np.hstack([p1, p2, p3, p4])

    label_text = f"TUMOR ({confidence*100:.1f}%)" if has_tumor else f"NO TUMOR ({confidence*100:.1f}%)"
    return {
        "has_tumor": has_tumor,
        "probability": pred_prob,
        "confidence": confidence,
        "label": label_text,
        "orig_img": orig_img,
        "cropped_img": cropped,
        "heatmap": heatmap,
        "colored_heatmap": colored_heatmap,
        "overlay": overlay,
        "panel": panel
    }

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Tumor Localization")
    parser.add_argument("--image", type=str, default="yes/Y1.jpg", help="Path to input brain MRI image")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--output", type=str, default="gradcam_result.jpg", help="Path to save output panel")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha (0.0 to 1.0)")
    args = parser.parse_args()

    print(f"Loading model and running Grad-CAM on: {args.image}...")
    model = load_brain_tumor_model(args.model)
    res = generate_gradcam_panel(model, args.image, alpha=args.alpha)

    cv2.imwrite(args.output, res["panel"])
    print("--------------------------------------------------")
    print(f"Prediction:        {res['label']}")
    print(f"Tumor Probability: {res['probability']:.4f}")
    print(f"Confidence:        {res['confidence']*100:.2f}%")
    print(f"Diagnostic Panel:  {os.path.abspath(args.output)}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
