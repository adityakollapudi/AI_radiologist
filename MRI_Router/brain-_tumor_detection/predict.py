import os
import sys

# Auto-redirect to .venv if running under external Python without dependencies
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
import imutils

# Suppress verbose TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_brain_tumor_model(model_path=None):
    """Loads the brain tumor detection model with fallback support."""
    if model_path is None:
        candidates = [
            'models/cnn-parameters-improvement-23-0.91.model',
            'models/cnn-parameters-improvement-23-0.91.h5',
            'models/best_model.keras'
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Try tf_keras first (most compatible with legacy .model files), then tensorflow.keras
    try:
        import tf_keras as keras
        return keras.models.load_model(model_path)
    except Exception:
        from tensorflow.keras.models import load_model
        return load_model(model_path)

def crop_brain_contour(image):
    """
    Finds the brain contour in the MRI image and crops extraneous background.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold and erode/dilate to isolate brain mass
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if not cnts:
        return image

    c = max(cnts, key=cv2.contourArea)

    # Extreme contour points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    cropped = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    if cropped.size == 0:
        return image
    return cropped

def preprocess_image(image_path, target_size=(240, 240)):
    """
    Reads, crops brain contour, resizes, and scales image to [0, 1].
    """
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Could not read image from {image_path}")

    cropped = crop_brain_contour(orig_img)
    resized = cv2.resize(cropped, dsize=target_size, interpolation=cv2.INTER_CUBIC)
    normalized = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)
    return input_tensor, orig_img, cropped

def predict(model, image_path):
    input_tensor, orig_img, cropped = preprocess_image(image_path)
    raw_pred = model.predict(input_tensor, verbose=0)
    prob = float(raw_pred[0][0])
    has_tumor = prob > 0.5
    confidence = prob if has_tumor else (1.0 - prob)
    return {
        "image_path": image_path,
        "has_tumor": has_tumor,
        "label": "TUMOR DETECTED (YES)" if has_tumor else "NO TUMOR (NO)",
        "probability": prob,
        "confidence_pct": confidence * 100.0
    }

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Detection Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to brain MRI image")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM tumor localization visualization")
    parser.add_argument("--save-cam", type=str, default=None, help="File path to save the Grad-CAM diagnostic panel")
    args = parser.parse_args()

    print("Loading model...")
    model = load_brain_tumor_model(args.model)
    print("Model loaded successfully.\n")

    if args.image:
        images = [args.image]
    else:
        # Default test run on a couple samples from both classes
        print("No image specified. Running test inference on sample images from yes/ and no/...\n")
        images = [
            "yes/Y1.jpg",
            "yes/Y10.jpg",
            "no/1 no.jpeg",
            "no/10 no.jpg"
        ]

    for img_path in images:
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        result = predict(model, img_path)
        print("--------------------------------------------------")
        print(f"Image:       {result['image_path']}")
        print(f"Result:      {result['label']}")
        print(f"Confidence:  {result['confidence_pct']:.2f}% (tumor probability: {result['probability']:.4f})")

        if args.gradcam or args.save_cam:
            from gradcam import generate_gradcam_panel
            cam_res = generate_gradcam_panel(model, img_path)
            out_file = args.save_cam or f"gradcam_{os.path.basename(img_path)}"
            cv2.imwrite(out_file, cam_res["panel"])
            print(f"Grad-CAM:    Saved diagnostic panel to {out_file}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
