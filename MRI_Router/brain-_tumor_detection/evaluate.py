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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_brain_tumor_model(model_path=None):
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

    try:
        import tf_keras as keras
        return keras.models.load_model(model_path)
    except Exception:
        from tensorflow.keras.models import load_model
        return load_model(model_path)

def crop_brain_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if not cnts:
        return image

    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    cropped = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    if cropped.size == 0:
        return image
    return cropped

def load_dataset(data_dir, img_size=(240, 240)):
    yes_dir = os.path.join(data_dir, 'yes')
    no_dir = os.path.join(data_dir, 'no')

    X = []
    y = []

    print(f"Loading data from {data_dir}...")
    for label, folder in [(1, yes_dir), (0, no_dir)]:
        if not os.path.exists(folder):
            continue
        files = os.listdir(folder)
        for fname in files:
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            cropped = crop_brain_contour(img)
            resized = cv2.resize(cropped, dsize=img_size, interpolation=cv2.INTER_CUBIC)
            normalized = resized.astype(np.float32) / 255.0
            X.append(normalized)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} images (Tumorous: {np.sum(y == 1)}, Non-Tumorous: {np.sum(y == 0)})")
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Detection Model")
    parser.add_argument("--data", type=str, default="original", choices=["original", "augmented"],
                        help="Dataset to evaluate on ('original' for yes/no, 'augmented' for augmented data)")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    args = parser.parse_args()

    data_dir = "." if args.data == "original" else "augmented data"
    
    print("Loading model...")
    model = load_brain_tumor_model(args.model)
    print("Model loaded successfully.")

    X, y_true = load_dataset(data_dir)
    if len(X) == 0:
        print("No images found to evaluate.")
        return

    print("Running predictions...")
    y_prob = model.predict(X, batch_size=32, verbose=1)
    y_pred = (y_prob > 0.5).astype(int).flatten()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n================ EVALUATION RESULTS ================")
    print(f"Dataset:       {args.data.capitalize()} ({len(X)} images)")
    print(f"Accuracy:      {acc * 100:.2f}%")
    print(f"F1 Score:      {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"  [TN={cm[0, 0]:<4} FP={cm[0, 1]:<4}]  (Non-tumorous actual)")
    print(f"  [FN={cm[1, 0]:<4} TP={cm[1, 1]:<4}]  (Tumorous actual)")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Tumor", "Tumor"]))
    print("====================================================")

if __name__ == "__main__":
    main()
