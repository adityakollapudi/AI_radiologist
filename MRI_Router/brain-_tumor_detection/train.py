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

import time
import argparse
import numpy as np
import cv2
import imutils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tf_keras as keras
    from tf_keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
    from tf_keras.models import Model
    from tf_keras.callbacks import TensorBoard, ModelCheckpoint
except Exception:
    import tensorflow.keras as keras
    from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

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

def load_data(dir_list, image_size=(240, 240)):
    X, y = [], []
    for directory in dir_list:
        if not os.path.exists(directory):
            continue
        label = 1 if directory.rstrip("/\\").endswith('yes') else 0
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath)
            if image is None:
                continue
            image = crop_brain_contour(image)
            image = cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_CUBIC)
            image = image.astype(np.float32) / 255.0
            X.append(image)
            y.append([label])
            
    X, y = shuffle(np.array(X), np.array(y), random_state=42)
    return X, y

def build_model(input_shape=(240, 240, 3)):
    X_input = Input(input_shape)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((4, 4), name='max_pool0')(X)
    X = MaxPooling2D((4, 4), name='max_pool1')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    return Model(inputs=X_input, outputs=X, name='BrainDetectionModel')

def main():
    parser = argparse.ArgumentParser(description="Train Brain Tumor Detection Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--data", type=str, default="augmented", choices=["augmented", "original"],
                        help="Dataset to train on")
    args = parser.parse_args()

    data_path = 'augmented data' if args.data == 'augmented' else '.'
    yes_dir = os.path.join(data_path, 'yes')
    no_dir = os.path.join(data_path, 'no')

    print(f"Loading data from {yes_dir} and {no_dir}...")
    X, y = load_data([yes_dir, no_dir])
    print(f"Loaded {len(X)} images. Splitting into train, validation, and test sets...")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    model = build_model((240, 240, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    log_name = f"train_run_{int(time.time())}"
    tensorboard = TensorBoard(log_dir=f"logs/{log_name}")
    checkpoint = ModelCheckpoint(
        "models/trained_best_model.h5",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max"
    )

    print(f"\nStarting training for {args.epochs} epochs...")
    model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard, checkpoint]
    )

    print("\nEvaluating on held-out test set...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc * 100:.2f}%")
    print(f"Model saved to models/trained_best_model.h5")

if __name__ == "__main__":
    main()
