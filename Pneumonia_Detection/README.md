# AI Radiologist - Pneumonia Detection

## Overview
The **AI Radiologist** project is a deep learning-based diagnostic tool designed to assist in analyzing medical X-ray images. This specific module focuses on **Pneumonia Detection** in Chest X-rays, identifying whether a patient has pneumonia or their lungs are healthy (normal).

## Features
- **Pneumonia Detection:** Automatically classifies chest X-ray scans as either Normal or Pneumonia.
- **Explainable AI (GradCAM):** Generates GradCAM (Gradient-weighted Class Activation Mapping) overlay images to highlight the exact regions the model focused on when predicting pneumonia. This provides transparency and builds trust in the AI's diagnostic suggestions.
- **EfficientNet Architecture:** Leverages the powerful and efficient `EfficientNetB3` deep neural network for high-accuracy image classification.
- **Evaluation & Metrics:** Provides comprehensive evaluation scripts that generate precision-recall curves, confusion matrices, and detailed threshold selections.

## Project Structure
- `train_pneumonia.py`: Script used for training the EfficientNetB3 model on the pneumonia dataset.
- `test_pneumonia.py`: Core script for running inference, evaluating model performance, and generating GradCAM visualizations on test images.
- `pneumonia_checkpoints/`: Contains the trained model weights (`pneumonia_efficientnet_b3_best.pth`).
- `pneumonia_outputs/`: Directory containing generated heatmaps, overlays, training histories, evaluation plots, and prediction JSON results for various test images.
- Sample Images: `Chest-X-ray-radiograph-with-Pneumonia.png` and `Chest-X-ray-radiograph-without-Pneumonia.png` for testing purposes.

## Technologies Used
- Deep Learning (PyTorch)
- EfficientNetB3
- GradCAM (Gradient-weighted Class Activation Mapping)
- Python
- OpenCV / PIL for image processing
- Matplotlib / Seaborn for evaluation plots

## Future Enhancements
- Expand the detection capabilities to other types of respiratory conditions (e.g., COVID-19, Tuberculosis).
- Improve the model's accuracy with a larger and more diverse dataset.
- Develop a user-friendly web interface for doctors to upload X-rays and instantly receive analysis reports.
