# AI Radiologist - Foreign Object Detection

## Overview
The **AI Radiologist** project is a deep learning-based diagnostic tool designed to assist in analyzing medical X-ray images. This specific module of the project focuses on **Foreign Object Detection** in X-rays, identifying the presence of objects such as nails, metal plates, screws, and other implants or anomalies.

## Features
- **Foreign Object Detection:** Automatically detects foreign objects in X-ray scans.
- **Explainable AI (GradCAM):** Generates GradCAM (Gradient-weighted Class Activation Mapping) overlay images to highlight the exact regions the model focused on when making its predictions. This provides transparency and builds trust in the AI's diagnostic suggestions.
- **EfficientNet Architecture:** Leverages the powerful and efficient `EfficientNetB3` deep neural network for high-accuracy image classification.

## Project Structure
- `MURA_EfficientNetB3_GradCAM.py`: Core script for running the model and generating GradCAM visualizations.
- `mura_checkpoints/`: Contains the trained model weights (`mura_efficientnet_b3_best.pth`).
- `gradcam_results/` & `mura_outputs/`: Directories containing the generated heatmaps, overlays, and prediction JSON results for various test images.

## Technologies Used
- Deep Learning (PyTorch/TensorFlow - based on `.pth` weights, it uses PyTorch)
- EfficientNetB3
- GradCAM (Gradient-weighted Class Activation Mapping)
- Python
- OpenCV / PIL for image processing

## Future Enhancements
- Expand the detection capabilities to other types of radiological abnormalities (e.g., fractures, tumors).
- Improve the model's accuracy with a larger and more diverse dataset.
- Develop a user-friendly web interface for doctors to upload X-rays and instantly receive analysis reports.
