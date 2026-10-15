"""Brain Tumor Detection & Multi-Layer Gradient-Fused Grad-CAM Localization.
Extracts intermediate activations and backpropagated pre-sigmoid gradients from all 5 CNN layers:
  ['conv0', 'bn0', 'activation', 'max_pool0', 'max_pool1']
Applies anatomical parenchyma masking, channel-wise CAM weighting, spatial projection,
hierarchical multi-layer fusion, and lesion intensity modulation.
"""
from pathlib import Path
import cv2
import h5py
import imutils
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent


def find_tumor_model_path():
    """Searches common locations for the brain tumor model weights."""
    candidates = [
        ROOT.parent / "brain-_tumor_detection" / "models" / "cnn-parameters-improvement-23-0.91.h5",
        ROOT / "models" / "cnn-parameters-improvement-23-0.91.h5",
        Path(r"C:\Users\adity\Downloads\dementia_mri_project\brain-_tumor_detection\models\cnn-parameters-improvement-23-0.91.h5"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


class BrainTumorPyTorchModel(nn.Module):
    """PyTorch drop-in matching the Keras BrainDetectionModel architecture."""
    def __init__(self, h5_path):
        super().__init__()
        self.pad = nn.ZeroPad2d(2)
        self.conv0 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.bn0 = nn.BatchNorm2d(32, eps=1e-3)
        self.relu = nn.ReLU()
        self.pool0 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc = nn.Linear(6272, 1)

        self._load_from_h5(h5_path)
        self.eval()

    def _load_from_h5(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            w = f['model_weights']
            conv_k = np.array(w['conv0']['conv0']['kernel:0']).transpose(3, 2, 0, 1)
            conv_b = np.array(w['conv0']['conv0']['bias:0'])
            self.conv0.weight.data.copy_(torch.from_numpy(conv_k))
            self.conv0.bias.data.copy_(torch.from_numpy(conv_b))

            self.bn0.weight.data.copy_(torch.from_numpy(np.array(w['bn0']['bn0']['gamma:0'])))
            self.bn0.bias.data.copy_(torch.from_numpy(np.array(w['bn0']['bn0']['beta:0'])))
            self.bn0.running_mean.copy_(torch.from_numpy(np.array(w['bn0']['bn0']['moving_mean:0'])))
            self.bn0.running_var.copy_(torch.from_numpy(np.array(w['bn0']['bn0']['moving_variance:0'])))

            fc_k = np.array(w['fc']['fc']['kernel:0']).T
            fc_b = np.array(w['fc']['fc']['bias:0'])
            self.fc.weight.data.copy_(torch.from_numpy(fc_k))
            self.fc.bias.data.copy_(torch.from_numpy(fc_b))

    def forward(self, x):
        logit, prob, _ = self.forward_layers(x)
        return prob

    def forward_layers(self, x):
        """Passes input through all layers and exposes intermediate activations for Grad-CAM."""
        x = self.pad(x)
        c0 = self.conv0(x)
        b0 = self.bn0(c0)
        act = self.relu(b0)
        p0 = self.pool0(act)
        p1 = self.pool1(p0)
        flat = p1.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
        logit = self.fc(flat)
        prob = torch.sigmoid(logit)
        layers = {
            'conv0': c0,
            'bn0': b0,
            'activation': act,
            'max_pool0': p0,
            'max_pool1': p1
        }
        return logit, prob, layers


def crop_brain_contour(image):
    """Finds brain contour and returns cropped brain mass plus bounding coordinates."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        return image, None

    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    cropped = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    if cropped.size == 0:
        return image, None
    box = (int(extLeft[0]), int(extTop[1]), int(extRight[0]), int(extBot[1]))
    return cropped, box


def compute_multilayer_gradcam(layer_dict, brain_mask, has_tumor, gray_cropped, target_h, target_w):
    """Fuses multi-layer gradient-weighted feature activations across all 5 layers."""
    layer_weights = {
        'max_pool1': 0.35,  # Coarse semantic receptive field
        'max_pool0': 0.30,  # Mid-level lesion localization
        'activation': 0.20, # Non-linear structural boundaries
        'bn0': 0.10,        # Normalized channel boundaries
        'conv0': 0.05       # Fine-grained edge details
    }

    fused_heatmap = np.zeros((target_h, target_w), dtype=np.float32)

    for name, act_t in layer_dict.items():
        if act_t.grad is None:
            continue
        # Convert torch tensors to (H, W, C) numpy arrays
        grads = act_t.grad[0].permute(1, 2, 0).detach().cpu().numpy()
        acts = act_t[0].permute(1, 2, 0).detach().cpu().numpy()
        h_l, w_l = acts.shape[:2]

        m_l = cv2.resize(brain_mask, (w_l, h_l), interpolation=cv2.INTER_NEAREST)
        m_g = grads * m_l[..., np.newaxis]
        m_a = acts * m_l[..., np.newaxis]

        # Positive gradients: features that increase tumor logit
        pos_g = np.maximum(m_g, 0.0)

        # 1. Channel-wise Grad-CAM weight: GAP over positive gradients
        alpha = np.mean(pos_g, axis=(0, 1))
        cam = np.zeros((h_l, w_l), dtype=np.float32)
        for k in range(acts.shape[-1]):
            if alpha[k] > 0:
                cam += alpha[k] * np.maximum(m_a[:, :, k], 0.0)

        # 2. Element-wise spatial gradient projection
        proj = np.sum(pos_g * np.maximum(m_a, 0.0), axis=-1)

        if cam.max() > 0:
            cam = cam / cam.max()
        if proj.max() > 0:
            proj = proj / proj.max()

        combined_layer = 0.5 * cam + 0.5 * proj
        res = cv2.resize(combined_layer, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        blur_k = max(9, int(min(target_h, target_w) * 0.06) | 1)
        res = cv2.GaussianBlur(res, (blur_k, blur_k), 0)
        if res.max() > 0:
            res = res / res.max()

        w_layer = layer_weights.get(name, 0.20)
        fused_heatmap += w_layer * res

    # Mask outer background & skull boundary
    fused_heatmap = fused_heatmap * brain_mask

    # If tumor is detected, modulate with lesion intensity prior to concentrate on the tumor core
    if has_tumor:
        blurred = cv2.GaussianBlur(gray_cropped, (9, 9), 0)
        bp = blurred[brain_mask > 0]
        if len(bp) > 0:
            med = np.median(bp)
            std = np.std(bp) + 1e-5
            z = np.maximum((blurred.astype(np.float32) - med) / std, 0.0) * brain_mask
            if z.max() > 0:
                z = z / z.max()
                fused_heatmap = fused_heatmap * (0.3 + 1.2 * z)
                fused_heatmap = fused_heatmap * brain_mask

    final_blur = max(7, int(min(target_h, target_w) * 0.04) | 1)
    fused_heatmap = cv2.GaussianBlur(fused_heatmap, (final_blur, final_blur), 0)

    # Normalize heatmap
    if fused_heatmap.max() > 0:
        if has_tumor:
            p99 = np.percentile(fused_heatmap[fused_heatmap > 0], 99.0)
            fused_heatmap = np.clip(fused_heatmap, 0.0, p99) / (p99 + 1e-10)
        else:
            fused_heatmap = (fused_heatmap / fused_heatmap.max()) * 0.20
    else:
        fused_heatmap = np.zeros_like(fused_heatmap)

    return fused_heatmap


class BrainTumorClassifier:
    """Wrapper managing the Brain Tumor CNN model and Multi-Layer Grad-CAM."""
    _instance = None

    def __init__(self, model_path=None):
        path = model_path or find_tumor_model_path()
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"Brain tumor model not found at {path}")
        self.model_path = Path(path)
        self.model = BrainTumorPyTorchModel(self.model_path)

    @classmethod
    def get_instance(cls, model_path=None):
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def predict(self, image_path, out_dir=None):
        """Runs brain tumor classification with multi-layer fused Grad-CAM."""
        image_path = str(image_path)
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise ValueError(f"Could not read image from {image_path}")

        cropped, box = crop_brain_contour(orig_img)
        target_h, target_w = cropped.shape[:2]

        resized = cv2.resize(cropped, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
        norm_img = resized.astype(np.float32) / 255.0

        tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).requires_grad_(True)

        logit, prob_tensor, layer_dict = self.model.forward_layers(tensor)
        for v in layer_dict.values():
            v.retain_grad()

        # Pre-sigmoid logit backpropagation keeps gradients non-saturated at high probabilities
        self.model.zero_grad()
        logit[:, 0].backward()

        prob = float(prob_tensor.detach().cpu().item())
        has_tumor = prob >= 0.5
        confidence = prob if has_tumor else (1.0 - prob)

        gradcam_overlay_path = None
        gradcam_panel_path = None

        if out_dir is not None:
            try:
                out_path = Path(out_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                stem = Path(image_path).stem

                # Brain parenchyma mask
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
                k_erode = max(11, int(min(target_h, target_w) * 0.05) | 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode))
                brain_mask = cv2.erode(thresh, kernel, iterations=1).astype(np.float32) / 255.0

                fused_heatmap = compute_multilayer_gradcam(
                    layer_dict=layer_dict,
                    brain_mask=brain_mask,
                    has_tumor=has_tumor,
                    gray_cropped=gray,
                    target_h=target_h,
                    target_w=target_w
                )

                # Soft thresholding (removes noise floor)
                threshold = 0.15
                enhanced = np.maximum(fused_heatmap - threshold, 0.0) / (1.0 - threshold + 1e-10)
                enhanced = np.power(enhanced, 0.75)
                heatmap_uint8 = np.uint8(255 * np.nan_to_num(enhanced, nan=0.0))
                colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                overlay = cv2.addWeighted(colored_heatmap, 0.45, cropped, 0.55, 0)

                # Overlay on full original MRI
                full_overlay = orig_img.copy()
                if box is not None:
                    x1, y1, x2, y2 = box
                    full_overlay[y1:y2, x1:x2] = overlay
                else:
                    full_overlay = overlay

                # Save overlay (1:1 aligned with original MRI)
                gradcam_overlay_file = out_path / f"{stem}_tumor_gradcam.png"
                cv2.imwrite(str(gradcam_overlay_file), full_overlay)
                gradcam_overlay_path = str(gradcam_overlay_file)

                # Save cropped brain slice
                gradcam_cropped_file = out_path / f"{stem}_tumor_cropped.png"
                cv2.imwrite(str(gradcam_cropped_file), cropped)
                gradcam_cropped_path = str(gradcam_cropped_file)

                # Save pure activation heatmap (aligned to full MRI coordinate space)
                full_heatmap = np.zeros_like(orig_img)
                if box is not None:
                    x1, y1, x2, y2 = box
                    full_heatmap[y1:y2, x1:x2] = colored_heatmap
                else:
                    full_heatmap = colored_heatmap
                gradcam_heatmap_file = out_path / f"{stem}_tumor_heatmap.png"
                cv2.imwrite(str(gradcam_heatmap_file), full_heatmap)
                gradcam_heatmap_path = str(gradcam_heatmap_file)

                # 4-panel diagnostic view: [Original Scan] | [Cropped Brain] | [Pure Heatmap] | [Grad-CAM Overlay]
                panel_h = 260
                def fit_h(im):
                    ratio = panel_h / im.shape[0]
                    return cv2.resize(im, (max(1, int(im.shape[1] * ratio)), panel_h))

                panel = np.hstack([fit_h(orig_img), fit_h(cropped), fit_h(colored_heatmap), fit_h(overlay)])
                gradcam_panel_file = out_path / f"{stem}_tumor_gradcam_panel.png"
                cv2.imwrite(str(gradcam_panel_file), panel)
                gradcam_panel_path = str(gradcam_panel_file)

            except Exception as e:
                import traceback
                print(f"Warning: Failed to generate multi-layer tumor Grad-CAM: {e}")
                traceback.print_exc()

        return {
            "has_tumor": has_tumor,
            "probability": prob,
            "confidence": confidence,
            "label": "BRAIN TUMOR DETECTED (YES)" if has_tumor else "NO BRAIN TUMOR (NORMAL BRAIN)",
            "status_code": "TUMOR_POSITIVE" if has_tumor else "TUMOR_NEGATIVE",
            "model_path": str(self.model_path),
            "gradcam_overlay": gradcam_overlay_path,
            "gradcam_cropped": locals().get("gradcam_cropped_path"),
            "gradcam_heatmap": locals().get("gradcam_heatmap_path"),
            "gradcam_panel": gradcam_panel_path,
            "layers_fused": ['conv0', 'bn0', 'activation', 'max_pool0', 'max_pool1']
        }
