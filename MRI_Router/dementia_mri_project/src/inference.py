"""Single-image inference: quality check -> preprocessing -> four-class probabilities -> calibrated confidence,
uncertainty (entropy, optional MC dropout) -> Grad-CAM. Output wording: model confidence / model uncertainty."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import load_model
from .preprocessing import load_rgb, model_input_image, normalize_tensor, quality_check
from .uncertainty import mc_dropout_probs, mc_dropout_summary, normalized_entropy, uncertainty_status
from .utils import DISCLAIMER, get_device, load_json

DEFAULT_UNC = {"low_confidence": 0.80, "low_entropy": 0.35, "high_confidence": 0.60, "high_entropy": 0.65}


class DementiaPredictor:
    def __init__(self, checkpoint, meta_path=None, device=None):
        self.device = device or get_device()
        self.model, self.ck = load_model(checkpoint, self.device)
        meta_path = Path(meta_path) if meta_path else Path(checkpoint).with_name("inference_meta.json")
        self.meta = load_json(meta_path, {}) or {}
        self.T = float(self.meta.get("temperature") or 1.0)
        self.unc = {"uncertainty": self.meta.get("uncertainty_thresholds") or DEFAULT_UNC}
        self.classes = self.ck["class_names"]
        self.spec = self.ck["preprocessing"]
        self.calibrated = "temperature" in self.meta
        if not self.calibrated:
            print(f"!! {meta_path} not found: probabilities are NOT temperature-calibrated (run the evaluation stage).")

    def predict(self, image_path, gradcam_dir=None, target_class=None, mc_passes=0, multi_layer=True):
        img = load_rgb(image_path)
        ok, warnings = quality_check(img)
        t0 = time.time()
        inp = model_input_image(img, self.spec)
        x = normalize_tensor(torch.from_numpy(np.asarray(inp, dtype=np.float32) / 255.0).permute(2, 0, 1), self.spec)[None].to(self.device)
        with torch.no_grad():
            logits = self.model(x).float()
            prob = torch.softmax(logits / self.T, 1)[0].cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        k = int(prob.argmax())
        ne = float(normalized_entropy(prob[None])[0])
        res = {"task": "dementia_mri_classification", "prediction": self.classes[k], "confidence": float(prob[k]),
               "uncertainty": ne, "uncertainty_measure": "normalized predictive entropy (0 = certain, 1 = uniform)",
               "uncertainty_status": uncertainty_status(float(prob[k]), ne, self.unc),
               "probabilities": {c: float(p) for c, p in zip(self.classes, prob)},
               "top3": [self.classes[i] for i in np.argsort(-prob)[:3]],
               "model": self.ck["model_display"], "input_size": [self.spec["image_size"]] * 2,
               "temperature_calibrated": self.calibrated, "temperature": self.T,
               "inference_ms": ms, "image_quality_ok": ok, "image_quality_warnings": warnings,
               "image": str(image_path), "disclaimer": DISCLAIMER}
        if mc_passes:
            mean, ps = mc_dropout_probs(self.model, x, mc_passes, self.T)
            s = mc_dropout_summary(ps.cpu().numpy())
            res["mc_dropout"] = {"passes": mc_passes, "mutual_information": float(s["mutual_information"][0]),
                                 "mean_probabilities": {c: float(p) for c, p in zip(self.classes, s["mean_prob"][0])}}
        res["gradcam"] = None
        if gradcam_dir is not None:
            from .gradcam import explain_image
            tgt = None if target_class is None else self.classes.index(target_class)
            info, _, _ = explain_image(self.model, image_path, self.spec, self.classes, target=tgt, out_dir=gradcam_dir,
                                       name=Path(image_path).stem + "_gradcam", multi=multi_layer)
            res["gradcam"] = info["files"]["overlay"]
            res["gradcam_panel"] = info["files"]["panel"]
            res["gradcam_target_class"] = info["target_class"]
            res["gradcam_checks_passed"] = info["checks"]["all_layer_checks_passed"]
        return res


def format_report(r):
    L = ["=" * 60, "DEMENTIA MRI ANALYSIS", "=" * 60, "",
         "Prediction:", r["prediction"], "",
         "Model Confidence:", f"{r['confidence'] * 100:.2f}%", "",
         "Model Uncertainty:", f"{r['uncertainty']:.2f}  ({r['uncertainty_status']})", "",
         "CLASS PROBABILITIES:", ""]
    for c, p in r["probabilities"].items():
        L += [f"{c}:", f"{p * 100:.2f}%", ""]
    L += ["TOP-3:", ""] + [f"{i + 1}. {c}" for i, c in enumerate(r["top3"])] + ["",
          "Inference Time:", f"{r['inference_ms']:.0f} ms", "", "Grad-CAM:", str(r["gradcam"] or "not generated"), ""]
    if r.get("mc_dropout"):
        L += ["MC-dropout mutual information:", f"{r['mc_dropout']['mutual_information']:.4f}", ""]
    if r["image_quality_warnings"]:
        L += ["Image quality warnings:"] + [f"  - {w}" for w in r["image_quality_warnings"]] + [""]
    if not r["temperature_calibrated"]:
        L += ["(probabilities not temperature-calibrated)", ""]
    L += [r["disclaimer"], "=" * 60]
    return "\n".join(L)


def save_prediction_json(r, path):
    keys = ["task", "prediction", "confidence", "uncertainty", "probabilities", "model", "input_size", "gradcam"]
    out = {k: r[k] for k in keys}
    out.update({k: r[k] for k in r if k not in keys})
    with open(path, "w") as f:
        json.dump(out, f, indent=4)
    return path
