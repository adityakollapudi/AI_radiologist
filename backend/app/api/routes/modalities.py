from fastapi import APIRouter
from ...schemas.prediction import ModalitiesResponse, ModalityCapability

router = APIRouter(tags=["Modalities"])

MODALITIES_INFO = {
    "xray": ModalityCapability(
        name="X-Ray Radiography",
        supported_extensions=[".png", ".jpg", ".jpeg", ".bmp", ".webp"],
        tasks=["chest_pneumonia", "musculoskeletal_mura"],
        default_pipeline="Anatomical Router (Chest vs. Musculoskeletal)",
        description="High-throughput ResNet-50 routing to Chest Pneumonia Multi-Model Ensemble or Stanford MURA Specialist with Grad-CAM explainability."
    ),
    "ct": ModalityCapability(
        name="Computed Tomography (CT)",
        supported_extensions=[".png", ".jpg", ".jpeg", ".dcm"],
        tasks=["thoracic_covid19_pneumonia", "cranial_brain_bleed"],
        default_pipeline="CT Anatomy Router (Chest vs. Brain)",
        description="EfficientNet-B0 anatomy triage to Chest COVID-19/Pneumonia Specialist or SE-ResNeXt-101 6-Subtype Hemorrhage Expert with Multi-Layer Grad-CAM."
    ),
    "mri": ModalityCapability(
        name="Magnetic Resonance Imaging (MRI)",
        supported_extensions=[".png", ".jpg", ".jpeg"],
        tasks=["brain_tumor_screener", "reflex_dementia_atrophy"],
        default_pipeline="Two-Stage Cascade (Tumor Screener -> Reflex Dementia Assessment)",
        description="Custom 5-Layer CNN with 5-layer gradient-fused Grad-CAM, triggering Reflex EfficientNetV2-S Dementia Classifier when brain is normal."
    )
}

VERIFIED_MODELS_SPECS = {
    "xray_chest": {
        "name": "Chest Radiography Multi-Model Ensemble",
        "framework": "PyTorch 2.x",
        "architecture": "Weighted Ensemble (EfficientNet-V2-S + B3 + B7)",
        "input_size": "300x300, 384x384, 600x600 px",
        "dataset": "NIH ChestX-ray14 & RSNA Pneumonia",
        "threshold": 0.50,
        "metrics": {"F1 Score": "94.3%", "Accuracy": "94.8%"}
    },
    "xray_mura": {
        "name": "Stanford MURA Musculoskeletal Specialist",
        "framework": "PyTorch 2.x",
        "architecture": "EfficientNet-B3",
        "input_size": "300x300 px",
        "dataset": "Stanford MURA Dataset",
        "threshold": 0.50,
        "metrics": {"Cohen Kappa": "0.78", "Accuracy": "86.4%"}
    },
    "ct": {
        "name": "Chest CT COVID-19 / Pneumonia Specialist",
        "framework": "PyTorch 2.x",
        "architecture": "EfficientNet-B0 + Multi-Layer LayerCAM",
        "input_size": "256x256 px",
        "dataset": "COVIDx CT-2A Benchmark",
        "threshold": 0.50,
        "metrics": {"Accuracy": "96.7%", "Macro F1": "0.96"}
    },
    "ct_brain_bleed": {
        "name": "RSNA Intracranial Hemorrhage Specialist",
        "framework": "PyTorch 2.x",
        "architecture": "SE-ResNeXt-101 (32x4d)",
        "input_size": "256x256 px",
        "dataset": "RSNA 2019 Intracranial Hemorrhage Challenge",
        "threshold": 0.50,
        "metrics": {"AUROC": "0.97", "Accuracy": "94.2%"}
    },
    "mri": {
        "name": "Brain Tumor Screener (Primary)",
        "framework": "PyTorch 2.x",
        "architecture": "Custom 5-Layer CNN + 5-Layer Fused Grad-CAM",
        "input_size": "240x240 px",
        "dataset": "BraTS & Kaggle Brain MRI Dataset",
        "threshold": 0.50,
        "metrics": {"Accuracy": "91.2%", "Sensitivity": "92.0%"}
    },
    "mri_dementia": {
        "name": "Dementia Severity Staging Specialist",
        "framework": "PyTorch 2.x",
        "architecture": "EfficientNetV2-S + Temperature Calibration",
        "input_size": "384x384 px",
        "dataset": "OASIS / ADNI MRI Cohorts",
        "threshold": 0.50,
        "metrics": {"Macro F1": "0.93", "Accuracy": "93.4%"}
    }
}

@router.get("/modalities", response_model=ModalitiesResponse)
@router.get("/api/modalities", response_model=ModalitiesResponse)
async def get_modalities():
    return ModalitiesResponse(modalities=MODALITIES_INFO)

@router.get("/models")
@router.get("/api/models")
async def get_models():
    return VERIFIED_MODELS_SPECS
