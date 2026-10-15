from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    modality: str
    routed_to: str
    prediction: str
    probabilities: Dict[str, float]
    confidence: float
    threshold: Optional[float] = None
    model_name: str
    framework: str = "PyTorch"
    input_size: Optional[str] = None
    dataset: Optional[str] = None
    metrics: Optional[Dict[str, str]] = None
    gradcam_png_base64: Optional[str] = None
    original_png_base64: Optional[str] = None
    visualizations: Optional[Dict[str, str]] = None
    disclaimer: str
    warnings: List[str] = Field(default_factory=list)
    status: str = "success"
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    online: bool = True
    status: str = "healthy"
    device: str
    version: str
    available_modalities: List[str]

class ModalityCapability(BaseModel):
    name: str
    supported_extensions: List[str]
    tasks: List[str]
    default_pipeline: str
    description: str

class ModalitiesResponse(BaseModel):
    modalities: Dict[str, ModalityCapability]
