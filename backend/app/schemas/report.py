from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class ReportRequest(BaseModel):
    modality: str
    prediction: str
    routed_to: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    threshold: Optional[float] = None
    model_name: Optional[str] = None
    framework: Optional[str] = "PyTorch"
    input_size: Optional[str] = None
    dataset: Optional[str] = None
    metrics: Optional[Dict[str, str]] = None
    gradcam_png_base64: Optional[str] = None
    original_png_base64: Optional[str] = None
    visualizations: Optional[Dict[str, str]] = None
    disclaimer: Optional[str] = None
    warnings: Optional[List[str]] = Field(default_factory=list)
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
