from typing import Optional
import uuid
from fastapi import APIRouter, UploadFile, File, Form
from ...schemas.prediction import PredictionResponse
from ...utils.file_validation import validate_uploaded_image
from ...utils.image_utils import bytes_to_pil_image
from ...services.xray_service import XRayService

router = APIRouter(tags=["X-Ray"])

@router.post("/predict/xray", response_model=PredictionResponse)
@router.post("/api/xray/predict", response_model=PredictionResponse)
async def predict_xray(
    file: UploadFile = File(...),
    subtype: Optional[str] = Form(None)
):
    contents = await validate_uploaded_image(file)
    image = bytes_to_pil_image(contents, file.filename or "")
    
    service = XRayService.get_instance()
    result = service.predict(image, subtype=subtype)
    result.request_id = str(uuid.uuid4())
    return result
