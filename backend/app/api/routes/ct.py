from typing import Optional
import uuid
from fastapi import APIRouter, UploadFile, File, Form
from ...schemas.prediction import PredictionResponse
from ...utils.file_validation import validate_uploaded_image
from ...utils.image_utils import bytes_to_pil_image
from ...services.ct_service import CTService

router = APIRouter(tags=["CT Scan"])

@router.post("/predict/ct", response_model=PredictionResponse)
@router.post("/api/ct/predict", response_model=PredictionResponse)
async def predict_ct(
    file: UploadFile = File(...),
    subtype: Optional[str] = Form(None)
):
    contents = await validate_uploaded_image(file)
    image = bytes_to_pil_image(contents, file.filename or "")
    
    service = CTService.get_instance()
    result = service.predict(image, subtype=subtype)
    result.request_id = str(uuid.uuid4())
    return result
