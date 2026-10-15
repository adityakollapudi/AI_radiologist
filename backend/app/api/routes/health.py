from fastapi import APIRouter
from ...core.config import settings
from ...schemas.prediction import HealthResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
@router.get("/api/health", response_model=HealthResponse)
async def get_health():
    return HealthResponse(
        online=True,
        status="healthy",
        device=str(settings.DEVICE),
        version=settings.VERSION,
        available_modalities=["xray", "ct", "mri"]
    )
