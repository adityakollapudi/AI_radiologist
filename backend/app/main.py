from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from .core.config import settings
from .core.errors import (
    MedicalAIException,
    UnsupportedModalityException,
    medical_ai_exception_handler,
    generic_exception_handler
)
from .api.routes import health, modalities, xray, ct, mri, report, assistant
from .schemas.prediction import PredictionResponse

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Unified Multimodal AI Radiology Backend for X-Ray, CT Scan, and Brain MRI Analysis",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
app.add_exception_handler(MedicalAIException, medical_ai_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Include Route Modules
app.include_router(health.router)
app.include_router(modalities.router)
app.include_router(xray.router)
app.include_router(ct.router)
app.include_router(mri.router)
app.include_router(report.router)
app.include_router(assistant.router)

# Unified Root Level Prediction Route for Frontend Compatibility
@app.post("/predict/{modality}", response_model=PredictionResponse, tags=["Unified Predict"])
async def predict_unified(
    modality: str,
    file: UploadFile = File(...),
    subtype: Optional[str] = Form(None)
):
    mod_lower = modality.lower()
    if mod_lower == "xray":
        return await xray.predict_xray(file=file, subtype=subtype)
    elif mod_lower == "ct":
        return await ct.predict_ct(file=file, subtype=subtype)
    elif mod_lower == "mri":
        return await mri.predict_mri(file=file, subtype=subtype)
    else:
        raise UnsupportedModalityException(modality)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
