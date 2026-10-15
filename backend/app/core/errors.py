from fastapi import Request, status
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("medvision")

class MedicalAIException(Exception):
    """Base exception for MedVision backend."""
    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class InvalidFileException(MedicalAIException):
    def __init__(self, message: str = "Invalid file or unsupported image format."):
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST)

class UnsupportedModalityException(MedicalAIException):
    def __init__(self, modality: str):
        super().__init__(f"Unsupported modality '{modality}'. Supported: xray, ct, mri.", status_code=status.HTTP_400_BAD_REQUEST)

class ModelNotFoundException(MedicalAIException):
    def __init__(self, message: str = "Required model checkpoint not found on server."):
        super().__init__(message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

class ModelInferenceException(MedicalAIException):
    def __init__(self, message: str = "Error encountered during model execution."):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

async def medical_ai_exception_handler(request: Request, exc: MedicalAIException):
    logger.warning(f"MedicalAIException on {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": exc.message,
            "details": None
        }
    )

async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error": "An internal error occurred during medical scan analysis. Please check inputs and retry.",
            "details": None
        }
    )
