import io
from pathlib import Path
from fastapi import UploadFile
from PIL import Image

from ..core.config import settings
from ..core.errors import InvalidFileException

ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".dcm"
}

async def validate_uploaded_image(file: UploadFile) -> bytes:
    """Validates the uploaded file extension, size, and image integrity. Returns raw bytes."""
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    
    if suffix not in ALLOWED_EXTENSIONS:
        raise InvalidFileException(
            f"Unsupported file format '{suffix}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
        
    contents = await file.read()
    if not contents:
        raise InvalidFileException("The uploaded file is empty.")
        
    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise InvalidFileException(
            f"File size exceeds maximum allowed upload size ({settings.MAX_UPLOAD_SIZE_MB}MB)."
        )
        
    # Verify image decodability
    if suffix == ".dcm":
        try:
            import pydicom
            pydicom.dcmread(io.BytesIO(contents))
        except Exception as e:
            raise InvalidFileException(f"Invalid DICOM file: {str(e)}")
    else:
        try:
            with Image.open(io.BytesIO(contents)) as img:
                img.verify()
        except Exception:
            raise InvalidFileException("The file content is corrupted or not a recognized image format.")
            
    return contents
