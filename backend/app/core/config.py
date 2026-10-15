import os
from pathlib import Path
from typing import List
import torch

class Settings:
    PROJECT_NAME: str = "MEDVISION AI — Unified Diagnostic Backend"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Path configuration: repo root is parent of backend/
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent.parent
    REPO_ROOT: Path = BACKEND_DIR.parent
    
    # Device
    FORCE_DEVICE: str = os.getenv("DEVICE", "").lower()
    if FORCE_DEVICE in ("cuda", "cpu"):
        DEVICE: torch.device = torch.device(FORCE_DEVICE)
    else:
        DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CORS
    raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000")
    CORS_ORIGINS: List[str] = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    if "*" not in CORS_ORIGINS:
        CORS_ORIGINS.append("*")  # Allow local development

    # Upload constraints
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Output / Scratch directories
    TEMP_DIR: Path = BACKEND_DIR / "temp"

    # Disclaimer
    DEFAULT_DISCLAIMER: str = (
        "Research and educational use only. This system does not provide a clinical diagnosis "
        "and must not replace evaluation by a qualified healthcare professional. "
        "Grad-CAM heatmaps highlight model-attributed regions and are not lesion segmentation."
    )

    # AI Assistant (Gemini API with Smart Clinical Knowledge Engine fallback)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

settings = Settings()
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
