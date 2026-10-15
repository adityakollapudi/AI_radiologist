import re
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Response, HTTPException
from ...schemas.report import ReportRequest
from ...services.pdf_service import generate_medical_pdf_report
from ...core.errors import MedicalAIException

router = APIRouter(tags=["Report"])

def _sanitize_filename(name: str) -> str:
    """Sanitizes filename string to safe characters."""
    return re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)

@router.post("/report/pdf", summary="Generate downloadable medical PDF report")
@router.post("/api/report/pdf", summary="Generate downloadable medical PDF report (alias)")
async def export_pdf_report(request: ReportRequest):
    """
    Accepts the multimodal AI prediction result and generates a professional,
    clinical-grade A4 PDF report with side-by-side scan comparison.
    """
    try:
        pdf_bytes = generate_medical_pdf_report(request)
        
        # Format a clean filename
        report_id = request.request_id or f"MEDV-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        safe_id = _sanitize_filename(report_id)
        filename = f"MEDVISION_AI_Report_{safe_id}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except MedicalAIException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate medical PDF report: {str(e)}"
        )
