from fastapi import APIRouter, HTTPException
from ...schemas.assistant import ChatQueryRequest, ChatQueryResponse
from ...services.assistant_service import generate_assistant_reply

router = APIRouter(tags=["AI Assistant"])

@router.post("/assistant/chat", response_model=ChatQueryResponse, summary="Query the MedVision AI Radiologist Assistant")
@router.post("/api/assistant/chat", response_model=ChatQueryResponse, summary="Query the MedVision AI Radiologist Assistant (alias)")
async def chat_with_assistant(request: ChatQueryRequest):
    """
    Submits a query to the AI Radiologist Assistant with optional scan context.
    Uses Gemini API when configured, or the built-in Clinical Knowledge Engine.
    """
    try:
        response = await generate_assistant_reply(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI Assistant encountered an error processing query: {str(e)}"
        )
