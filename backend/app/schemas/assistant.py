from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text content")

class ChatQueryRequest(BaseModel):
    message: str = Field(..., description="User query or question")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional active scan analysis context")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Recent conversation turns")

class ChatQueryResponse(BaseModel):
    reply: str
    source: str = Field(..., description="'gemini' or 'clinical_engine'")
    suggested_queries: List[str] = Field(default_factory=list)
    timestamp: str
    disclaimer: str
