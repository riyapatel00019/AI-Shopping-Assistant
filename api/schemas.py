"""
============================================================
AI SHOPPING ASSISTANT
FastAPI Schemas
============================================================
"""

from pydantic import BaseModel


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    query: str


# ============================================================
# RESPONSE MODEL
# ============================================================

class ChatResponse(BaseModel):

    response: str

    provider: str

    success: bool