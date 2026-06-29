"""
============================================================
AI SHOPPING ASSISTANT
FastAPI Schemas
============================================================

Purpose
-------
Request and Response models used by FastAPI.

Used By
-------
• routes.py
• shopping_agent.py
• Streamlit Frontend
"""

# ============================================================
# IMPORTS
# ============================================================

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

# ============================================================
# CHAT REQUEST
# ============================================================

class ChatRequest(BaseModel):
    """
    Incoming request from Streamlit.
    """

    query: str = Field(
        ...,
        description="User shopping query",
        examples=[
            "Recommend a laptop under ₹60,000"
        ]
    )

# ============================================================
# CHAT RESPONSE
# ============================================================

class ChatResponse(BaseModel):
    """
    Response returned to the Streamlit frontend.
    """

    # --------------------------------------------------------
    # AI Response
    # --------------------------------------------------------

    response: str = Field(
        ...,
        description="Assistant response."
    )

    provider: str = Field(
        ...,
        description="Active LLM provider."
    )

    success: bool = Field(
        default=True,
        description="Whether the request was processed successfully."
    )

    # --------------------------------------------------------
    # Conversation
    # --------------------------------------------------------

    conversation_complete: bool = Field(
        default=True,
        description="True if enough information has been collected."
    )

    # --------------------------------------------------------
    # Shopping Context
    # --------------------------------------------------------

    shopping_context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Structured shopping context extracted from the conversation."
    )

    # --------------------------------------------------------
    # Recommendation Results
    # --------------------------------------------------------

    products: List[Any] = Field(
        default_factory=list,
        description="Recommended products."
    )

    # --------------------------------------------------------
    # Comparison Results
    # --------------------------------------------------------

    comparison: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Comparison result."
    )

    # --------------------------------------------------------
    # Bundle Results
    # --------------------------------------------------------

    bundle: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Shopping bundle."
    )

# ============================================================
# PYDANTIC CONFIGURATION
# ============================================================

ChatResponse.model_rebuild()

# ============================================================
# EXPORTS
# ============================================================

__all__ = [

    "ChatRequest",

    "ChatResponse"

]