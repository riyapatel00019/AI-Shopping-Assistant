"""
============================================================
AI SHOPPING ASSISTANT
FastAPI Routes
============================================================
"""

from fastapi import APIRouter, HTTPException

from api.schemas import ChatRequest, ChatResponse

try:

    from agent.shopping_agent import ShoppingAgent

except ImportError:

    from shopping_agent import ShoppingAgent


router = APIRouter()

agent = ShoppingAgent()


# ============================================================
# CHAT
# ============================================================

@router.post("/chat", response_model=ChatResponse)

def chat(request: ChatRequest):

    try:

        result = agent.chat(request.query)

        messages = result["messages"]

        answer = ""

        for message in reversed(messages):

            if hasattr(message, "content"):

                answer = message.content

                break

        return ChatResponse(

            response=answer,

            provider=agent.get_provider(),

            success=True

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )