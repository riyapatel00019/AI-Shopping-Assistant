"""
============================================================
AI SHOPPING ASSISTANT
FastAPI Routes
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import logging

from fastapi import APIRouter
from fastapi import HTTPException

from api.schemas import ChatRequest
from api.schemas import ChatResponse

from agent.shopping_agent import ShoppingAgent


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter()

agent = ShoppingAgent()


# ============================================================
# CHAT
# ============================================================

@router.post(

    "/chat",

    response_model=ChatResponse

)

def chat(

    request: ChatRequest

):

    """
    Main Shopping Chat Endpoint.
    """

    logger.info("=" * 60)

    logger.info("CHAT REQUEST RECEIVED")

    logger.info("=" * 60)

    logger.info(f"Query : {request.query}")

    try:

        result = agent.chat(

            request.query

        )

        logger.info("Shopping Agent Completed Successfully")

        return ChatResponse(

            response=result["response"],

            provider=result["provider"],

            success=True,

            conversation_complete=result["conversation_complete"],

            shopping_context=result["shopping_context"],

            products=result["products"],

            comparison=result["comparison"],

            bundle=result["bundle"]

        )

    except Exception as e:

        logger.exception(

            "Chat Endpoint Error"

        )

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ============================================================
# HEALTH CHECK
# ============================================================

@router.get("/health")

def health():

    """
    API Health Check
    """

    return {

        "status": "healthy",

        "service": "AI Shopping Assistant",

        "provider": agent.get_provider()

    }


# ============================================================
# ROOT
# ============================================================

@router.get("/")

def root():

    """
    Root Endpoint
    """

    return {

        "message": "AI Shopping Assistant API",

        "provider": agent.get_provider(),

        "status": "running"

    }