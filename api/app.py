"""
============================================================
AI SHOPPING ASSISTANT
FastAPI Application
============================================================

Purpose
-------

Main FastAPI application.

Responsibilities

✓ Initialize FastAPI
✓ Register API routes
✓ Configure metadata

Future

✓ Middleware
✓ Authentication
✓ Streamlit
"""

# ============================================================
# IMPORTS
# ============================================================

from fastapi import FastAPI

from api.routes import router


# ============================================================
# CREATE APP
# ============================================================

app = FastAPI(

    title="AI Shopping Assistant",

    description="""
AI Shopping Assistant powered by

• LangGraph
• LangChain
• Groq
• Gemini
• Semantic Search
• FAISS
""",

    version="1.0.0"

)


# ============================================================
# INCLUDE ROUTES
# ============================================================

app.include_router(router)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "message": "AI Shopping Assistant API",

        "version": "1.0.0",

        "status": "Running"

    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "Healthy"

    }