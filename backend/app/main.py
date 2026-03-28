"""PRISM FastAPI application."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts on startup (optional warmup)."""
    yield
    # Shutdown cleanup if needed


app = FastAPI(
    title=settings.app_name,
    description="PRISM — Human-centred Explainable AI (XAI). Decision engine, confidence, decision factors.",
    version="0.1.0",
    lifespan=lifespan,
    swagger_ui_parameters={
        "docExpansion": "list",
        "defaultModelsExpandDepth": -1,
        "filter": True,
        "displayRequestDuration": False,
    },
)
# allow_origins=["*"] requires allow_credentials=False (browser CORS spec).
# Frontend uses default fetch (no credentials) to the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
