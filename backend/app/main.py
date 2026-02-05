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
_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
