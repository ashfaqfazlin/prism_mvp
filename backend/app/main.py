"""PRISM FastAPI application."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router
from app.config import settings
from app.cors_middleware import AllowAnyOriginCorsMiddleware


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
# ASGI middleware: short-circuit OPTIONS + Access-Control-Allow-Origin on all responses.
app.add_middleware(AllowAnyOriginCorsMiddleware)
app.include_router(router)
