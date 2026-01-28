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
    description="PRISM — Human-centred Explainable AI (XAI) for credit-approval decision support. Decision engine, confidence, decision factors.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
