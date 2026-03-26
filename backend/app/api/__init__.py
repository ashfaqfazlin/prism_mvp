"""Aggregated PRISM API router (metadata + core routes)."""

from fastapi import APIRouter

from app.api.meta import router as meta_router
from app.api.routes import router as prism_router

router = APIRouter()
router.include_router(meta_router)
router.include_router(prism_router)

__all__ = ["router"]
