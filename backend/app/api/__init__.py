"""Aggregated PRISM API router (metadata + core routes)."""

from fastapi import APIRouter

from app.api.meta import router as meta_router
from app.api.routes import router as prism_router
from app.api.xai import router as xai_router

router = APIRouter()
router.include_router(meta_router)
router.include_router(prism_router)
router.include_router(xai_router)

__all__ = ["router"]
