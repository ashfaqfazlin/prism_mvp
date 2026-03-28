"""PRISM API metadata and human-readable documentation for clients (e.g. SHAP guide).

Kept separate from the large routes module so the API surface stays discoverable.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import settings

router = APIRouter(prefix="/api", tags=["meta"])

API_VERSION = "0.1.0"


@router.get("/meta")
def api_meta() -> dict:
    """Machine-readable API overview + SHAP explanation guide for the frontend."""
    return {
        "name": settings.app_name,
        "version": API_VERSION,
        "openapi_docs": "/docs",
        "shap_guide": {
            "title": "How to read SHAP in PRISM",
            "summary": (
                "SHAP values show how much each factor pushed the model toward one outcome "
                "or the other for this single row. They are not probabilities and do not "
                "replace the final decision."
            ),
            "points": [
                "Each bar is one factor (after preprocessing). Longer bars matter more for this prediction.",
                "Green (positive SHAP) pushes the score toward the positive class label shown for this dataset.",
                "Red (negative SHAP) pushes the score toward the negative class label.",
                "Bars are ranked by impact for this row only. Use “Overall factors” for typical importance across many rows.",
                "SHAP is an explanation of the model’s behaviour, not proof of real-world causation.",
            ],
            "x_axis": "SHAP value (contribution to model output for the positive class)",
        },
        "endpoint_groups": {
            "system": ["GET /api/health", "GET /api/meta"],
            "datasets": [
                "GET /api/datasets/catalog",
                "GET /api/datasets/{dataset_id}",
                "GET /api/datasets/{dataset_id}/global-explainability",
                "GET /api/datasets/recent/uploads",
                "GET /api/datasets/recent/{upload_id}",
                "DELETE /api/datasets/recent",
            ],
            "decisions": [
                "POST /api/decision",
                "POST /api/decision/{domain_id}",
                "POST /api/decision/{domain_id}/batch",
            ],
            "upload_train": [
                "POST /api/upload",
                "GET /api/datasets/upload/{upload_id}/analyze",
                "GET /api/datasets/upload/{upload_id}/target-info",
                "POST /api/datasets/upload/{upload_id}/configure",
                "POST /api/datasets/upload/{upload_id}/train",
                "GET /api/datasets/upload/{upload_id}/status",
                "GET /api/datasets/upload/trained",
                "DELETE /api/datasets/upload/{upload_id}",
            ],
            "export": ["POST /api/export", "POST /api/export/bulk"],
            "xai_diagnostics": [
                "GET /api/domains/{domain_id}/calibration",
                "GET /api/domains/{domain_id}/fairness",
                "GET /api/domains/{domain_id}/xai-profile",
            ],
            "legacy": ["GET /api/feature-ranges", "GET /api/decision-factor-ranges", "POST /api/predict"],
        },
    }
