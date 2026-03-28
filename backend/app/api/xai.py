"""XAI-focused diagnostic endpoints (calibration, fairness, limitations)."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException

from app.domain_config import get_domain, is_model_trained

router = APIRouter(prefix="/api", tags=["xai"])


def _load_meta(domain_id: str) -> dict[str, Any]:
    domain = get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")
    if not is_model_trained(domain_id):
        raise HTTPException(status_code=400, detail=f"Model not trained for domain '{domain_id}'")
    meta_path = domain.artifacts_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Meta artifact not found for '{domain_id}'")
    with open(meta_path, "r") as f:
        return json.load(f)


@router.get("/domains/{domain_id}/calibration")
def get_domain_calibration(domain_id: str) -> dict[str, Any]:
    """Return empirical calibration profile for a trained domain."""
    meta = _load_meta(domain_id)
    calibration = meta.get("calibration") or {}
    return {
        "domain_id": domain_id,
        "ece": calibration.get("ece"),
        "bins": calibration.get("bins", []),
        "brier_score": meta.get("brier_score"),
        "roc_auc": meta.get("roc_auc"),
        "notes": [
            "Calibration shows how predicted confidence aligns with observed outcomes.",
            "Lower ECE and lower Brier score indicate better probabilistic reliability.",
        ],
    }


@router.get("/domains/{domain_id}/fairness")
def get_domain_fairness(domain_id: str) -> dict[str, Any]:
    """Return descriptive group fairness diagnostics for configured sensitive attributes."""
    meta = _load_meta(domain_id)
    fairness = meta.get("fairness") or {}
    return {
        "domain_id": domain_id,
        "sensitive_features": fairness.get("sensitive_features", []),
        "groups": fairness.get("groups", {}),
        "notes": fairness.get("notes", []),
        "disclaimer": (
            "These metrics are descriptive diagnostics from held-out test data and "
            "do not prove causality or legal compliance by themselves."
        ),
    }


@router.get("/domains/{domain_id}/xai-profile")
def get_domain_xai_profile(domain_id: str) -> dict[str, Any]:
    """Combined XAI diagnostics for reporting and UI consumption."""
    meta = _load_meta(domain_id)
    calibration = meta.get("calibration") or {}
    fairness = meta.get("fairness") or {}
    return {
        "domain_id": domain_id,
        "calibration": {
            "ece": calibration.get("ece"),
            "bins": calibration.get("bins", []),
            "brier_score": meta.get("brier_score"),
            "roc_auc": meta.get("roc_auc"),
        },
        "fairness": {
            "sensitive_features": fairness.get("sensitive_features", []),
            "groups": fairness.get("groups", {}),
            "notes": fairness.get("notes", []),
        },
        "limitations": [
            "Local SHAP explanations describe model behaviour, not causal effects.",
            "Global SHAP importance is aggregated and not personalized to one case.",
            "Fairness metrics depend on available labels and selected sensitive attributes.",
        ],
    }
