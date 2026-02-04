"""API routes for PRISM."""
from __future__ import annotations

import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models import (
    CounterfactualPreviewItem,
    CounterfactualResponse,
    DecisionResponse,
    ExplanationLayerResponse,
    ExportRequest,
    FeedbackCreate,
    InteractionLog,
    PostStudyQuestionnaire,
    RichInteraction,
    ShapValuesResponse,
    StudyMetrics,
    StudySessionCreate,
    StudySessionResponse,
    TaskResponse,
    TrustCalibration,
    UncertaintyResponse,
)
from app.services import (
    counterfactual_preview,
    data_service,
    explainability_service,
    get_feature_ranges,
    model_service,
    plain_language_explanations,
    study_service,
    uncertainty_stability,
)
from app.services.dataset_service import dataset_service
from app.services.domain_model_service import domain_model_service
from app.services.domain_explainability_service import domain_explainability_service
from app.domain_config import get_domain, is_model_trained, list_domains

router = APIRouter(prefix="/api", tags=["prism"])
logger = logging.getLogger("prism")
LOG_DIR = Path(settings.base_dir) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_PATH = LOG_DIR / "audit.jsonl"


def _audit(event: str, payload: dict[str, Any] | None = None) -> None:
    line = {"ts": datetime.utcnow().isoformat(), "event": event, **(payload or {})}
    try:
        with open(AUDIT_PATH, "a") as f:
            f.write(json.dumps(line, default=str) + "\n")
    except Exception as e:
        logger.warning("audit write failed: %s", e)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": "PRISM"}


@router.get("/decision-factor-ranges")
def decision_factor_ranges() -> dict[str, Any]:
    """PRISM: Min/max for numeric decision factors (sliders). From default dataset."""
    try:
        return get_feature_ranges()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="PRISM default dataset not found. Run train_model.py.")


@router.get("/feature-ranges")
def feature_ranges() -> dict[str, Any]:
    """Alias for /decision-factor-ranges. Deprecated."""
    return decision_factor_ranges()


@router.get("/datasets/default")
def get_default_dataset(limit: int = 50) -> dict[str, Any]:
    """PRISM: Sample rows from UCI Credit dataset and decision-factor ranges for sliders."""
    try:
        df = data_service.load_default_dataset()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    n = min(max(1, limit), len(df))
    sample = df.head(n)
    ranges = get_feature_ranges()
    return {
        "columns": list(sample.columns),
        "rows": sample.fillna("").to_dict(orient="records"),
        "total": len(df),
        "feature_ranges": ranges,
        "decision_factor_ranges": ranges,
    }


@router.post("/upload")
def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Validate uploaded CSV. Returns validation result and row count."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV file required")
    raw = file.file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {settings.max_upload_mb} MB)")
    df, errs = data_service.validate_csv(raw, max_records=settings.max_records)
    _audit("upload", {"filename": file.filename, "rows": len(df), "errors": errs})
    if df.empty and errs:
        raise HTTPException(status_code=400, detail="; ".join(errs))
    rows = df.fillna("").head(100).to_dict(orient="records")
    
    # Track in recent uploads
    upload_entry = dataset_service.add_recent_upload(
        filename=file.filename,
        row_count=len(df),
        columns=list(df.columns),
        sample_rows=rows,
    )
    
    return {
        "ok": True, 
        "errors": errs, 
        "row_count": len(df), 
        "columns": list(df.columns), 
        "sample": rows,
        "upload_id": upload_entry["id"],
    }


# ============== DATASET CATALOG ROUTES ==============

@router.get("/datasets/catalog")
def get_dataset_catalog(featured_only: bool = False) -> dict[str, Any]:
    """Get list of available pre-loaded datasets."""
    datasets = dataset_service.get_catalog(featured_only=featured_only)
    return {
        "datasets": datasets,
        "count": len(datasets),
    }


@router.get("/datasets/{dataset_id}")
def load_dataset(dataset_id: str, limit: int = 80) -> dict[str, Any]:
    """Load a dataset by ID and return sample rows."""
    try:
        result = dataset_service.load_dataset(dataset_id, limit=limit)
        _audit("load_dataset", {"dataset_id": dataset_id, "rows": len(result.get("rows", []))})
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/datasets/{dataset_id}/global-explainability")
def get_global_explainability(dataset_id: str, sample_size: int = 50) -> dict[str, Any]:
    """Dataset-level explainability: mean absolute SHAP across a sample (global feature importance)."""
    domain = get_domain(dataset_id)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    if not is_model_trained(dataset_id):
        return {
            "dataset_id": dataset_id,
            "feature_names": [],
            "mean_abs_shap": [],
            "feature_labels": {},
            "sample_size": 0,
            "message": "Model not trained for this dataset",
        }
    try:
        result = domain_explainability_service.global_feature_importance(dataset_id, sample_size=sample_size)
        result["dataset_id"] = dataset_id
        return result
    except Exception as e:
        logger.warning("Global explainability failed for %s: %s", dataset_id, e)
        return {
            "dataset_id": dataset_id,
            "feature_names": [],
            "mean_abs_shap": [],
            "feature_labels": {},
            "sample_size": 0,
            "error": str(e),
        }


@router.get("/datasets/recent/uploads")
def get_recent_uploads() -> dict[str, Any]:
    """Get list of recent user uploads."""
    uploads = dataset_service.get_recent_uploads()
    return {
        "uploads": uploads,
        "count": len(uploads),
    }


@router.get("/datasets/recent/{upload_id}")
def load_recent_upload(upload_id: str) -> dict[str, Any]:
    """Load a recent upload by ID."""
    upload = dataset_service.get_recent_upload(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail=f"Upload '{upload_id}' not found")
    
    return {
        "upload_id": upload["id"],
        "filename": upload["filename"],
        "uploaded_at": upload["uploaded_at"],
        "columns": upload["columns"],
        "rows": upload["sample_rows"],
        "row_count": upload["row_count"],
    }


@router.delete("/datasets/recent")
def clear_recent_uploads() -> dict[str, str]:
    """Clear all recent uploads."""
    dataset_service.clear_recent_uploads()
    return {"status": "cleared"}


# ============== UPLOAD TRAINING ROUTES ==============

from app.services.auto_schema_service import auto_schema_service
from app.services.dynamic_domain_service import dynamic_domain_service
from app.services.auto_trainer_service import auto_trainer_service
from pydantic import BaseModel, Field


class UploadConfigureRequest(BaseModel):
    """Request to configure an upload for training."""
    target_col: str
    positive_value: Any
    negative_value: Any
    positive_label: str = "Positive"
    negative_label: str = "Negative"
    name: str | None = None
    description: str | None = None


@router.get("/datasets/upload/{upload_id}/analyze")
def analyze_upload(upload_id: str) -> dict[str, Any]:
    """Analyze an uploaded dataset to detect schema and suggest target."""
    upload = dataset_service.get_recent_upload(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail=f"Upload '{upload_id}' not found")
    
    # Create DataFrame from stored sample rows
    df = pd.DataFrame(upload["sample_rows"])
    
    # Analyze schema
    analysis = dynamic_domain_service.analyze_upload(upload_id, df)
    
    return analysis


@router.get("/datasets/upload/{upload_id}/target-info")
def get_target_info(upload_id: str, target_col: str) -> dict[str, Any]:
    """Get information about a potential target column."""
    upload = dataset_service.get_recent_upload(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail=f"Upload '{upload_id}' not found")
    
    df = pd.DataFrame(upload["sample_rows"])
    
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_col}' not found")
    
    info = dynamic_domain_service.get_target_info(upload_id, df, target_col)
    
    # Add suggested labels based on data
    labels = auto_schema_service.suggest_labels(df)
    info["suggested_labels"] = labels
    
    return info


@router.post("/datasets/upload/{upload_id}/configure")
def configure_upload(upload_id: str, config: UploadConfigureRequest) -> dict[str, Any]:
    """Configure an upload for training by specifying target and labels."""
    upload = dataset_service.get_recent_upload(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail=f"Upload '{upload_id}' not found")
    
    df = pd.DataFrame(upload["sample_rows"])
    
    if config.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{config.target_col}' not found")
    
    # Create dynamic domain configuration
    domain_config = dynamic_domain_service.create_domain(
        upload_id=upload_id,
        df=df,
        target_col=config.target_col,
        positive_value=config.positive_value,
        negative_value=config.negative_value,
        positive_label=config.positive_label,
        negative_label=config.negative_label,
        name=config.name or upload.get("filename", "Custom Dataset"),
        description=config.description,
    )
    
    return {
        "status": "configured",
        "upload_id": upload_id,
        "domain_id": domain_config.domain_id,
        "config": {
            "name": domain_config.name,
            "target_col": domain_config.target_col,
            "feature_cols": domain_config.feature_cols,
            "categorical_cols": domain_config.categorical_cols,
            "numeric_cols": domain_config.numeric_cols,
            "positive_label": domain_config.positive_label,
            "negative_label": domain_config.negative_label,
        },
    }


@router.post("/datasets/upload/{upload_id}/train")
def train_upload(upload_id: str, async_mode: bool = False) -> dict[str, Any]:
    """Train a model on the configured upload."""
    # Get domain config
    domain_config = dynamic_domain_service.get_domain(upload_id)
    if not domain_config:
        raise HTTPException(
            status_code=400, 
            detail=f"Upload '{upload_id}' not configured. Call /configure first."
        )
    
    # Get upload data
    upload = dataset_service.get_recent_upload(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail=f"Upload '{upload_id}' not found")
    
    df = pd.DataFrame(upload["sample_rows"])
    
    # Start training
    if async_mode:
        job = auto_trainer_service.start_training(upload_id, df, domain_config)
    else:
        job = auto_trainer_service.start_training_sync(upload_id, df, domain_config)
    
    return {
        "status": job.status.value,
        "job_id": job.job_id,
        "upload_id": upload_id,
        "domain_id": domain_config.domain_id,
        "accuracy": job.accuracy,
        "progress": job.progress,
        "error": job.error,
    }


@router.get("/datasets/upload/{upload_id}/status")
def get_training_status(upload_id: str) -> dict[str, Any]:
    """Get the training status for an upload."""
    # Check if configured
    domain_config = dynamic_domain_service.get_domain(upload_id)
    if not domain_config:
        return {
            "status": "not_configured",
            "upload_id": upload_id,
            "is_trained": False,
        }
    
    # Check for active training job
    job = auto_trainer_service.get_job_by_upload(upload_id)
    
    return {
        "status": job.status.value if job else ("trained" if domain_config.is_trained else "configured"),
        "upload_id": upload_id,
        "domain_id": domain_config.domain_id,
        "is_trained": domain_config.is_trained,
        "accuracy": domain_config.training_accuracy or (job.accuracy if job else None),
        "progress": job.progress if job else (100 if domain_config.is_trained else 0),
        "error": job.error if job else None,
    }


@router.get("/datasets/upload/trained")
def list_trained_uploads() -> dict[str, Any]:
    """List all trained dynamic domains."""
    domains = dynamic_domain_service.list_domains()
    trained = [d for d in domains if d["is_trained"]]
    return {
        "domains": trained,
        "count": len(trained),
    }


@router.delete("/datasets/upload/{upload_id}")
def delete_upload_domain(upload_id: str) -> dict[str, str]:
    """Delete a dynamic domain and its trained model."""
    deleted = dynamic_domain_service.delete_domain(upload_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Domain for upload '{upload_id}' not found")
    return {"status": "deleted"}


# ============== DOMAIN/MODEL ROUTES ==============

@router.get("/domains")
def list_available_domains() -> dict[str, Any]:
    """List all available domains with training status."""
    domains = domain_model_service.list_available_domains()
    current = domain_model_service.current_domain_id
    return {
        "domains": domains,
        "current_domain": current,
    }


@router.get("/domains/{domain_id}")
def get_domain_info(domain_id: str) -> dict[str, Any]:
    """Get info about a specific domain."""
    domain = get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")
    
    trained = is_model_trained(domain_id)
    return {
        "id": domain.id,
        "name": domain.name,
        "description": domain.description,
        "trained": trained,
        "feature_cols": domain.feature_cols,
        "feature_labels": domain.feature_labels,
        "positive_label": domain.positive_label,
        "negative_label": domain.negative_label,
        "categorical_cols": domain.categorical_cols,
        "numeric_cols": domain.numeric_cols,
    }


@router.post("/domains/{domain_id}/activate")
def activate_domain(domain_id: str) -> dict[str, Any]:
    """Switch to a different domain/model."""
    domain = get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")
    
    if not is_model_trained(domain_id):
        raise HTTPException(
            status_code=400, 
            detail=f"Model not trained for domain '{domain_id}'. Run training first."
        )
    
    try:
        domain_model_service.load_domain(domain_id)
        _audit("domain_switch", {"domain_id": domain_id})
        return {
            "status": "activated",
            "domain": domain_model_service.get_domain_info(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decision/{domain_id}")
def domain_decision(domain_id: str, row: dict[str, Any]) -> dict[str, Any]:
    """PRISM decision engine for a specific domain. Domain-aware predictions and explanations."""
    # Load domain if needed
    domain = get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")
    
    if not is_model_trained(domain_id):
        raise HTTPException(
            status_code=400,
            detail=f"Model not trained for domain '{domain_id}'."
        )
    
    try:
        domain_model_service.load_domain(domain_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load domain: {e}")
    
    # Build DataFrame from row
    df = pd.DataFrame([row])
    
    # Validate and transform
    X, df_clean, errs = domain_model_service.validate_and_transform(df)
    
    if X.size == 0:
        raise HTTPException(status_code=400, detail="; ".join(errs) or "Invalid input")
    
    # Predict
    dec, conf, probs = domain_model_service.predict_single(X[0])
    
    out: dict[str, Any] = {
        "decision": {
            "decision": dec,
            "confidence": conf,
            "probabilities": probs,
            "domain_id": domain_id,
            "positive_label": domain.positive_label,
            "negative_label": domain.negative_label,
        },
        "domain_info": domain_model_service.get_domain_info(),
    }
    
    # SHAP explanations and explanation fidelity
    explanation_fidelity: dict[str, Any] = {}
    try:
        sh = domain_explainability_service.shap_values(X[0], domain_id)
        out["shap"] = {
            "decision_factor_names": sh.get("feature_names", []),
            "values": sh.get("values", []),
            "base_value": sh.get("base_value", 0.0),
            "decision": dec,
            "feature_labels": sh.get("feature_labels", {}),
        }
        shap_pred = sh.get("prediction")
        fidelity_match = shap_pred == dec if shap_pred is not None else True
        explanation_fidelity = {
            "prediction_match": fidelity_match,
            "shap_base_value": sh.get("base_value", 0.0),
            "model_decision": dec,
        }
    except Exception as e:
        logger.warning("SHAP failed for domain %s: %s", domain_id, e)
        out["shap"] = {
            "decision_factor_names": [],
            "values": [],
            "base_value": 0.0,
            "decision": dec,
            "feature_labels": {},
        }
        explanation_fidelity = {"prediction_match": False, "error": str(e)}
    out["explanation_fidelity"] = explanation_fidelity
    
    # Domain-aware explanation layer
    row_dict = df_clean.iloc[0].to_dict() if not df_clean.empty else row
    try:
        expl = _domain_plain_language_explanations(out["shap"], dec, row_dict, domain, top_k=5)
        out["explanation_layer"] = expl
    except Exception as e:
        logger.warning("Explanation layer failed: %s", e)
        out["explanation_layer"] = {"bullets": [], "summary": [], "directional_reasoning": ""}
    
    # Domain-aware uncertainty and stability testing
    try:
        unc = _domain_uncertainty_stability(row_dict, dec, conf, domain, domain_model_service)
        out["uncertainty"] = unc
    except Exception as e:
        logger.warning("Uncertainty/stability failed for domain %s: %s", domain_id, e)
        out["uncertainty"] = {
            "confidence_band": "high" if conf >= 0.85 else ("medium" if conf >= 0.6 else "low"),
            "stable": conf >= 0.6,
            "warning": None,
            "volatility_note": None,
            "unstable_features": [],
        }
    
    # Trust calibration
    try:
        shap_vals = out["shap"].get("values", [])
        trust_cal = _compute_trust_calibration(conf, shap_vals)
        out["trust_calibration"] = trust_cal
    except Exception as e:
        out["trust_calibration"] = {
            "model_confidence": conf,
            "historical_accuracy": None,
            "calibration_warning": None,
            "complexity_score": 0.5,
            "estimated_read_time_seconds": 30,
        }
    
    # Domain-aware counterfactual preview
    try:
        cf_preview = _domain_counterfactual_preview(row_dict, out["shap"], dec, domain)
        out["counterfactuals"] = []  # Keep for compatibility
        out["counterfactual_preview"] = cf_preview
    except Exception as e:
        logger.warning("Counterfactual preview failed for domain %s: %s", domain_id, e)
        out["counterfactuals"] = []
        out["counterfactual_preview"] = []
    
    _audit("domain_decision", {"domain_id": domain_id, "decision": dec})
    return out


@router.post("/decision/{domain_id}/batch")
def domain_decision_batch(domain_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch prediction for multiple rows. Returns only decisions (no full explanations for performance).
    
    Use this to preview outcomes for all rows in a dataset.
    """
    domain = get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")
    
    if not is_model_trained(domain_id):
        raise HTTPException(status_code=400, detail=f"Model not trained for domain '{domain_id}'.")
    
    try:
        domain_model_service.load_domain(domain_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load domain: {e}")
    
    predictions = []
    
    for i, row in enumerate(rows):
        try:
            df = pd.DataFrame([row])
            X, df_clean, errs = domain_model_service.validate_and_transform(df)
            
            if X.size == 0:
                predictions.append({
                    "index": i,
                    "decision": None,
                    "confidence": None,
                    "error": "; ".join(errs) or "Invalid input"
                })
                continue
            
            dec, conf, probs = domain_model_service.predict_single(X[0])
            
            predictions.append({
                "index": i,
                "decision": dec,
                "confidence": conf,
                "positive_label": domain.positive_label,
                "negative_label": domain.negative_label,
            })
        except Exception as e:
            predictions.append({
                "index": i,
                "decision": None,
                "confidence": None,
                "error": str(e)
            })
    
    return {
        "domain_id": domain_id,
        "predictions": predictions,
        "count": len(predictions),
        "positive_label": domain.positive_label,
        "negative_label": domain.negative_label,
    }


def _decode_shap_feature_name(encoded_name: str) -> str:
    """Decode SHAP encoded feature name to original feature name.
    
    Examples:
        'num__A2' -> 'A2'
        'cat__A5_0' -> 'A5'
        'cat__checking_status_A11' -> 'checking_status'
        'num__credit_amount' -> 'credit_amount'
    """
    import re
    
    # Handle num__ prefix
    if encoded_name.startswith("num__"):
        return encoded_name[5:]  # Remove 'num__'
    
    # Handle cat__ prefix with encoded categories
    if encoded_name.startswith("cat__"):
        rest = encoded_name[5:]  # Remove 'cat__'
        # Try to match pattern like 'A5_0' or 'checking_status_A11'
        # Find the last underscore that precedes a category value
        parts = rest.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0]
        return rest
    
    # No prefix - might be direct feature name
    return encoded_name


def _domain_plain_language_explanations(
    shap_data: dict[str, Any],
    decision: str,
    row: dict[str, Any],
    domain,
    top_k: int = 5
) -> dict[str, Any]:
    """Generate plain language explanations with domain-aware labels."""
    feature_names = shap_data.get("decision_factor_names", [])
    values = shap_data.get("values", [])
    feature_labels = domain.feature_labels if domain else {}
    
    if not feature_names or not values:
        return {"bullets": [], "summary": [], "directional_reasoning": ""}
    
    # Separate positive and negative factors
    positive_factors = []
    negative_factors = []
    
    seen_features = set()
    
    for encoded_name, shap_val in zip(feature_names, values):
        # Decode the SHAP feature name to original
        orig_feature = _decode_shap_feature_name(encoded_name)
        
        # Skip duplicates (same feature with different encodings)
        if orig_feature in seen_features:
            continue
        seen_features.add(orig_feature)
        
        # Get human-readable label
        label = feature_labels.get(orig_feature, orig_feature.replace("_", " ").title())
        
        # Get row value
        row_val = row.get(orig_feature, "")
        
        factor_info = {
            "feature": orig_feature,
            "label": label,
            "shap_value": float(shap_val),
            "row_value": row_val,
        }
        
        if shap_val > 0:
            positive_factors.append(factor_info)
        else:
            negative_factors.append(factor_info)
    
    bullets = []
    summary = []
    
    # Determine if decision is positive outcome
    is_positive_decision = decision == domain.positive_label
    
    if is_positive_decision:
        # For positive decisions: show what helped first
        for f in positive_factors[:3]:
            impact = "strongly" if abs(f["shap_value"]) > 0.3 else "moderately" if abs(f["shap_value"]) > 0.1 else ""
            if f["row_value"] not in (None, "", "nan"):
                bullet = f"Your {f['label']} ({f['row_value']}) {impact} supported the {domain.positive_label.lower()} decision.".replace("  ", " ")
            else:
                bullet = f"Your {f['label']} {impact} supported the {domain.positive_label.lower()} decision.".replace("  ", " ")
            bullets.append(bullet)
            summary.append({"decision_factor": f["feature"], "label": f["label"], "direction": "positive", "value": f["shap_value"]})
        
        # Then show minor negative factors
        for f in negative_factors[:2]:
            if f["row_value"] not in (None, "", "nan"):
                bullet = f"Your {f['label']} ({f['row_value']}) slightly reduced confidence."
            else:
                bullet = f"Your {f['label']} slightly reduced confidence."
            bullets.append(bullet)
            summary.append({"decision_factor": f["feature"], "label": f["label"], "direction": "negative", "value": f["shap_value"]})
        
        # Directional reasoning for positive decision
        if positive_factors:
            top_factor = positive_factors[0]["label"]
            directional = f"The decision engine predicts {domain.positive_label}. Key factors like {top_factor} contributed positively to this outcome."
        else:
            directional = f"The decision engine predicts {domain.positive_label} based on the overall profile assessment."
    else:
        # For negative decisions: show what hurt first
        for f in negative_factors[:3]:
            impact = "significantly" if abs(f["shap_value"]) > 0.5 else "moderately" if abs(f["shap_value"]) > 0.2 else ""
            if f["row_value"] not in (None, "", "nan"):
                bullet = f"Your {f['label']} ({f['row_value']}) {impact} contributed to the {domain.negative_label.lower()} prediction.".replace("  ", " ")
            else:
                bullet = f"Your {f['label']} {impact} contributed to the {domain.negative_label.lower()} prediction.".replace("  ", " ")
            bullets.append(bullet)
            summary.append({"decision_factor": f["feature"], "label": f["label"], "direction": "negative", "value": f["shap_value"]})
        
        # Then show factors that were favorable
        for f in positive_factors[:2]:
            if f["row_value"] not in (None, "", "nan"):
                bullet = f"Your {f['label']} ({f['row_value']}) was favorable but not enough to change the outcome."
            else:
                bullet = f"Your {f['label']} was favorable but not enough to change the outcome."
            bullets.append(bullet)
            summary.append({"decision_factor": f["feature"], "label": f["label"], "direction": "positive", "value": f["shap_value"]})
        
        # Directional reasoning for negative decision
        if negative_factors:
            top_factor = negative_factors[0]["label"]
            directional = f"The decision engine predicts {domain.negative_label}. Factors like {top_factor} weighed against a positive outcome."
        else:
            directional = f"The decision engine predicts {domain.negative_label} based on the overall risk assessment."
    
    return {
        "bullets": bullets[:top_k],
        "summary": summary[:top_k],
        "directional_reasoning": directional,
    }


def _domain_uncertainty_stability(
    row_dict: dict[str, Any],
    decision: str,
    confidence: float,
    domain,
    domain_model_service,
    delta_pct: float = 0.05,
) -> dict[str, Any]:
    """Domain-aware uncertainty and stability testing.
    
    Perturbs numeric features slightly and checks if the decision flips.
    Uses domain-specific feature columns and labels.
    """
    import numpy as np
    
    # Basic confidence band
    conf_band = "high" if confidence >= 0.85 else ("medium" if confidence >= 0.6 else "low")
    is_low_conf = confidence < 0.6
    is_positive = decision == domain.positive_label
    
    # Stability testing: perturb numeric features
    stable = True
    unstable_features = []
    
    try:
        df_base = pd.DataFrame([row_dict])
        
        # Replace missing values marker
        df_base = df_base.replace("?", np.nan)
        
        # Convert numeric columns
        for col in domain.numeric_cols:
            if col in df_base.columns:
                df_base[col] = pd.to_numeric(df_base[col], errors="coerce")
        
        # Test perturbations on numeric features
        for col in domain.numeric_cols[:10]:  # Limit to first 10 numeric cols for performance
            if col not in df_base.columns:
                continue
            val = df_base[col].iloc[0]
            if pd.isna(val):
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            
            # Try small perturbations in both directions
            for sign in (-1, 1):
                df_perturbed = df_base.copy()
                delta = max(1, abs(v) * delta_pct) * sign
                df_perturbed[col] = v + delta
                
                try:
                    X_perturbed, _, _ = domain_model_service.validate_and_transform(df_perturbed)
                    if X_perturbed.size == 0:
                        continue
                    p_dec, _, _ = domain_model_service.predict_single(X_perturbed[0])
                    if p_dec != decision:
                        stable = False
                        feature_label = domain.feature_labels.get(col, col.replace("_", " ").title())
                        if feature_label not in unstable_features:
                            unstable_features.append(feature_label)
                        break
                except Exception:
                    continue
    except Exception as e:
        logger.warning("Stability testing failed: %s", e)
    
    # Generate contextual warnings
    warning = None
    volatility_note = None
    
    if is_low_conf and not stable:
        if is_positive:
            warning = f"This {domain.positive_label.lower()} prediction has low confidence ({confidence*100:.0f}%) and is sensitive to small changes. The decision could easily flip."
        else:
            warning = f"This {domain.negative_label.lower()} prediction has low confidence ({confidence*100:.0f}%) and is borderline. Small changes could lead to a different outcome."
    elif is_low_conf:
        if is_positive:
            warning = f"This {domain.positive_label.lower()} prediction has low confidence ({confidence*100:.0f}%). Consider it a tentative assessment."
        else:
            warning = f"This {domain.negative_label.lower()} prediction has low confidence ({confidence*100:.0f}%). The case is borderline."
    elif not stable:
        if unstable_features:
            features_str = ", ".join(unstable_features[:2])
            if is_positive:
                warning = f"This {domain.positive_label.lower()} prediction is sensitive to changes in {features_str}. Small negative changes could flip the decision."
            else:
                warning = f"This {domain.negative_label.lower()} prediction is sensitive to changes in {features_str}. Small improvements could lead to a different outcome."
        else:
            warning = "This decision is unstable under small input changes."
    
    if not stable:
        if is_positive:
            volatility_note = f"The decision engine indicates this {domain.positive_label.lower()} prediction is near the decision boundary. Monitor for changes in key factors."
        else:
            volatility_note = f"The decision engine indicates this case is near the decision boundary. Use the What-If tool to explore potential improvements."
    
    return {
        "confidence_band": conf_band,
        "stable": stable,
        "warning": warning,
        "volatility_note": volatility_note,
        "unstable_features": unstable_features,
    }


def _domain_counterfactual_preview(
    row_dict: dict[str, Any],
    shap_data: dict[str, Any],
    decision: str,
    domain,
) -> list[dict[str, Any]]:
    """Domain-aware counterfactual suggestions based on SHAP values.
    
    For negative outcomes: suggests improvements to key factors.
    For positive outcomes: shows what could threaten the outcome.
    """
    out: list[dict[str, Any]] = []
    
    feature_names = shap_data.get("decision_factor_names", [])
    values = shap_data.get("values", [])
    feature_labels = domain.feature_labels if domain else {}
    
    if not feature_names or not values:
        return out
    
    is_positive = decision == domain.positive_label
    seen_features = set()
    
    # Pair features with their SHAP values
    feature_shap_pairs = list(zip(feature_names, values))
    
    if not is_positive:
        # For negative outcomes: find factors that hurt (negative SHAP) and suggest improvements
        # Sort by most negative SHAP values first
        negative_factors = [(f, v) for f, v in feature_shap_pairs if v < 0]
        negative_factors.sort(key=lambda x: x[1])
        
        for encoded_name, shap_val in negative_factors[:5]:
            orig_feature = _decode_shap_feature_name(encoded_name)
            
            if orig_feature in seen_features:
                continue
            seen_features.add(orig_feature)
            
            label = feature_labels.get(orig_feature, orig_feature.replace("_", " ").title())
            raw_value = row_dict.get(orig_feature, "")
            
            # Check if it's a numeric feature we can suggest improvement for
            if orig_feature in domain.numeric_cols:
                if raw_value not in (None, "", "nan"):
                    try:
                        current_val = float(raw_value)
                        # Suggest improvement direction
                        out.append({
                            "suggestion": f"Improving your {label} (currently {current_val:.1f}) could positively impact the outcome.",
                            "decision_factor": orig_feature,
                            "label": label,
                            "current_value": current_val,
                            "change_direction": "improve",
                            "impact": abs(shap_val),
                        })
                    except (TypeError, ValueError):
                        out.append({
                            "suggestion": f"Changes to your {label} could affect the outcome.",
                            "decision_factor": orig_feature,
                            "label": label,
                            "change_direction": "unknown",
                            "impact": abs(shap_val),
                        })
                else:
                    out.append({
                        "suggestion": f"Improving your {label} could positively impact the outcome.",
                        "decision_factor": orig_feature,
                        "label": label,
                        "change_direction": "improve",
                        "impact": abs(shap_val),
                    })
            else:
                # Categorical feature
                out.append({
                    "suggestion": f"A different {label} status could affect the outcome.",
                    "decision_factor": orig_feature,
                    "label": label,
                    "change_direction": "change",
                    "impact": abs(shap_val),
                })
            
            if len(out) >= 3:
                break
    else:
        # For positive outcomes: show what factors are supporting it (could threaten if changed)
        positive_factors = [(f, v) for f, v in feature_shap_pairs if v > 0]
        positive_factors.sort(key=lambda x: -x[1])  # Most positive first
        
        for encoded_name, shap_val in positive_factors[:3]:
            orig_feature = _decode_shap_feature_name(encoded_name)
            
            if orig_feature in seen_features:
                continue
            seen_features.add(orig_feature)
            
            label = feature_labels.get(orig_feature, orig_feature.replace("_", " ").title())
            raw_value = row_dict.get(orig_feature, "")
            
            if raw_value not in (None, "", "nan"):
                out.append({
                    "suggestion": f"Your {label} ({raw_value}) is favorable — significant changes here could affect the outcome.",
                    "decision_factor": orig_feature,
                    "label": label,
                    "current_value": raw_value,
                    "change_direction": "maintain",
                    "impact": shap_val,
                })
            else:
                out.append({
                    "suggestion": f"Your {label} is favorable — changes here could affect the outcome.",
                    "decision_factor": orig_feature,
                    "label": label,
                    "change_direction": "maintain",
                    "impact": shap_val,
                })
    
    # If no suggestions generated, provide general guidance
    if not out:
        if not is_positive:
            out.append({
                "suggestion": f"Consider improving key factors to increase chances of a {domain.positive_label.lower()} outcome.",
                "decision_factor": "general",
                "label": "General",
                "change_direction": "improve",
                "impact": 0,
            })
        else:
            out.append({
                "suggestion": f"Current profile supports the {domain.positive_label.lower()} prediction. Maintain key factors.",
                "decision_factor": "general",
                "label": "General",
                "change_direction": "maintain",
                "impact": 0,
            })
    
    return out[:5]


@router.post("/decision")
def decision(row: dict[str, Any]) -> dict[str, Any]:
    """PRISM decision engine: decision, confidence, SHAP, counterfactuals for one row (A1–A15)."""
    try:
        csv_bytes = _dict_to_csv(row).encode("utf-8")
        df, errs = data_service.validate_csv(csv_bytes, max_records=1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid row: {e}")
    if errs and any("Missing columns" in str(e) for e in errs):
        raise HTTPException(status_code=400, detail="; ".join(errs))
    if df.empty:
        raise HTTPException(status_code=400, detail="Could not parse row")
    X = data_service.transform(df)
    dec, conf, probs = model_service.predict_single(X[0])
    out: dict[str, Any] = {
        "decision": DecisionResponse(
            decision=dec, confidence=conf, probabilities=probs
        ).model_dump(),
    }
    explanation_fidelity: dict[str, Any] = {}
    try:
        sh = explainability_service.shap_values(X[0])
        out["shap"] = ShapValuesResponse(
            decision_factor_names=sh.get("feature_names") or [],
            values=sh.get("values") or [],
            base_value=sh.get("base_value") or 0.0,
            decision=sh.get("prediction") or dec,
        ).model_dump()
        # Explanation fidelity: does SHAP prediction match model prediction? Stability = consistent under small input change
        shap_pred = sh.get("prediction")
        shap_base = sh.get("base_value", 0.0)
        shap_vals = sh.get("values") or []
        shap_sum = sum(shap_vals) if shap_vals else 0.0
        # For binary: base + sum(SHAP) ≈ logit or prob; prediction match is primary
        fidelity_match = shap_pred == dec if shap_pred else True
        explanation_fidelity = {
            "prediction_match": fidelity_match,
            "shap_base_value": shap_base,
            "shap_sum": shap_sum,
            "model_decision": dec,
        }
    except Exception as e:
        logger.warning("SHAP failed: %s", e)
        out["shap"] = ShapValuesResponse(
            decision_factor_names=data_service.feature_names,
            values=[],
            base_value=0.0,
            decision=dec,
        ).model_dump()
        explanation_fidelity = {"prediction_match": False, "error": str(e)}
    try:
        cfs_raw = explainability_service.counterfactuals(df, total_cfs=2)
        out["counterfactuals"] = [
            CounterfactualResponse(
                original=c["original"],
                modified=c["modified"],
                original_decision=c["original_prediction"],
                new_decision=c["new_prediction"],
                changed_decision_factors=c["changed_features"],
            ).model_dump()
            for c in cfs_raw
        ]
    except Exception as e:
        logger.warning("Counterfactuals failed: %s", e)
        out["counterfactuals"] = []

    row_dict = df.fillna("").iloc[0].to_dict()
    try:
        expl = plain_language_explanations(out["shap"], dec, row_dict, top_k=5)
        out["explanation_layer"] = ExplanationLayerResponse(**expl).model_dump()
    except Exception as e:
        logger.warning("Explanation layer failed: %s", e)
        out["explanation_layer"] = ExplanationLayerResponse(
            bullets=[], summary=[], directional_reasoning=""
        ).model_dump()

    try:
        unc = uncertainty_stability(row_dict, dec, conf, delta_pct=0.05)
        out["uncertainty"] = UncertaintyResponse(**unc).model_dump()
    except Exception as e:
        logger.warning("Uncertainty/stability failed: %s", e)
        out["uncertainty"] = UncertaintyResponse(
            confidence_band="medium", stable=True, warning=None, volatility_note=None
        ).model_dump()

    try:
        preview = counterfactual_preview(row_dict, out["counterfactuals"], out["shap"], dec)
        out["counterfactual_preview"] = [CounterfactualPreviewItem(**p).model_dump() for p in preview]
    except Exception as e:
        logger.warning("Counterfactual preview failed: %s", e)
        out["counterfactual_preview"] = []

    # Trust calibration for cognitive load and accuracy awareness
    try:
        shap_vals = out["shap"].get("values", [])
        trust_cal = _compute_trust_calibration(conf, shap_vals)
        out["trust_calibration"] = TrustCalibration(**trust_cal).model_dump()
    except Exception as e:
        logger.warning("Trust calibration failed: %s", e)
        out["trust_calibration"] = TrustCalibration(
            model_confidence=conf,
            historical_accuracy=None,
            calibration_warning=None,
            complexity_score=0.5,
            estimated_read_time_seconds=30,
        ).model_dump()

    out["explanation_fidelity"] = explanation_fidelity

    _audit("decision", {"decision": dec})
    return out


@router.post("/predict")
def predict(row: dict[str, Any]) -> dict[str, Any]:
    """Alias for POST /decision. Deprecated: use /decision."""
    return decision(row)


def _dict_to_csv(row: dict[str, Any]) -> str:
    import csv
    buf = io.StringIO()
    cols = [f"A{i}" for i in range(1, 16)]
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerow({c: row.get(c, "") for c in cols})
    return buf.getvalue()


class BulkExportRequest(BaseModel):
    """Bulk export: dataset rows + predictions (+ optional explanation bullets)."""

    format: str = Field("csv", pattern="^(csv)$")
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    predictions: dict[str, Any] = Field(default_factory=dict)  # index -> { decision, confidence, positive_label?, negative_label? }
    include_explanations: bool = False  # If True, predictions may include bullets per row (from client)


@router.post("/export")
def export(req: ExportRequest) -> StreamingResponse:
    """Export last results as CSV or PDF. Body: { format, data }."""
    data = req.data or {}
    fmt = req.format
    if fmt == "csv":
        buf = _export_csv(data)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=prism_export.csv"},
        )
    if fmt == "pdf":
        buf = _export_pdf(data)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=prism_export.pdf"},
        )
    raise HTTPException(status_code=400, detail="format must be csv or pdf")


@router.post("/export/bulk")
def export_bulk(req: BulkExportRequest) -> StreamingResponse:
    """Export dataset rows with batch predictions as CSV (full audit trail)."""
    import csv as csv_module
    buf = io.StringIO()
    cols = list(req.columns) if req.columns else []
    # Add decision columns
    out_cols = ["row_index", "decision", "confidence"] + cols
    w = csv_module.DictWriter(buf, fieldnames=out_cols, extrasaction="ignore")
    w.writeheader()
    for i, row in enumerate(req.rows):
        pred = req.predictions.get(str(i)) or req.predictions.get(i)
        dec = (pred.get("decision") or "") if pred else ""
        conf = pred.get("confidence")
        conf_str = f"{conf * 100:.1f}%" if isinstance(conf, (int, float)) else str(conf or "")
        out_row = {"row_index": i + 1, "decision": dec, "confidence": conf_str}
        for c in cols:
            if c in row:
                out_row[c] = row[c]
        w.writerow(out_row)
    out_buf = io.BytesIO(buf.getvalue().encode("utf-8"))
    out_buf.seek(0)
    return StreamingResponse(
        iter([out_buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prism_bulk_export.csv"},
    )


def _export_csv(data: dict[str, Any]) -> io.BytesIO:
    """Layer 2: Technical export with full data for auditability."""
    import csv
    from datetime import datetime
    
    buf = io.StringIO()
    w = csv.writer(buf)
    
    # Header
    w.writerow(["PRISM Technical Export"])
    w.writerow(["Generated", datetime.utcnow().isoformat()])
    w.writerow([])
    
    # Decision summary
    decision_data = data.get("decision", {})
    w.writerow(["=== DECISION ==="])
    w.writerow(["Outcome", decision_data.get("decision", "")])
    w.writerow(["Confidence", f"{decision_data.get('confidence', 0) * 100:.1f}%"])
    probs = decision_data.get("probabilities", {})
    w.writerow(["P(Approved)", f"{probs.get('+', 0) * 100:.1f}%"])
    w.writerow(["P(Rejected)", f"{probs.get('-', 0) * 100:.1f}%"])
    w.writerow([])
    
    # Uncertainty
    uncertainty = data.get("uncertainty", {})
    w.writerow(["=== UNCERTAINTY & STABILITY ==="])
    w.writerow(["Confidence Band", uncertainty.get("confidence_band", "")])
    w.writerow(["Stable", "Yes" if uncertainty.get("stable") else "No"])
    if uncertainty.get("warning"):
        w.writerow(["Warning", uncertainty.get("warning")])
    w.writerow([])
    
    # SHAP values
    shap_data = data.get("shap", {})
    w.writerow(["=== SHAP VALUES (Feature Attribution) ==="])
    w.writerow(["Base Value", shap_data.get("base_value", 0)])
    w.writerow(["Feature", "SHAP Value"])
    names = shap_data.get("decision_factor_names", [])
    values = shap_data.get("values", [])
    for n, v in zip(names, values):
        w.writerow([n, f"{v:.6f}"])
    w.writerow([])
    
    # Counterfactuals
    cfs = data.get("counterfactuals", [])
    if cfs:
        w.writerow(["=== COUNTERFACTUALS ==="])
        for i, cf in enumerate(cfs):
            w.writerow([f"Counterfactual {i+1}"])
            w.writerow(["Changed Factors", ", ".join(cf.get("changed_decision_factors", []))])
            w.writerow(["Original Decision", cf.get("original_decision", "")])
            w.writerow(["New Decision", cf.get("new_decision", "")])
        w.writerow([])
    
    # Raw data dump for full auditability
    w.writerow(["=== RAW JSON (for audit) ==="])
    w.writerow([json.dumps(data, default=str)])
    
    out = io.BytesIO(buf.getvalue().encode("utf-8"))
    out.seek(0)
    return out


def _export_pdf(data: dict[str, Any]) -> io.BytesIO:
    """Two-layer PDF: Human-readable summary (Layer 1) + Technical appendix (Layer 2)."""
    from datetime import datetime
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak, HRFlowable
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    
    # Custom styles (prefixed to avoid conflicts with built-in styles)
    styles.add(ParagraphStyle(
        'PRISMTitle', parent=styles['Title'], fontSize=28, textColor=HexColor('#1a1a2e'),
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        'PRISMSubtitle', parent=styles['Normal'], fontSize=11, textColor=HexColor('#666666'),
        spaceAfter=20
    ))
    styles.add(ParagraphStyle(
        'PRISMSection', parent=styles['Heading2'], fontSize=14, textColor=HexColor('#1a1a2e'),
        spaceBefore=16, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        'PRISMBody', parent=styles['Normal'], fontSize=11, leading=16, textColor=HexColor('#333333')
    ))
    styles.add(ParagraphStyle(
        'PRISMDisclaimer', parent=styles['Normal'], fontSize=9, textColor=HexColor('#888888'),
        spaceBefore=20, leading=12
    ))
    styles.add(ParagraphStyle(
        'PRISMAppendix', parent=styles['Heading2'], fontSize=12, textColor=HexColor('#666666'),
        spaceBefore=12, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        'PRISMMono', parent=styles['Normal'], fontSize=8, fontName='Courier', textColor=HexColor('#555555')
    ))

    story = []
    
    # Extract data
    decision_data = data.get("decision", {})
    uncertainty = data.get("uncertainty", {})
    explanation = data.get("explanation_layer", {})
    cf_preview = data.get("counterfactual_preview", [])
    shap_data = data.get("shap", {})
    
    outcome = decision_data.get("decision", "")
    confidence = decision_data.get("confidence", 0)
    conf_band = uncertainty.get("confidence_band", "medium")
    is_stable = uncertainty.get("stable", True)
    
    # ========== LAYER 1: HUMAN-READABLE SUMMARY ==========
    
    story.append(Paragraph("PRISM", styles['PRISMTitle']))
    story.append(Paragraph("Decision Summary Export", styles['PRISMSubtitle']))
    story.append(Spacer(1, 12))
    
    # Decision Outcome Box
    story.append(Paragraph("Decision Outcome", styles['PRISMSection']))
    outcome_text = "Approved" if outcome == "+" else "Rejected"
    conf_label = "High" if conf_band == "high" else ("Medium" if conf_band == "medium" else "Low")
    story.append(Paragraph(
        f"<b>{outcome_text}</b>",
        ParagraphStyle('OutcomeText', parent=styles['Normal'], fontSize=20, 
                      textColor=HexColor('#22c55e') if outcome == "+" else HexColor('#ef4444'))
    ))
    story.append(Paragraph(f"Confidence: {conf_label} ({confidence*100:.0f}%)", styles['PRISMBody']))
    story.append(Spacer(1, 16))
    
    # Stability
    story.append(Paragraph("Stability", styles['PRISMSection']))
    if is_stable:
        story.append(Paragraph("This decision is <b>stable</b> under small changes.", styles['PRISMBody']))
        story.append(Paragraph("Low risk of flip due to minor input variation.", styles['PRISMBody']))
    else:
        story.append(Paragraph("This decision is <b>sensitive</b> to small changes.", styles['PRISMBody']))
        if uncertainty.get("warning"):
            story.append(Paragraph(uncertainty["warning"], styles['PRISMBody']))
    story.append(Spacer(1, 16))
    
    # Top Reasons
    story.append(Paragraph("Top Reasons", styles['PRISMSection']))
    bullets = explanation.get("bullets", [])
    if bullets:
        for bullet in bullets[:5]:
            story.append(Paragraph(f"• {bullet}", styles['PRISMBody']))
    else:
        story.append(Paragraph("• Detailed reasoning not available for this decision.", styles['PRISMBody']))
    story.append(Spacer(1, 16))
    
    # What Could Change the Outcome
    story.append(Paragraph("What Could Change the Outcome", styles['PRISMSection']))
    if cf_preview:
        for item in cf_preview[:3]:
            suggestion = item.get("suggestion", "")
            if suggestion:
                story.append(Paragraph(f"• {suggestion}", styles['PRISMBody']))
    else:
        story.append(Paragraph("• No specific actionable changes identified.", styles['PRISMBody']))
    story.append(Spacer(1, 24))
    
    # Important Note (Disclaimer)
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#dddddd')))
    story.append(Paragraph(
        "<b>Important Note:</b> This explanation reflects model behaviour based on historical patterns. "
        "It does not guarantee real-world outcomes. Final decisions should incorporate additional context, "
        "regulatory requirements, and human judgment.",
        styles['PRISMDisclaimer']
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Generated by PRISM on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        styles['PRISMDisclaimer']
    ))
    
    # ========== LAYER 2: TECHNICAL APPENDIX ==========
    
    story.append(PageBreak())
    story.append(Paragraph("Technical Appendix", styles['PRISMTitle']))
    story.append(Paragraph("For transparency, ethics, and auditability", styles['PRISMSubtitle']))
    
    # Probabilities
    story.append(Paragraph("Probabilities", styles['PRISMAppendix']))
    probs = decision_data.get("probabilities", {})
    prob_data = [
        ["Class", "Probability"],
        ["Approved (+)", f"{probs.get('+', 0)*100:.2f}%"],
        ["Rejected (−)", f"{probs.get('-', 0)*100:.2f}%"],
    ]
    prob_table = Table(prob_data, colWidths=[2*inch, 1.5*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f3f4f6')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dddddd')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 12))
    
    # SHAP Values
    story.append(Paragraph("SHAP Values (Feature Attribution)", styles['PRISMAppendix']))
    story.append(Paragraph(f"Base value: {shap_data.get('base_value', 0):.4f}", styles['PRISMMono']))
    shap_names = shap_data.get("decision_factor_names", [])
    shap_vals = shap_data.get("values", [])
    if shap_names and shap_vals:
        shap_table_data = [["Decision Factor", "SHAP Value", "Direction"]]
        for n, v in zip(shap_names[:15], shap_vals[:15]):
            direction = "→ Approval" if v > 0 else "→ Rejection"
            shap_table_data.append([n[:40], f"{v:.4f}", direction])
        shap_table = Table(shap_table_data, colWidths=[2.5*inch, 1*inch, 1.2*inch])
        shap_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f3f4f6')),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 1), (1, -1), 'Courier'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dddddd')),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(shap_table)
    story.append(Spacer(1, 12))
    
    # Uncertainty Metadata
    story.append(Paragraph("Uncertainty Metadata", styles['PRISMAppendix']))
    unc_data = [
        ["Metric", "Value"],
        ["Confidence Band", conf_band.capitalize()],
        ["Stable Under Perturbation", "Yes" if is_stable else "No"],
        ["Raw Confidence", f"{confidence:.4f}"],
    ]
    if uncertainty.get("volatility_note"):
        unc_data.append(["Volatility Note", uncertainty["volatility_note"][:60] + "..."])
    unc_table = Table(unc_data, colWidths=[2*inch, 3*inch])
    unc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f3f4f6')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dddddd')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(unc_table)
    story.append(Spacer(1, 12))
    
    # Counterfactuals (if any)
    cfs = data.get("counterfactuals", [])
    if cfs:
        story.append(Paragraph("Counterfactual Scenarios", styles['PRISMAppendix']))
        for i, cf in enumerate(cfs[:3]):
            changed = cf.get("changed_decision_factors", [])
            orig_dec = cf.get("original_decision", "")
            new_dec = cf.get("new_decision", "")
            story.append(Paragraph(
                f"<b>Scenario {i+1}:</b> Change {', '.join(changed[:3])} → Decision flips from "
                f"{'Approved' if orig_dec == '+' else 'Rejected'} to {'Approved' if new_dec == '+' else 'Rejected'}",
                styles['PRISMMono']
            ))
        story.append(Spacer(1, 12))
    
    # Footer
    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#dddddd')))
    story.append(Paragraph(
        "PRISM — Human-centred Explainable AI for Credit Decisions. "
        "This technical appendix is provided for transparency and audit purposes.",
        styles['PRISMDisclaimer']
    ))

    doc.build(story)
    buf.seek(0)
    return buf


@router.post("/feedback")
def feedback(fb: FeedbackCreate) -> dict[str, str]:
    """Store user feedback for evaluation."""
    _audit("feedback", fb.model_dump(exclude_none=True))
    # Log to study if session provided
    if fb.session_id:
        study_service.log_interaction(fb.session_id, "feedback_submit", fb.model_dump(exclude_none=True))
    return {"status": "recorded"}


# ============== STUDY MANAGEMENT ROUTES ==============
# SLR demo only: study/participant/questionnaire collection disabled (no human subjects).
# These endpoints return 410 Gone so no participant data is ever stored.
STUDY_DISABLED = True

def _study_disabled() -> None:
    if STUDY_DISABLED:
        raise HTTPException(
            status_code=410,
            detail="Study and participant data collection is disabled. This deployment is for SLR demo only (no user study, no questionnaires).",
        )


@router.post("/study/session")
def create_study_session(req: StudySessionCreate) -> StudySessionResponse:
    """Create a new study session for a participant."""
    _study_disabled()
    pre_q = req.pre_questionnaire.model_dump() if req.pre_questionnaire else None
    session = study_service.create_session(
        participant_id=req.participant_id,
        condition=req.condition,
        pre_questionnaire=pre_q,
        within_subjects=req.within_subjects,
    )
    _audit("study_session_start", {"session_id": session["session_id"], "condition": req.condition})
    return StudySessionResponse(**session)


@router.get("/study/session/{session_id}")
def get_study_session(session_id: str) -> StudySessionResponse:
    """Get study session details."""
    _study_disabled()
    session = study_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return StudySessionResponse(**session)


@router.post("/study/session/{session_id}/end")
def end_study_session(session_id: str) -> StudySessionResponse:
    """End a study session."""
    _study_disabled()
    session = study_service.end_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _audit("study_session_end", {"session_id": session_id})
    return StudySessionResponse(**session)


@router.post("/study/interaction")
def log_study_interaction(req: InteractionLog) -> dict[str, str]:
    """Log a user interaction for study analysis."""
    _study_disabled()
    success = study_service.log_interaction(
        session_id=req.session_id,
        action=req.action,
        details=req.details,
        timestamp=req.timestamp,
    )
    if not success:
        raise HTTPException(status_code=400, detail="Invalid or inactive session")
    return {"status": "logged"}


@router.get("/study/session/{session_id}/metrics")
def get_study_metrics(session_id: str) -> StudyMetrics:
    """Get computed metrics for a study session."""
    _study_disabled()
    metrics = study_service.get_session_metrics(session_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="Session not found")
    return StudyMetrics(**metrics)


@router.get("/study/sessions")
def list_study_sessions() -> list[StudySessionResponse]:
    """List all study sessions."""
    _study_disabled()
    sessions = study_service.get_all_sessions()
    return [StudySessionResponse(**s) for s in sessions]


@router.get("/study/export")
def export_study_data() -> dict[str, Any]:
    """Export all study data for analysis (sessions + interactions + summary)."""
    _study_disabled()
    data = study_service.export_study_data()
    _audit("study_export", {"sessions_count": len(data["sessions"])})
    return data


@router.get("/study/export/csv")
def export_study_csv() -> StreamingResponse:
    """Export study data as CSV for statistical analysis."""
    _study_disabled()
    csv_data = study_service.export_for_r()
    out = io.BytesIO(csv_data.encode("utf-8"))
    out.seek(0)
    return StreamingResponse(
        iter([out.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prism_study_sessions.csv"},
    )


@router.get("/study/export/interactions")
def export_interactions_csv() -> StreamingResponse:
    """Export interactions in long format for R/Python analysis."""
    _study_disabled()
    csv_data = study_service.export_interactions_long()
    out = io.BytesIO(csv_data.encode("utf-8"))
    out.seek(0)
    return StreamingResponse(
        iter([out.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prism_interactions.csv"},
    )


@router.get("/study/export/questionnaires")
def export_questionnaires() -> dict[str, str]:
    """Export pre/post questionnaires as CSV strings."""
    _study_disabled()
    return study_service.export_questionnaires()


# ============== TASK-BASED EVALUATION ROUTES ==============

@router.get("/study/session/{session_id}/task")
def get_current_task(session_id: str) -> dict[str, Any]:
    """Get the current task for a study session."""
    _study_disabled()
    task = study_service.get_current_task(session_id)
    if task is None:
        # Check if session exists
        session = study_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        # All tasks completed
        return {"completed": True, "message": "All tasks completed"}
    return task


@router.post("/study/session/{session_id}/task")
def submit_task_response(session_id: str, req: TaskResponse) -> dict[str, Any]:
    """Submit a response to a task."""
    _study_disabled()
    if req.session_id != session_id:
        raise HTTPException(status_code=400, detail="Session ID mismatch")
    
    result = study_service.submit_task_response(
        session_id=session_id,
        task_id=req.task_id,
        response=req.response,
        confidence=req.confidence,
        time_taken_seconds=req.time_taken_seconds,
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/study/session/{session_id}/post-questionnaire")
def submit_post_questionnaire(session_id: str, req: PostStudyQuestionnaire) -> dict[str, str]:
    """Submit post-study questionnaire (NASA-TLX, Trust, SUS)."""
    _study_disabled()
    result = study_service.submit_post_questionnaire(
        session_id=session_id,
        questionnaire=req.model_dump(),
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    _audit("post_questionnaire_submit", {"session_id": session_id})
    return result


def _compute_trust_calibration(confidence: float, shap_values: list[float]) -> dict[str, Any]:
    """Compute trust calibration data for a decision."""
    # Historical accuracy simulation based on confidence bands
    # In a real system, this would come from a calibration dataset
    if confidence >= 0.85:
        historical_accuracy = 0.88  # High confidence = ~88% historical accuracy
        calibration_warning = None
    elif confidence >= 0.6:
        historical_accuracy = 0.72  # Medium confidence = ~72% historical accuracy
        calibration_warning = "This decision has moderate confidence. The model has been accurate about 72% of the time at this confidence level."
    else:
        historical_accuracy = 0.58  # Low confidence = ~58% historical accuracy
        calibration_warning = "This decision has low confidence. Consider this assessment tentative — the model is accurate only about 58% of the time at this confidence level."
    
    # Complexity score based on number of significant SHAP values
    significant_features = sum(1 for v in shap_values if abs(v) > 0.1) if shap_values else 0
    complexity_score = min(1.0, significant_features / 10)
    
    # Estimated read time based on complexity
    base_time = 15  # seconds
    estimated_read_time = int(base_time + (complexity_score * 45))
    
    return {
        "model_confidence": confidence,
        "historical_accuracy": historical_accuracy,
        "calibration_warning": calibration_warning,
        "complexity_score": complexity_score,
        "estimated_read_time_seconds": estimated_read_time,
    }
