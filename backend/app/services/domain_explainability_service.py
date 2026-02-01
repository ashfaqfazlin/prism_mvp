"""Domain-aware SHAP explanations for PRISM multi-model support."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from app.config import settings
from app.services.domain_model_service import domain_model_service


class DomainExplainabilityService:
    """SHAP explanations with domain-aware model switching."""

    def __init__(self) -> None:
        # Cache explainers by domain_id
        self._explainers: dict[str, shap.TreeExplainer] = {}
        self._current_domain_id: str | None = None

    def _get_or_build_explainer(self, domain_id: str) -> shap.TreeExplainer | None:
        """Get cached explainer or build new one."""
        if domain_id in self._explainers:
            return self._explainers[domain_id]

        # Ensure domain is loaded
        domain_model_service.ensure_loaded(domain_id)
        
        domain = domain_model_service.current_domain
        if domain is None:
            return None

        # Load dataset for background
        datasets_dir = settings.base_dir / "datasets"
        filename_map = {
            "uci_credit_approval": "uci_credit_approval.csv",
            "german_credit": "german_credit.csv",
            "taiwan_credit_card": "taiwan_credit_card.csv",
        }
        
        filename = filename_map.get(domain_id)
        if not filename:
            return None
            
        dataset_path = datasets_dir / filename
        if not dataset_path.exists():
            return None

        try:
            df = pd.read_csv(dataset_path)
            
            # Get available feature columns
            feature_cols = [c for c in domain.feature_cols if c in df.columns]
            df = df[feature_cols].replace("?", np.nan)
            
            # Convert numeric columns
            for c in domain.numeric_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            
            # Transform background sample
            X_bg = domain_model_service._preproc.transform(df.head(100))
            
            # Build explainer
            explainer = shap.TreeExplainer(
                domain_model_service._model,
                X_bg,
                feature_names=domain_model_service.feature_names
            )
            
            self._explainers[domain_id] = explainer
            return explainer
            
        except Exception as e:
            print(f"Failed to build SHAP explainer for {domain_id}: {e}")
            return None

    def shap_values(self, x: np.ndarray, domain_id: str, top_k: int = 20) -> dict[str, Any]:
        """Compute SHAP values for one encoded instance."""
        # Ensure correct domain is loaded
        domain_model_service.ensure_loaded(domain_id)
        
        # Get prediction
        pred, _, _ = domain_model_service.predict_single(np.asarray(x).reshape(-1))
        
        feature_names = domain_model_service.feature_names
        feature_labels = domain_model_service.feature_labels
        
        default = {
            "feature_names": feature_names,
            "values": [],
            "base_value": 0.0,
            "prediction": pred,
            "feature_labels": feature_labels,
        }
        
        explainer = self._get_or_build_explainer(domain_id)
        if explainer is None:
            return default
        
        try:
            x = np.asarray(x).reshape(1, -1)
            vals = explainer.shap_values(x)
            
            # Handle multi-class output
            if isinstance(vals, list):
                vals = vals[1]  # Use positive class
            
            vals = np.asarray(vals)
            if vals.ndim == 2:
                vals = vals[0]
            
            # Get base value
            base = explainer.expected_value
            if isinstance(base, np.ndarray):
                base = float(base[1]) if len(base) > 1 else float(base[0])
            else:
                base = float(base)
            
            names = feature_names
            if len(names) != len(vals):
                # Mismatch - return default
                return default
            
            # Top-k by absolute value
            idx = np.argsort(np.abs(vals))[::-1][:top_k]
            
            return {
                "feature_names": [names[i] for i in idx],
                "values": [float(vals[i]) for i in idx],
                "base_value": base,
                "prediction": pred,
                "feature_labels": feature_labels,
            }
            
        except Exception as e:
            print(f"SHAP computation error: {e}")
            return default

    def clear_cache(self, domain_id: str | None = None) -> None:
        """Clear cached explainers."""
        if domain_id:
            self._explainers.pop(domain_id, None)
        else:
            self._explainers.clear()


# Singleton
domain_explainability_service = DomainExplainabilityService()
