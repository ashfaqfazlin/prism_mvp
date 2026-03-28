"""Domain-aware SHAP explanations for PRISM multi-model support."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from app.services.domain_model_service import domain_model_service


class DomainExplainabilityService:
    """SHAP explanations with domain-aware model switching."""

    def __init__(self) -> None:
        # Cache explainers by domain_id
        self._explainers: dict[str, shap.TreeExplainer] = {}
        self._current_domain_id: str | None = None
        # Cache global feature importance results to avoid recomputation
        # Keyed by (domain_id, sample_size)
        self._global_importance_cache: dict[tuple[str, int], dict[str, Any]] = {}

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
        dataset_path = domain.dataset_path
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

    def global_feature_importance(self, domain_id: str, sample_size: int = 50) -> dict[str, Any]:
        """Compute mean absolute SHAP across a sample of rows for dataset-level explainability."""
        cache_key = (domain_id, int(sample_size))
        if cache_key in self._global_importance_cache:
            return self._global_importance_cache[cache_key]

        domain_model_service.ensure_loaded(domain_id)
        domain = domain_model_service.current_domain
        if domain is None:
            return {"feature_names": [], "mean_abs_shap": [], "feature_labels": {}, "sample_size": 0}

        dataset_path = domain.dataset_path
        if not dataset_path.exists():
            return {"feature_names": [], "mean_abs_shap": [], "feature_labels": domain.feature_labels or {}, "sample_size": 0}

        def _decode_shap_feature_name(encoded_name: str) -> str:
            # Mirrors the decoding logic used elsewhere in the project (remove encoding prefixes)
            if encoded_name.startswith("num__"):
                return encoded_name[5:]
            if encoded_name.startswith("cat__"):
                rest = encoded_name[5:]
                # E.g. "checking_status_A11" -> "checking_status"
                parts = rest.rsplit("_", 1)
                return parts[0] if len(parts) == 2 else rest
            return encoded_name

        try:
            # Read only a bounded sample to keep this endpoint responsive.
            n = max(1, int(sample_size))
            df = pd.read_csv(dataset_path, nrows=n)
            feature_cols = [c for c in domain.feature_cols if c in df.columns]
            df = df[feature_cols].replace("?", np.nan)
            for c in domain.numeric_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            X = domain_model_service._preproc.transform(df.head(n))
            feature_names = domain_model_service.feature_names
            explainer = self._get_or_build_explainer(domain_id)
            if explainer is None:
                return {"feature_names": [], "mean_abs_shap": [], "feature_labels": domain.feature_labels or {}, "sample_size": 0}

            # Compute SHAP for the whole sample at once (more efficient than per-row loops).
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # class 1
            shap_vals = np.asarray(shap_vals)
            # Ensure shape is (n_samples, n_features)
            if shap_vals.ndim == 3:
                # Common patterns:
                # - (n_samples, n_classes, n_features)
                # - (n_classes, n_samples, n_features)
                if shap_vals.shape[1] in (1, 2):
                    shap_vals = shap_vals[:, 1, :] if shap_vals.shape[1] > 1 else shap_vals[:, 0, :]
                else:
                    shap_vals = shap_vals[0]
            if shap_vals.ndim == 1:
                shap_vals = shap_vals.reshape(1, -1)

            mean_abs_encoded = np.mean(np.abs(shap_vals), axis=0)

            # Aggregate encoded-feature importances back to original (human) feature names.
            # This avoids showing "num__/cat__" tokens in the UI.
            agg: dict[str, float] = {}
            for j, enc_name in enumerate(feature_names):
                orig = _decode_shap_feature_name(str(enc_name))
                agg[orig] = agg.get(orig, 0.0) + float(mean_abs_encoded[j])

            sorted_orig = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
            top = sorted_orig[:10]

            result = {
                "feature_names": [k for k, _ in top],
                "mean_abs_shap": [v for _, v in top],
                "feature_labels": domain.feature_labels or {},
                "sample_size": int(min(n, len(X))),
            }
            self._global_importance_cache[cache_key] = result
            return result
        except Exception as e:
            print(f"Global SHAP failed for {domain_id}: {e}")
            result = {"feature_names": [], "mean_abs_shap": [], "feature_labels": domain.feature_labels or {}, "sample_size": 0}
            self._global_importance_cache[cache_key] = result
            return result

    def clear_cache(self, domain_id: str | None = None) -> None:
        """Clear cached explainers."""
        if domain_id:
            self._explainers.pop(domain_id, None)
        else:
            self._explainers.clear()


# Singleton
domain_explainability_service = DomainExplainabilityService()
