"""Multi-domain model service for PRISM.

Manages loading and switching between models for different domains/datasets.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from app.domain_config import DomainConfig, get_domain, is_model_trained, DOMAIN_REGISTRY


class DomainModelService:
    """Service for loading and using domain-specific models."""

    def __init__(self) -> None:
        self._current_domain_id: str | None = None
        self._model = None
        self._preproc = None
        self._meta: dict[str, Any] = {}
        self._domain: DomainConfig | None = None

    @property
    def current_domain_id(self) -> str | None:
        return self._current_domain_id

    @property
    def current_domain(self) -> DomainConfig | None:
        return self._domain

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def feature_names(self) -> list[str]:
        return self._meta.get("encoded_feature_names", [])

    @property
    def feature_labels(self) -> dict[str, str]:
        return self._meta.get("feature_labels", {})

    def is_loaded(self, domain_id: str) -> bool:
        """Check if a specific domain is currently loaded."""
        return self._current_domain_id == domain_id and self._model is not None

    def load_domain(self, domain_id: str, force: bool = False) -> bool:
        """Load model and preprocessing for a domain."""
        if not force and self.is_loaded(domain_id):
            return True

        domain = get_domain(domain_id)
        if not domain:
            raise ValueError(f"Unknown domain: {domain_id}")

        if not is_model_trained(domain_id):
            raise FileNotFoundError(
                f"Model not trained for domain '{domain_id}'. "
                f"Run: python scripts/train_all_models.py {domain_id}"
            )

        # Load model
        self._model = joblib.load(domain.model_path)

        # Load preprocessing and meta
        obj = joblib.load(domain.preprocessing_path)
        self._preproc = obj["preproc"]
        self._meta = obj["meta"]

        self._domain = domain
        self._current_domain_id = domain_id

        return True

    def ensure_loaded(self, domain_id: str = "uci_credit_approval") -> None:
        """Ensure a domain is loaded (default to UCI Credit)."""
        if not self.is_loaded(domain_id):
            self.load_domain(domain_id)

    def validate_and_transform(self, raw: bytes | pd.DataFrame, max_records: int = 50000) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
        """Validate input data and transform for prediction.
        
        Returns: (transformed_array, original_df, errors)
        """
        if self._preproc is None or self._domain is None:
            raise RuntimeError("No domain loaded. Call load_domain() first.")

        errors: list[str] = []

        # Parse if bytes
        if isinstance(raw, bytes):
            try:
                df = pd.read_csv(io.BytesIO(raw), nrows=max_records + 1)
            except Exception as e:
                return np.array([]), pd.DataFrame(), [f"Invalid CSV: {e}"]
        else:
            df = raw.copy()

        # Check for required columns
        required = self._domain.feature_cols
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")
            # Try to proceed with available columns
            available = [c for c in required if c in df.columns]
            if not available:
                return np.array([]), df, errors

        # Select feature columns that exist
        feature_cols = [c for c in self._domain.feature_cols if c in df.columns]
        df_features = df[feature_cols].copy()

        # Handle missing values marker
        df_features = df_features.replace("?", np.nan)

        # Convert numeric columns
        for col in self._domain.numeric_cols:
            if col in df_features.columns:
                df_features[col] = pd.to_numeric(df_features[col], errors="coerce")

        # Truncate if needed
        if len(df_features) > max_records:
            errors.append(f"Truncated to {max_records} records.")
            df_features = df_features.iloc[:max_records]

        # Transform
        try:
            X = self._preproc.transform(df_features)
        except Exception as e:
            errors.append(f"Transform error: {e}")
            return np.array([]), df_features, errors

        return X, df_features, errors

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict labels (0/1) and class probabilities."""
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_domain() first.")

        probs = self._model.predict_proba(X)
        labels = (probs[:, 1] >= 0.5).astype(int)
        return labels, probs

    def predict_single(self, x: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Single instance prediction with domain-aware labels."""
        x = x.reshape(1, -1)
        labels, probs = self.predict(x)
        lab = int(labels[0])

        # Use domain-specific labels
        if self._domain:
            pos_label = self._domain.positive_label
            neg_label = self._domain.negative_label
            pred = pos_label if lab == 1 else neg_label
        else:
            pred = "+" if lab == 1 else "-"
            pos_label = "+"
            neg_label = "-"

        conf = float(probs[0, lab])
        probs_d = {pos_label: float(probs[0, 1]), neg_label: float(probs[0, 0])}

        return pred, conf, probs_d

    def get_feature_ranges(self, df: pd.DataFrame | None = None) -> dict[str, dict[str, float]]:
        """Get min/max ranges for numeric features."""
        if self._domain is None:
            return {}

        ranges = {}
        numeric = self._domain.numeric_cols

        if df is not None:
            for col in numeric:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(vals) > 0:
                        ranges[col] = {
                            "min": float(vals.min()),
                            "max": float(vals.max()),
                            "mean": float(vals.mean()),
                        }
        return ranges

    def get_domain_info(self) -> dict[str, Any]:
        """Get info about the currently loaded domain."""
        if self._domain is None:
            return {}

        return {
            "domain_id": self._domain.id,
            "domain_name": self._domain.name,
            "description": self._domain.description,
            "positive_label": self._domain.positive_label,
            "negative_label": self._domain.negative_label,
            "feature_cols": self._domain.feature_cols,
            "feature_labels": self._domain.feature_labels,
            "categorical_cols": self._domain.categorical_cols,
            "numeric_cols": self._domain.numeric_cols,
        }

    def list_available_domains(self) -> list[dict[str, Any]]:
        """List all domains with training status."""
        result = []
        for domain_id, domain in DOMAIN_REGISTRY.items():
            trained = is_model_trained(domain_id)
            result.append({
                "id": domain_id,
                "name": domain.name,
                "description": domain.description,
                "trained": trained,
                "positive_label": domain.positive_label,
                "negative_label": domain.negative_label,
            })
        return result


# Singleton instance
domain_model_service = DomainModelService()
