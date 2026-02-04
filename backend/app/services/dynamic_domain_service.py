"""Dynamic domain creation for user-uploaded datasets.

This service creates DomainConfig objects on-the-fly from uploaded data,
enabling full PRISM functionality for arbitrary datasets.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import settings
from app.domain_config import DomainConfig
from app.services.auto_schema_service import auto_schema_service, SchemaAnalysis


@dataclass
class DynamicDomainConfig:
    """Configuration for a dynamically created domain."""
    upload_id: str
    domain_id: str
    name: str
    description: str
    feature_cols: list[str]
    target_col: str
    categorical_cols: list[str]
    numeric_cols: list[str]
    feature_labels: dict[str, str]
    positive_label: str
    negative_label: str
    positive_value: Any
    negative_value: Any
    created_at: str
    row_count: int
    is_trained: bool = False
    training_accuracy: float | None = None


class DynamicDomainService:
    """Manage dynamically created domains from uploaded data."""

    def __init__(self) -> None:
        self._domains_dir = settings.artifacts_dir / "uploads"
        self._domains_dir.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, DynamicDomainConfig] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load existing dynamic domains from disk."""
        registry_file = self._domains_dir / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    data = json.load(f)
                    for domain_data in data.get("domains", []):
                        config = DynamicDomainConfig(**domain_data)
                        self._registry[config.upload_id] = config
            except Exception as e:
                print(f"Failed to load dynamic domain registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self._domains_dir / "registry.json"
        data = {
            "domains": [asdict(d) for d in self._registry.values()],
            "updated_at": datetime.now().isoformat(),
        }
        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def analyze_upload(
        self, upload_id: str, df: pd.DataFrame
    ) -> dict[str, Any]:
        """Analyze an uploaded dataset and return schema info."""
        analysis = auto_schema_service.analyze(df)
        
        return {
            "upload_id": upload_id,
            "row_count": analysis.row_count,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "unique_count": c.unique_count,
                    "null_count": c.null_count,
                    "sample_values": c.sample_values[:3],
                    "suggested_label": c.suggested_label,
                    "is_target_candidate": c.is_target_candidate,
                }
                for c in analysis.columns
            ],
            "suggested_target": analysis.suggested_target,
            "numeric_cols": analysis.numeric_cols,
            "categorical_cols": analysis.categorical_cols,
            "warnings": analysis.warnings,
        }

    def get_target_info(
        self, upload_id: str, df: pd.DataFrame, target_col: str
    ) -> dict[str, Any]:
        """Get information about a selected target column."""
        return auto_schema_service.get_target_values(df, target_col)

    def create_domain(
        self,
        upload_id: str,
        df: pd.DataFrame,
        target_col: str,
        positive_value: Any,
        negative_value: Any,
        positive_label: str = "Positive",
        negative_label: str = "Negative",
        name: str | None = None,
        description: str | None = None,
        feature_labels: dict[str, str] | None = None,
    ) -> DynamicDomainConfig:
        """Create a dynamic domain configuration from uploaded data."""
        # Analyze schema
        analysis = auto_schema_service.analyze(df)
        
        # Filter out target from features
        feature_cols = [c for c in df.columns if c != target_col]
        numeric_cols = [c for c in analysis.numeric_cols if c != target_col]
        categorical_cols = [c for c in analysis.categorical_cols if c != target_col]
        
        # Generate domain ID
        domain_id = f"upload_{upload_id}"
        
        # Generate feature labels if not provided
        if not feature_labels:
            feature_labels = {}
            for col_info in analysis.columns:
                if col_info.name != target_col:
                    feature_labels[col_info.name] = col_info.suggested_label
        
        # Create config
        config = DynamicDomainConfig(
            upload_id=upload_id,
            domain_id=domain_id,
            name=name or f"Custom Dataset ({upload_id})",
            description=description or f"User-uploaded dataset with {len(df)} rows",
            feature_cols=feature_cols,
            target_col=target_col,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            feature_labels=feature_labels,
            positive_label=positive_label,
            negative_label=negative_label,
            positive_value=positive_value,
            negative_value=negative_value,
            created_at=datetime.now().isoformat(),
            row_count=len(df),
            is_trained=False,
        )
        
        # Save to registry
        self._registry[upload_id] = config
        self._save_registry()
        
        # Create artifacts directory for this domain
        domain_dir = self._domains_dir / domain_id
        domain_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config as JSON
        config_file = domain_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)
        
        return config

    def get_domain(self, upload_id: str) -> DynamicDomainConfig | None:
        """Get a dynamic domain by upload ID."""
        return self._registry.get(upload_id)

    def get_domain_by_domain_id(self, domain_id: str) -> DynamicDomainConfig | None:
        """Get a dynamic domain by its domain_id."""
        for config in self._registry.values():
            if config.domain_id == domain_id:
                return config
        return None

    def to_domain_config(self, config: DynamicDomainConfig) -> DomainConfig:
        """Convert a DynamicDomainConfig to a DomainConfig."""
        return DomainConfig(
            id=config.domain_id,
            name=config.name,
            description=config.description,
            feature_cols=config.feature_cols,
            target_col=config.target_col,
            categorical_cols=config.categorical_cols,
            numeric_cols=config.numeric_cols,
            feature_labels=config.feature_labels,
            positive_label=config.positive_label,
            negative_label=config.negative_label,
            positive_value=config.positive_value,
            negative_value=config.negative_value,
        )

    def mark_trained(
        self, upload_id: str, accuracy: float
    ) -> DynamicDomainConfig | None:
        """Mark a domain as trained with its accuracy."""
        config = self._registry.get(upload_id)
        if config:
            config.is_trained = True
            config.training_accuracy = accuracy
            self._save_registry()
            
            # Update config file
            domain_dir = self._domains_dir / config.domain_id
            config_file = domain_dir / "config.json"
            with open(config_file, "w") as f:
                json.dump(asdict(config), f, indent=2, default=str)
        
        return config

    def list_domains(self) -> list[dict[str, Any]]:
        """List all dynamic domains."""
        return [
            {
                "upload_id": c.upload_id,
                "domain_id": c.domain_id,
                "name": c.name,
                "created_at": c.created_at,
                "row_count": c.row_count,
                "is_trained": c.is_trained,
                "training_accuracy": c.training_accuracy,
            }
            for c in self._registry.values()
        ]

    def delete_domain(self, upload_id: str) -> bool:
        """Delete a dynamic domain."""
        config = self._registry.pop(upload_id, None)
        if config:
            self._save_registry()
            
            # Remove artifacts directory
            domain_dir = self._domains_dir / config.domain_id
            if domain_dir.exists():
                import shutil
                shutil.rmtree(domain_dir)
            
            return True
        return False

    def get_artifacts_path(self, upload_id: str) -> Path | None:
        """Get the artifacts directory for a dynamic domain."""
        config = self._registry.get(upload_id)
        if config:
            return self._domains_dir / config.domain_id
        return None


# Singleton
dynamic_domain_service = DynamicDomainService()
