"""Dataset catalog and recent uploads management for PRISM."""
from __future__ import annotations

import json
import io
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import settings
from app.domain_config import is_model_trained


class DatasetService:
    """Manage dataset catalog and recent uploads."""

    def __init__(self) -> None:
        self._datasets_dir = settings.base_dir / "datasets"
        self._catalog_path = self._datasets_dir / "catalog.json"
        self._recent_uploads_path = settings.base_dir / "app_data" / "recent_uploads.json"
        self._catalog: list[dict[str, Any]] | None = None
        self._recent_uploads: list[dict[str, Any]] = []
        self._max_recent = 10  # Keep last 10 uploads
        self._load_recent_uploads()

    def _load_catalog(self) -> list[dict[str, Any]]:
        """Load dataset catalog from JSON."""
        if self._catalog is not None:
            return self._catalog
        
        if not self._catalog_path.exists():
            self._catalog = []
            return self._catalog
        
        with open(self._catalog_path, "r") as f:
            data = json.load(f)
            self._catalog = data.get("datasets", [])
        
        return self._catalog

    def _load_recent_uploads(self) -> None:
        """Load recent uploads from file."""
        if self._recent_uploads_path.exists():
            try:
                with open(self._recent_uploads_path, "r") as f:
                    self._recent_uploads = json.load(f)
            except Exception:
                self._recent_uploads = []

    def _save_recent_uploads(self) -> None:
        """Persist recent uploads to file."""
        self._recent_uploads_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._recent_uploads_path, "w") as f:
            json.dump(self._recent_uploads, f, indent=2)

    def get_catalog(self, featured_only: bool = False) -> list[dict[str, Any]]:
        """Get list of available datasets with dynamic model compatibility check."""
        catalog = self._load_catalog()
        
        # Update model_compatible based on actual training status
        updated = []
        for ds in catalog:
            ds_copy = ds.copy()
            ds_id = ds.get("id", "")
            # Check if model is actually trained for this domain
            ds_copy["model_compatible"] = is_model_trained(ds_id)
            updated.append(ds_copy)
        
        if featured_only:
            return [d for d in updated if d.get("featured", False)]
        return updated

    def get_dataset_info(self, dataset_id: str) -> dict[str, Any] | None:
        """Get info for a specific dataset."""
        catalog = self._load_catalog()
        for ds in catalog:
            if ds["id"] == dataset_id:
                return ds
        return None

    def load_dataset(self, dataset_id: str, limit: int = 100) -> dict[str, Any]:
        """Load a dataset by ID and return sample rows."""
        from app.domain_config import get_domain
        
        info = self.get_dataset_info(dataset_id)
        if not info:
            raise ValueError(f"Dataset '{dataset_id}' not found in catalog")
        
        filepath = self._datasets_dir / info["filename"]
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {info['filename']}")
        
        df = pd.read_csv(filepath)
        
        # Get all columns for display
        columns = df.columns.tolist()
        
        # Sample rows
        sample_df = df.head(limit)
        rows = sample_df.to_dict(orient="records")
        
        # Compute numeric ranges for sliders (with labels from domain config)
        domain = get_domain(dataset_id)
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        # If domain exists, prioritize its numeric_cols to ensure correct feature order
        if domain and domain.numeric_cols:
            # Filter to only columns that exist and are numeric
            numeric_cols = [c for c in domain.numeric_cols if c in df.columns]
        
        ranges = {}
        for col in numeric_cols:
            label = col.replace("_", " ").title()
            if domain and domain.feature_labels:
                label = domain.feature_labels.get(col, label)
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            ranges[col] = {
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "label": label
            }
        
        # Enhanced info with domain labels
        enhanced_info = {**info}
        if domain:
            enhanced_info["positive_label"] = domain.positive_label
            enhanced_info["negative_label"] = domain.negative_label
            enhanced_info["feature_labels"] = domain.feature_labels

        # Dataset summary: class balance (if target column exists)
        summary: dict[str, Any] = {
            "row_count": len(df),
            "feature_count": len(columns),
        }
        target_col = info.get("target") or (domain.target_col if domain else None)
        if target_col and target_col in df.columns:
            value_counts = df[target_col].value_counts()
            summary["class_balance"] = {str(k): int(v) for k, v in value_counts.items()}
            summary["target_column"] = target_col
            if domain:
                pos_val = getattr(domain, "positive_value", None)
                neg_val = getattr(domain, "negative_value", None)
                pos_count = value_counts.get(pos_val, 0) if pos_val is not None else None
                neg_count = value_counts.get(neg_val, 0) if neg_val is not None else None
                if pos_count is not None and neg_count is not None:
                    summary["positive_count"] = int(pos_count)
                    summary["negative_count"] = int(neg_count)

        return {
            "dataset_id": dataset_id,
            "info": enhanced_info,
            "columns": columns,
            "rows": rows,
            "row_count": len(df),
            "feature_ranges": ranges,
            "model_compatible": info.get("model_compatible", False),
            "summary": summary,
        }

    def add_recent_upload(
        self, 
        filename: str, 
        row_count: int, 
        columns: list[str],
        sample_rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Track a new upload in recent uploads list."""
        upload_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        upload_entry = {
            "id": upload_id,
            "filename": filename,
            "uploaded_at": datetime.now().isoformat(),
            "row_count": row_count,
            "columns": columns,
            "sample_rows": sample_rows[:50],  # Store first 50 rows
            "column_count": len(columns)
        }
        
        # Add to front of list
        self._recent_uploads.insert(0, upload_entry)
        
        # Trim to max recent
        self._recent_uploads = self._recent_uploads[:self._max_recent]
        
        # Persist
        self._save_recent_uploads()
        
        return upload_entry

    def get_recent_uploads(self) -> list[dict[str, Any]]:
        """Get list of recent uploads (without full sample rows for listing)."""
        return [
            {
                "id": u["id"],
                "filename": u["filename"],
                "uploaded_at": u["uploaded_at"],
                "row_count": u["row_count"],
                "column_count": u["column_count"]
            }
            for u in self._recent_uploads
        ]

    def get_recent_upload(self, upload_id: str) -> dict[str, Any] | None:
        """Get a specific recent upload with sample rows."""
        for u in self._recent_uploads:
            if u["id"] == upload_id:
                return u
        return None

    def clear_recent_uploads(self) -> None:
        """Clear all recent uploads."""
        self._recent_uploads = []
        self._save_recent_uploads()


# Singleton instance
dataset_service = DatasetService()
