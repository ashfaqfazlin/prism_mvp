"""Data loading, validation, and preprocessing for PRISM."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from app.config import settings

FEATURE_COLS = [f"A{i}" for i in range(1, 16)]
TARGET_COL = "A16"
CATEGORICAL = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
NUMERIC = ["A2", "A3", "A8", "A11", "A14", "A15"]


class DataService:
    """Load preprocessing, validate CSV, transform to model input."""

    def __init__(self) -> None:
        self._preproc = None
        self._meta: dict[str, Any] | None = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        base = Path(settings.base_dir)
        path = base / settings.preprocessing_path
        if not path.exists():
            raise FileNotFoundError(
                "Preprocessing artifacts not found. Run: python scripts/train_model.py"
            )
        obj = joblib.load(path)
        self._preproc = obj["preproc"]
        self._meta = obj["meta"]
        self._loaded = True

    @property
    def meta(self) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._meta is not None
        return self._meta

    @property
    def preproc(self):
        self._ensure_loaded()
        return self._preproc

    @property
    def feature_names(self) -> list[str]:
        return list(self.meta.get("encoded_feature_names", []))

    def validate_csv(self, raw: bytes, max_records: int = 50_000) -> tuple[pd.DataFrame, list[str]]:
        """Validate uploaded CSV and return (features_df, errors)."""
        errs: list[str] = []
        try:
            df = pd.read_csv(io.BytesIO(raw), nrows=max_records + 1)
        except Exception as e:
            return pd.DataFrame(), [f"Invalid CSV: {e}"]

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            errs.append(f"Missing columns: {missing}. Required: {FEATURE_COLS}")
            return df, errs
        df = df[FEATURE_COLS].copy()

        # Replace ? with NaN, coerce numerics
        df = df.replace("?", np.nan)
        for c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if len(df) > max_records:
            errs.append(f"Truncated to {max_records} records.")
            df = df.iloc[:max_records]

        return df, errs

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform raw features to encoded matrix."""
        X = X[FEATURE_COLS].copy()
        X = X.replace("?", np.nan)
        for c in NUMERIC:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return self.preproc.transform(X)

    def load_default_dataset(self) -> pd.DataFrame:
        """Load bundled UCI Credit Approval CSV."""
        base = Path(settings.base_dir)
        path = base / settings.default_dataset_path
        if not path.exists():
            raise FileNotFoundError(
                "Default dataset not found. Run: python scripts/train_model.py"
            )
        df = pd.read_csv(path)
        return df[FEATURE_COLS].copy()


data_service = DataService()
