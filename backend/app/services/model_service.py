"""PRISM decision engine: model loading and inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import joblib

from app.config import settings
from app.services.data_service import data_service


class ModelService:
    """PRISM decision engine: load model, produce decision and confidence."""

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        base = Path(settings.base_dir)
        path = base / settings.model_path
        if not path.exists():
            raise FileNotFoundError(
                "Model not found. Run: python scripts/train_model.py"
            )
        self._model = joblib.load(path)
        self._loaded = True

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict labels (0/1) and class probabilities. Returns (labels, probs)."""
        self._ensure_loaded()
        data_service._ensure_loaded()
        probs = self.model.predict_proba(X)
        labels = (probs[:, 1] >= 0.5).astype(int)
        return labels, probs

    def predict_single(self, x: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Single instance -> (decision, confidence, {class: prob})."""
        x = x.reshape(1, -1)
        labels, probs = self.predict(x)
        lab = int(labels[0])
        pred = "+" if lab == 1 else "-"
        conf = float(probs[0, lab])
        probs_d = {"+": float(probs[0, 1]), "-": float(probs[0, 0])}
        return pred, conf, probs_d


model_service = ModelService()
