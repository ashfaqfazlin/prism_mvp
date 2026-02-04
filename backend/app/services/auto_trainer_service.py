"""Auto-training service for dynamically uploaded datasets.

This service trains XGBoost models on user-uploaded data to enable
full PRISM functionality (predictions, SHAP, counterfactuals).
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from app.config import settings
from app.services.dynamic_domain_service import (
    dynamic_domain_service,
    DynamicDomainConfig,
)


class TrainingStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingJob:
    """Represents a training job."""
    job_id: str
    upload_id: str
    status: TrainingStatus
    started_at: str | None = None
    completed_at: str | None = None
    accuracy: float | None = None
    error: str | None = None
    progress: int = 0  # 0-100


class AutoTrainerService:
    """Train models on dynamically uploaded datasets."""

    def __init__(self) -> None:
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def start_training(
        self,
        upload_id: str,
        df: pd.DataFrame,
        config: DynamicDomainConfig,
    ) -> TrainingJob:
        """Start a training job for an uploaded dataset."""
        job_id = f"train_{upload_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job = TrainingJob(
            job_id=job_id,
            upload_id=upload_id,
            status=TrainingStatus.IN_PROGRESS,
            started_at=datetime.now().isoformat(),
            progress=0,
        )
        
        with self._lock:
            self._jobs[job_id] = job
        
        # Run training in a thread (for non-blocking API)
        thread = threading.Thread(
            target=self._train_model,
            args=(job_id, df, config),
            daemon=True,
        )
        thread.start()
        
        return job

    def start_training_sync(
        self,
        upload_id: str,
        df: pd.DataFrame,
        config: DynamicDomainConfig,
    ) -> TrainingJob:
        """Start a synchronous training job (blocking)."""
        job_id = f"train_{upload_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job = TrainingJob(
            job_id=job_id,
            upload_id=upload_id,
            status=TrainingStatus.IN_PROGRESS,
            started_at=datetime.now().isoformat(),
            progress=0,
        )
        
        with self._lock:
            self._jobs[job_id] = job
        
        # Train synchronously
        self._train_model(job_id, df, config)
        
        return self._jobs[job_id]

    def get_job(self, job_id: str) -> TrainingJob | None:
        """Get a training job by ID."""
        return self._jobs.get(job_id)

    def get_job_by_upload(self, upload_id: str) -> TrainingJob | None:
        """Get the most recent training job for an upload."""
        matching = [j for j in self._jobs.values() if j.upload_id == upload_id]
        if matching:
            return max(matching, key=lambda x: x.started_at or "")
        return None

    def _train_model(
        self, job_id: str, df: pd.DataFrame, config: DynamicDomainConfig
    ) -> None:
        """Train a model for the given configuration."""
        try:
            self._update_job(job_id, progress=5)
            
            # Prepare data
            X, y = self._prepare_data(df, config)
            self._update_job(job_id, progress=15)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=settings.random_seed, stratify=y
            )
            self._update_job(job_id, progress=20)
            
            # Build preprocessing
            preproc = self._build_preprocessing(X_train, config)
            X_train_t = preproc.fit_transform(X_train)
            X_test_t = preproc.transform(X_test)
            self._update_job(job_id, progress=35)
            
            # Get feature names
            feature_names = self._get_feature_names(preproc, config)
            self._update_job(job_id, progress=40)
            
            # Train XGBoost
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=settings.random_seed,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            model.fit(X_train_t, y_train)
            self._update_job(job_id, progress=75)
            
            # Evaluate
            y_pred = model.predict(X_test_t)
            accuracy = accuracy_score(y_test, y_pred)
            self._update_job(job_id, progress=85)
            
            # Save artifacts
            self._save_artifacts(
                config, model, preproc, feature_names, accuracy
            )
            self._update_job(job_id, progress=95)
            
            # Mark domain as trained
            dynamic_domain_service.mark_trained(config.upload_id, accuracy)
            
            # Update job status
            self._update_job(
                job_id,
                status=TrainingStatus.COMPLETED,
                completed_at=datetime.now().isoformat(),
                accuracy=accuracy,
                progress=100,
            )
            
        except Exception as e:
            self._update_job(
                job_id,
                status=TrainingStatus.FAILED,
                completed_at=datetime.now().isoformat(),
                error=str(e),
            )

    def _prepare_data(
        self, df: pd.DataFrame, config: DynamicDomainConfig
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        df = df.copy()
        
        # Handle missing values marker
        df = df.replace("?", np.nan)
        
        # Convert numeric columns
        for col in config.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Get target
        target_col = config.target_col
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Map target to binary
        pos_val = config.positive_value
        neg_val = config.negative_value
        
        def map_target(val):
            # Handle type conversions
            val_str = str(val).strip().lower()
            pos_str = str(pos_val).strip().lower()
            neg_str = str(neg_val).strip().lower()
            
            if val == pos_val or val_str == pos_str:
                return 1
            elif val == neg_val or val_str == neg_str:
                return 0
            else:
                return np.nan
        
        df["_target"] = df[target_col].apply(map_target)
        
        # Drop rows with missing target
        df = df.dropna(subset=["_target"])
        
        if len(df) == 0:
            raise ValueError("No valid target values found after mapping")
        
        # Get features
        feature_cols = [c for c in config.feature_cols if c in df.columns]
        X = df[feature_cols].copy()
        y = df["_target"].astype(int)
        
        return X, y

    def _build_preprocessing(
        self, X: pd.DataFrame, config: DynamicDomainConfig
    ) -> ColumnTransformer:
        """Build preprocessing pipeline."""
        cat_cols = [c for c in config.categorical_cols if c in X.columns]
        num_cols = [c for c in config.numeric_cols if c in X.columns]
        
        transformers = []
        
        if cat_cols:
            cat_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipe, cat_cols))
        
        if num_cols:
            num_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, num_cols))
        
        return ColumnTransformer(transformers, remainder="drop")

    def _get_feature_names(
        self, preproc: ColumnTransformer, config: DynamicDomainConfig
    ) -> list[str]:
        """Extract feature names after preprocessing."""
        if hasattr(preproc, "get_feature_names_out"):
            return list(preproc.get_feature_names_out())
        
        # Manual extraction
        names = []
        for name, pipe, cols in preproc.transformers_:
            if name == "cat":
                ohe = pipe.named_steps["ohe"]
                for i, c in enumerate(cols):
                    for cat in ohe.categories_[i]:
                        names.append(f"{c}_{cat}")
            elif name == "num":
                names.extend(cols)
        return names

    def _save_artifacts(
        self,
        config: DynamicDomainConfig,
        model: xgb.XGBClassifier,
        preproc: ColumnTransformer,
        feature_names: list[str],
        accuracy: float,
    ) -> None:
        """Save model artifacts to disk."""
        artifacts_dir = dynamic_domain_service.get_artifacts_path(config.upload_id)
        if not artifacts_dir:
            raise ValueError(f"No artifacts path for upload {config.upload_id}")
        
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, artifacts_dir / "model.joblib")
        
        # Save preprocessing with metadata
        meta = {
            "domain_id": config.domain_id,
            "domain_name": config.name,
            "feature_cols": config.feature_cols,
            "target_col": config.target_col,
            "categorical": config.categorical_cols,
            "numeric": config.numeric_cols,
            "encoded_feature_names": feature_names,
            "positive_label": config.positive_label,
            "negative_label": config.negative_label,
            "feature_labels": config.feature_labels,
            "test_accuracy": accuracy,
        }
        
        joblib.dump({"preproc": preproc, "meta": meta}, artifacts_dir / "preprocessing.joblib")
        
        # Save meta as JSON
        with open(artifacts_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def _update_job(self, job_id: str, **updates) -> None:
        """Update job attributes."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for key, value in updates.items():
                    setattr(job, key, value)


# Singleton
auto_trainer_service = AutoTrainerService()
