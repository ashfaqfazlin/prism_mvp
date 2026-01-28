"""PRISM configuration."""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """PRISM app settings."""

    app_name: str = "PRISM"
    debug: bool = False
    
    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    artifacts_dir: Path = Path(__file__).resolve().parent.parent / "artifacts"
    
    # Model (relative to base_dir)
    model_path: str = "artifacts/model.joblib"
    preprocessing_path: str = "artifacts/preprocessing.joblib"
    
    # Data
    default_dataset_path: str = "artifacts/credit_approval.csv"
    
    # API
    max_upload_mb: int = 50
    max_records: int = 50_000
    
    # ============== STUDY CONFIGURATION ==============
    # Study identification
    study_id: str = "prism_study_v1"
    study_version: str = "1.0.0"
    
    # Reproducibility
    random_seed: int = 42  # For task randomization and model consistency
    
    # Study conditions available
    study_conditions: list[str] = ["interactive", "static", "minimal"]
    
    # Task configuration
    tasks_per_session: int = 4
    include_attention_checks: bool = True
    attention_check_frequency: int = 2  # Every N tasks
    randomize_task_order: bool = True
    
    # Data collection
    log_hover_events: bool = True
    log_scroll_events: bool = True
    log_timing_events: bool = True
    
    class Config:
        env_prefix = "PRISM_"
        env_file = ".env"


settings = Settings()
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

# Create study data directory
study_data_dir = settings.base_dir / "study_data"
study_data_dir.mkdir(parents=True, exist_ok=True)
