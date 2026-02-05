"""PRISM configuration."""
from pathlib import Path

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
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    
    # Reproducibility (e.g. for model/training)
    random_seed: int = 42
    
    class Config:
        env_prefix = "PRISM_"
        env_file = ".env"


settings = Settings()
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

# Data directory (for recent uploads and runtime data)
data_dir = settings.base_dir / "app_data"
data_dir.mkdir(parents=True, exist_ok=True)
