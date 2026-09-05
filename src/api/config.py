import os
from pathlib import Path
from typing import List

class Settings:
    """
    VoxGuard AI Application Configuration Settings.
    Manages neural model paths, acoustic feature thresholds, authentication, and persistence layers.
    """
    # Base Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # Model Artifact Path
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(MODELS_DIR / "best_model.onnx"))
    
    # Acoustic Explainability Feature Thresholds (Praat / Librosa standard voice metrics)
    JITTER_THRESHOLD_LOW: float = float(os.getenv("JITTER_THRESHOLD_LOW", "0.002"))    # < 0.20% -> Unnatural / Robotic pitch
    JITTER_THRESHOLD_HIGH: float = float(os.getenv("JITTER_THRESHOLD_HIGH", "0.015"))  # > 1.50% -> Vocal perturbation / noise
    SHIMMER_THRESHOLD: float = float(os.getenv("SHIMMER_THRESHOLD", "0.050"))          # > 5.00% -> Amplitude variation
    HNR_THRESHOLD: float = float(os.getenv("HNR_THRESHOLD", "20.0"))                  # < 20.0 dB -> Phase/Vocoder Noise
    
    # Temperature Calibration Value (calibrated via ECE minimization on validation split)
    CALIBRATED_TEMPERATURE: float = float(os.getenv("CALIBRATED_TEMPERATURE", "1.362"))
    
    # API Security: Allowed API keys (configurable via ENV or default development keys)
    API_KEYS: List[str] = [
        k.strip() for k in os.getenv("API_KEYS", "voxguard-demo-key-2026,voxguard-college-eval-key,admin-secret-key").split(",") if k.strip()
    ]
    
    # Cloud Database Logging (Supabase - Optional)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    
    # Local SQLite Audit Database (Automatic offline fallback for college demos)
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", str(LOGS_DIR / "audit_trail.db"))
    
    # Supported Languages for Inference
    SUPPORTED_LANGUAGES: List[str] = ["English", "Hindi", "Tamil", "Telugu", "Malayalam"]

settings = Settings()
