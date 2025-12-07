"""
F5-TTS API Server Configuration
Centralized configuration for all settings
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    WORKERS: int = 1
    DEBUG: bool = False

    # Paths
    BASE_DIR: Path = Path(__file__).parent
    MODELS_DIR: Path = BASE_DIR / "models"
    CHECKPOINTS_DIR: Path = BASE_DIR / "checkpoints"
    DATASETS_DIR: Path = BASE_DIR / "datasets"
    TEMP_DIR: Path = BASE_DIR / "temp"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # F5-TTS Model Settings
    MODEL_NAME: str = "F5TTS_v1_Base"
    DEVICE: Optional[str] = None  # Auto-detect if None
    USE_EMA: bool = True

    # Audio Settings
    TARGET_SAMPLE_RATE: int = 24000
    N_MEL_CHANNELS: int = 100
    HOP_LENGTH: int = 256
    WIN_LENGTH: int = 1024
    N_FFT: int = 1024
    MEL_SPEC_TYPE: str = "vocos"  # 'vocos' or 'bigvgan'

    # Generation Settings
    DEFAULT_NFE_STEP: int = 32
    DEFAULT_CFG_STRENGTH: float = 2.0
    DEFAULT_SWAY_SAMPLING_COEF: float = -1.0
    DEFAULT_SPEED: float = 1.0
    MAX_TEXT_LENGTH: int = 50000

    # Training Settings
    DEFAULT_LEARNING_RATE: float = 1e-5
    DEFAULT_BATCH_SIZE: int = 3200
    DEFAULT_EPOCHS: int = 100
    DEFAULT_WARMUP_UPDATES: int = 20000
    DEFAULT_SAVE_PER_UPDATES: int = 5000
    DEFAULT_GRAD_ACCUMULATION: int = 1
    DEFAULT_MAX_GRAD_NORM: float = 1.0

    # CORS Settings
    CORS_ORIGINS: list = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Create directories if not exist
for dir_path in [settings.MODELS_DIR, settings.CHECKPOINTS_DIR,
                 settings.DATASETS_DIR, settings.TEMP_DIR, settings.LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)