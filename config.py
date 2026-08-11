"""Application configuration for VisionGuard."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """Environment-overridable defaults shared by training and inference."""

    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", BASE_DIR / "static" / "uploads"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))
    MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "best_model.pkl"))
    FEATURE_COLUMNS_PATH = Path(
        os.getenv("FEATURE_COLUMNS_PATH", BASE_DIR / "models" / "feature_columns.pkl")
    )
    MODEL_METADATA_PATH = Path(
        os.getenv("MODEL_METADATA_PATH", BASE_DIR / "models" / "model_metadata.json")
    )
    ALLOWED_EXTENSIONS = frozenset(
        extension.strip().lower()
        for extension in os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png").split(",")
        if extension.strip()
    )
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
    SMOOTHING_ENABLED = _env_bool("SMOOTHING_ENABLED", False)
    SMOOTHING_VIEWS = int(os.getenv("SMOOTHING_VIEWS", "5"))
    SMOOTHING_SIGMA = float(os.getenv("SMOOTHING_SIGMA", "1.0"))
