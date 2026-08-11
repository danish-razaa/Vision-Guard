"""Chapter 1 configuration smoke tests."""

from pathlib import Path

from config import Config


def test_default_configuration_types() -> None:
    assert isinstance(Config.UPLOAD_FOLDER, Path)
    assert isinstance(Config.MODEL_PATH, Path)
    assert isinstance(Config.FEATURE_COLUMNS_PATH, Path)
    assert isinstance(Config.MODEL_METADATA_PATH, Path)
    assert Config.IMAGE_SIZE == 256
    assert Config.SMOOTHING_VIEWS == 5
    assert Config.SMOOTHING_SIGMA >= 0


def test_allowed_extensions_are_safe_image_types() -> None:
    assert Config.ALLOWED_EXTENSIONS == {"jpg", "jpeg", "png"}
