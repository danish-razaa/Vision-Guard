"""Tests for shared preprocessing and randomized smoothing."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.preprocessing import ImagePreprocessingError, preprocess_image
from core.smoothing import SmoothingConfig, generate_smoothing_views


def test_valid_image_preserves_original_and_outputs_256_square(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    source = np.zeros((20, 40, 3), dtype=np.uint8)
    source[:, :, 2] = 255  # Red in BGR representation
    assert cv2.imwrite(str(path), source)

    result = preprocess_image(path)

    assert result.original_bgr.shape == (20, 40, 3)
    assert result.bgr.shape == (256, 256, 3)
    assert result.rgb.shape == (256, 256, 3)
    assert result.grayscale.shape == (256, 256)
    assert result.image_size == (256, 256)
    assert result.source_path == path.resolve()
    assert np.array_equal(result.bgr[:, :, 2], result.rgb[:, :, 0])
    assert not np.shares_memory(result.original_bgr, result.bgr)


def test_corrupt_and_missing_images_raise_clear_errors(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not image data")

    with pytest.raises(ImagePreprocessingError, match="could not be decoded"):
        preprocess_image(corrupt)
    with pytest.raises(ImagePreprocessingError, match="does not exist"):
        preprocess_image(tmp_path / "missing.png")


def test_array_input_validation_and_custom_size() -> None:
    image = np.full((12, 18, 3), 100, dtype=np.uint8)
    result = preprocess_image(image, image_size=64)
    assert result.bgr.shape == (64, 64, 3)
    assert result.grayscale.shape == (64, 64)

    with pytest.raises(ImagePreprocessingError, match="uint8"):
        preprocess_image(image.astype(np.float32))


def test_disabled_smoothing_returns_one_unchanged_copy() -> None:
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    views = generate_smoothing_views(image, SmoothingConfig(enabled=False), seed=42)
    assert len(views) == 1
    assert np.array_equal(views[0], image)
    assert not np.shares_memory(views[0], image)


def test_seeded_smoothing_is_deterministic_and_weak() -> None:
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    config = SmoothingConfig(enabled=True, views=5, sigma=1.0, light_blur=False)
    first = generate_smoothing_views(image, config, seed=7)
    second = generate_smoothing_views(image, config, seed=7)

    assert len(first) == 5
    assert all(view.shape == image.shape for view in first)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert any(not np.array_equal(view, image) for view in first)
    assert max(np.abs(view.astype(float) - image).max() for view in first) <= 3


def test_smoothing_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="views"):
        SmoothingConfig(enabled=True, views=0)
    with pytest.raises(ValueError, match="sigma"):
        SmoothingConfig(enabled=True, sigma=-1.0)
