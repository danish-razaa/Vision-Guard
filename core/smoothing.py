"""Weak randomized smoothing views for robust feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from config import Config


@dataclass(frozen=True)
class SmoothingConfig:
    """Configuration shared by dataset extraction and inference."""

    enabled: bool = Config.SMOOTHING_ENABLED
    views: int = Config.SMOOTHING_VIEWS
    sigma: float = Config.SMOOTHING_SIGMA
    light_blur: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not isinstance(self.views, int) or isinstance(self.views, bool) or self.views < 1:
            raise ValueError("views must be a positive integer")
        if not np.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("sigma must be a non-negative finite number")
        if not isinstance(self.light_blur, bool):
            raise ValueError("light_blur must be a boolean")


def _validate_image(image: np.ndarray) -> NDArray[np.uint8]:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("image must be a uint8 grayscale or color array")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("color image must have exactly three channels")
    return image


def generate_smoothing_views(
    image: np.ndarray,
    config: SmoothingConfig | None = None,
    seed: int | None = None,
) -> list[NDArray[np.uint8]]:
    """Return the original image or deterministic weak randomized views.

    Disabled mode returns one independent copy. Enabled mode applies Gaussian
    pixel noise with standard deviation ``sigma``. Optional 3x3 Gaussian blur
    is deliberately light and is applied after noise.
    """
    source = _validate_image(image)
    settings = config or SmoothingConfig()
    if not settings.enabled:
        return [source.copy()]

    rng = np.random.default_rng(seed)
    float_source = source.astype(np.float32)
    results: list[NDArray[np.uint8]] = []
    for _ in range(settings.views):
        if settings.sigma == 0:
            view = float_source.copy()
        else:
            noise = rng.normal(0.0, settings.sigma, size=source.shape).astype(np.float32)
            # Keep smoothing weak even in the tails so anomaly structure survives.
            noise = np.clip(noise, -3.0 * settings.sigma, 3.0 * settings.sigma)
            view = float_source + noise
        view_uint8 = np.clip(view, 0, 255).round().astype(np.uint8)
        if settings.light_blur:
            view_uint8 = cv2.GaussianBlur(view_uint8, (3, 3), sigmaX=0.5)
        results.append(np.ascontiguousarray(view_uint8))
    return results
