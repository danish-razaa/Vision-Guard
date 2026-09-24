"""Stable multi-domain feature fusion shared by training and inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.dct_features import DCT_FEATURE_NAMES, extract_dct_features
from core.fft_features import FFT_FEATURE_NAMES, extract_fft_features
from core.preprocessing import preprocess_image
from core.perturbation_sensitive_features import (
    PERTURBATION_SENSITIVE_FEATURE_NAMES,
    extract_perturbation_sensitive_features,
)
from core.residual_features import RESIDUAL_FEATURE_NAMES, extract_residual_features
from core.smoothing import SmoothingConfig, generate_smoothing_views
from core.statistical_features import (
    STATISTICAL_FEATURE_NAMES,
    extract_statistical_features,
)
from core.texture_features import TEXTURE_FEATURE_NAMES, extract_texture_features


FEATURE_NAMES = (
    STATISTICAL_FEATURE_NAMES
    + DCT_FEATURE_NAMES
    + FFT_FEATURE_NAMES
    + TEXTURE_FEATURE_NAMES
    + PERTURBATION_SENSITIVE_FEATURE_NAMES
    + RESIDUAL_FEATURE_NAMES
)


def get_feature_names() -> list[str]:
    """Return a copy of the canonical ordered feature names."""
    return list(FEATURE_NAMES)


def _single_view_features(bgr_image: np.ndarray) -> dict[str, float]:
    grayscale = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    features: dict[str, float] = {}
    for extractor in (
        extract_statistical_features,
        extract_dct_features,
        extract_fft_features,
        extract_texture_features,
        extract_residual_features,
    ):
        features.update(extractor(grayscale))
    features.update(extract_perturbation_sensitive_features(bgr_image))
    return features


def _parse_smoothing_config(
    smoothing_config: SmoothingConfig | Mapping[str, Any] | None,
) -> tuple[SmoothingConfig, int]:
    if smoothing_config is None:
        return SmoothingConfig(), 42
    if isinstance(smoothing_config, SmoothingConfig):
        return smoothing_config, 42
    if isinstance(smoothing_config, Mapping):
        values = dict(smoothing_config)
        seed = int(values.pop("seed", 42))
        return SmoothingConfig(**values), seed
    raise TypeError("smoothing_config must be SmoothingConfig, mapping, or None")


def extract_features(
    image_path: str | Path,
    smoothing_config: SmoothingConfig | Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Extract the canonical finite feature vector from one image path."""
    preprocessed = preprocess_image(image_path)
    settings, seed = _parse_smoothing_config(smoothing_config)
    views = generate_smoothing_views(preprocessed.bgr, settings, seed=seed)
    view_vectors = [_single_view_features(view) for view in views]
    fused = {
        name: float(np.mean([vector[name] for vector in view_vectors]))
        for name in FEATURE_NAMES
    }
    values = np.asarray(list(fused.values()), dtype=np.float64)
    if values.size != len(FEATURE_NAMES) or not np.all(np.isfinite(values)):
        raise ValueError("feature extraction produced a non-finite or incomplete vector")
    return fused
