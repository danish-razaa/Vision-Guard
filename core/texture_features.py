"""Uniform LBP histogram and quantized multi-angle GLCM features."""

from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = LBP_POINTS + 2
GLCM_LEVELS = 16
GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
LBP_FEATURE_NAMES = tuple(f"lbp_uniform_bin_{index:02d}" for index in range(LBP_BINS))
GLCM_FEATURE_NAMES = tuple(
    f"glcm_{property_name}_{statistic}"
    for property_name in GLCM_PROPERTIES
    for statistic in ("mean", "std")
)
TEXTURE_FEATURE_NAMES = LBP_FEATURE_NAMES + GLCM_FEATURE_NAMES


def extract_texture_features(grayscale: np.ndarray) -> dict[str, float]:
    lbp = local_binary_pattern(
        grayscale,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform",
    )
    histogram, _ = np.histogram(lbp, bins=np.arange(LBP_BINS + 1), range=(0, LBP_BINS))
    histogram = histogram.astype(np.float64)
    histogram /= histogram.sum() if histogram.sum() else 1.0
    features = {
        name: float(value) for name, value in zip(LBP_FEATURE_NAMES, histogram, strict=True)
    }

    quantized = np.floor(grayscale.astype(np.float32) * GLCM_LEVELS / 256.0)
    quantized = np.clip(quantized, 0, GLCM_LEVELS - 1).astype(np.uint8)
    matrix = graycomatrix(
        quantized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    for property_name in GLCM_PROPERTIES:
        values = graycoprops(matrix, property_name).ravel()
        features[f"glcm_{property_name}_mean"] = float(np.mean(values))
        features[f"glcm_{property_name}_std"] = float(np.std(values))
    return features
