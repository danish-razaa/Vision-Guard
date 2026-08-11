"""Global grayscale statistical features."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew
from skimage.measure import shannon_entropy


STATISTICAL_FEATURE_NAMES = (
    "stat_mean",
    "stat_variance",
    "stat_std",
    "stat_skewness",
    "stat_kurtosis",
    "stat_shannon_entropy",
    "stat_median",
    "stat_min",
    "stat_max",
)


def extract_statistical_features(grayscale: np.ndarray) -> dict[str, float]:
    values = grayscale.astype(np.float64).ravel()
    standard_deviation = float(np.std(values))
    if standard_deviation <= np.finfo(float).eps:
        skewness = 0.0
        kurtosis_value = 0.0
    else:
        skewness = float(skew(values, bias=False))
        kurtosis_value = float(kurtosis(values, fisher=True, bias=False))
    return {
        "stat_mean": float(np.mean(values)),
        "stat_variance": float(np.var(values)),
        "stat_std": standard_deviation,
        "stat_skewness": skewness,
        "stat_kurtosis": kurtosis_value,
        "stat_shannon_entropy": float(shannon_entropy(grayscale, base=2)),
        "stat_median": float(np.median(values)),
        "stat_min": float(np.min(values)),
        "stat_max": float(np.max(values)),
    }
