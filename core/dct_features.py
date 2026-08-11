"""DCT energy-band features."""

from __future__ import annotations

import cv2
import numpy as np


DCT_FEATURE_NAMES = (
    "dct_total_energy",
    "dct_low_energy",
    "dct_mid_energy",
    "dct_high_energy",
    "dct_high_ratio",
)


def _radial_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return low/mid/high masks using normalized DCT distance from DC.

    Low frequencies occupy normalized radius <= 0.15, mid frequencies occupy
    (0.15, 0.50], and high frequencies occupy > 0.50. Radius is normalized by
    the diagonal distance to the farthest coefficient.
    """
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    denominator = np.hypot(max(height - 1, 1), max(width - 1, 1))
    radius = np.hypot(yy, xx) / denominator
    low = radius <= 0.15
    mid = (radius > 0.15) & (radius <= 0.50)
    high = radius > 0.50
    return low, mid, high


def extract_dct_features(grayscale: np.ndarray) -> dict[str, float]:
    normalized = grayscale.astype(np.float32) / 255.0
    coefficients = cv2.dct(normalized)
    energy = np.square(coefficients, dtype=np.float64)
    low_mask, mid_mask, high_mask = _radial_masks(grayscale.shape)
    total = float(np.sum(energy))
    low = float(np.sum(energy[low_mask]))
    mid = float(np.sum(energy[mid_mask]))
    high = float(np.sum(energy[high_mask]))
    return {
        "dct_total_energy": total,
        "dct_low_energy": low,
        "dct_mid_energy": mid,
        "dct_high_energy": high,
        "dct_high_ratio": high / total if total > 0 else 0.0,
    }
