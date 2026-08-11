"""FFT magnitude and radial spectral features."""

from __future__ import annotations

import numpy as np


FFT_FEATURE_NAMES = (
    "fft_magnitude_mean",
    "fft_magnitude_std",
    "fft_magnitude_max",
    "fft_low_ratio",
    "fft_mid_ratio",
    "fft_high_ratio",
    "spectral_entropy",
)


def _centered_radial_masks(
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use normalized radius <= .15, (.15, .50], and > .50 from FFT DC."""
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    cy, cx = height // 2, width // 2
    denominator = np.hypot(max(cy, height - 1 - cy, 1), max(cx, width - 1 - cx, 1))
    radius = np.hypot(yy - cy, xx - cx) / denominator
    low = radius <= 0.15
    mid = (radius > 0.15) & (radius <= 0.50)
    high = radius > 0.50
    return low, mid, high


def extract_fft_features(grayscale: np.ndarray) -> dict[str, float]:
    normalized = grayscale.astype(np.float64) / 255.0
    spectrum = np.fft.fftshift(np.fft.fft2(normalized))
    magnitude = np.abs(spectrum)
    power = np.square(magnitude)
    low_mask, mid_mask, high_mask = _centered_radial_masks(grayscale.shape)
    total_power = float(np.sum(power))
    probabilities = power.ravel()
    if total_power > 0:
        probabilities = probabilities / total_power
        nonzero = probabilities > 0
        entropy = float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))
    else:
        entropy = 0.0
    return {
        "fft_magnitude_mean": float(np.mean(magnitude)),
        "fft_magnitude_std": float(np.std(magnitude)),
        "fft_magnitude_max": float(np.max(magnitude)),
        "fft_low_ratio": float(np.sum(power[low_mask]) / total_power) if total_power else 0.0,
        "fft_mid_ratio": float(np.sum(power[mid_mask]) / total_power) if total_power else 0.0,
        "fft_high_ratio": float(np.sum(power[high_mask]) / total_power) if total_power else 0.0,
        "spectral_entropy": entropy,
    }
