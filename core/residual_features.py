"""High-pass denoising residual features."""

from __future__ import annotations

import cv2
import numpy as np


RESIDUAL_FEATURE_NAMES = (
    "residual_mean_abs",
    "residual_std",
    "residual_energy",
    "residual_max_abs",
)


def extract_residual_features(grayscale: np.ndarray) -> dict[str, float]:
    values = grayscale.astype(np.float32) / 255.0
    baseline = cv2.GaussianBlur(values, (5, 5), sigmaX=1.0, sigmaY=1.0)
    residual = values - baseline
    absolute = np.abs(residual)
    return {
        "residual_mean_abs": float(np.mean(absolute)),
        "residual_std": float(np.std(residual)),
        "residual_energy": float(np.mean(np.square(residual))),
        "residual_max_abs": float(np.max(absolute)),
    }
