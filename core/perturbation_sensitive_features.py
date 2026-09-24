"""Features targeting weak structured pixel perturbations without a reference image."""

from __future__ import annotations

import cv2
import numpy as np


SCALES = (0.5, 1.0, 2.0)
PERTURBATION_SENSITIVE_FEATURE_NAMES = tuple(
    f"msres_sigma_{str(scale).replace('.', '_')}_{stat}"
    for scale in SCALES
    for stat in ("mean_abs", "std", "energy")
) + (
    "directional_gradient_x_energy",
    "directional_gradient_y_energy",
    "directional_gradient_diagonal_energy",
    "directional_gradient_anisotropy",
    "spectral_peak_top1_ratio",
    "spectral_peak_high_zscore",
    "spectral_local_variation",
    "color_residual_rg_std",
    "color_residual_rb_std",
    "color_residual_gb_std",
    "color_residual_chroma_energy",
)


def extract_perturbation_sensitive_features(bgr_image: np.ndarray) -> dict[str, float]:
    """Extract multi-scale, directional, spectral-peak, and chroma residual signals."""
    values = bgr_image.astype(np.float32) / 255.0
    gray = cv2.cvtColor(values, cv2.COLOR_BGR2GRAY)
    features: dict[str, float] = {}
    for sigma in SCALES:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        residual = gray - blurred
        prefix = f"msres_sigma_{str(sigma).replace('.', '_')}"
        features[f"{prefix}_mean_abs"] = float(np.mean(np.abs(residual)))
        features[f"{prefix}_std"] = float(np.std(residual))
        features[f"{prefix}_energy"] = float(np.mean(np.square(residual)))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    diagonal = (gx + gy) / np.sqrt(2.0)
    ex, ey, ed = (float(np.mean(np.square(item))) for item in (gx, gy, diagonal))
    features.update({
        "directional_gradient_x_energy": ex,
        "directional_gradient_y_energy": ey,
        "directional_gradient_diagonal_energy": ed,
        "directional_gradient_anisotropy": abs(ex - ey) / max(ex + ey, 1e-12),
    })

    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    height, width = magnitude.shape
    magnitude[height // 2, width // 2] = 0.0
    flat = magnitude.ravel()
    cutoff = float(np.quantile(flat, 0.99))
    total = float(np.sum(flat))
    high = flat[flat >= cutoff]
    block_means = magnitude.reshape(16, height // 16, 16, width // 16).mean(axis=(1, 3))
    features.update({
        "spectral_peak_top1_ratio": float(np.sum(high) / total) if total else 0.0,
        "spectral_peak_high_zscore": float((np.max(flat) - np.mean(flat)) / max(np.std(flat), 1e-12)),
        "spectral_local_variation": float(np.std(block_means) / max(np.mean(block_means), 1e-12)),
    })

    # Channel-difference high-pass residuals reveal chromatic patterns that
    # grayscale fusion can suppress.
    blue, green, red = cv2.split(values)
    chroma_residuals = []
    for name, difference in (("rg", red-green), ("rb", red-blue), ("gb", green-blue)):
        residual = difference - cv2.GaussianBlur(difference, (0, 0), 1.0)
        chroma_residuals.append(residual)
        features[f"color_residual_{name}_std"] = float(np.std(residual))
    features["color_residual_chroma_energy"] = float(np.mean([np.mean(np.square(item)) for item in chroma_residuals]))
    return features
