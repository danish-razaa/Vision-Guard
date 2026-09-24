"""Deterministic, bounded prompt-conditioned perturbation simulation."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from PIL import Image


METHOD = "prompt_conditioned_spectral"
MIN_EPSILON = 0.5
MAX_EPSILON = 5.0


def _array(image: np.ndarray | Image.Image) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if value.ndim != 3 or value.shape[2] not in (3, 4) or value.size == 0:
        raise ValueError("image must be a non-empty RGB/RGBA image")
    return value[..., :3].astype(np.uint8, copy=True)


def apply_prompt_perturbation(
    image: np.ndarray | Image.Image, perturbation: np.ndarray, epsilon: float
) -> np.ndarray:
    """Apply a unit-amplitude perturbation without modifying the source."""
    source = _array(image).astype(np.float32)
    delta = np.asarray(perturbation, dtype=np.float32)
    if delta.shape != source.shape or not np.all(np.isfinite(delta)):
        raise ValueError("perturbation must be finite and match the image shape")
    if not np.isfinite(epsilon) or not MIN_EPSILON <= epsilon <= MAX_EPSILON:
        raise ValueError(f"epsilon must be between {MIN_EPSILON} and {MAX_EPSILON}")
    delta = np.clip(delta, -1.0, 1.0) * float(epsilon)
    return np.clip(source + delta, 0, 255).round().astype(np.uint8)


def generate_prompt_conditioned_perturbation(
    image: np.ndarray | Image.Image,
    prompt: str,
    epsilon: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return modified RGB image, unit perturbation, and non-sensitive metadata."""
    source = _array(image)
    clean_prompt = prompt.strip() if isinstance(prompt, str) else ""
    if not clean_prompt:
        raise ValueError("prompt must not be empty")
    digest = hashlib.sha256(clean_prompt.encode("utf-8")).digest()
    prompt_hash = digest.hex()
    effective_seed = int.from_bytes(digest[:8], "big") if seed is None else int(seed)
    rng = np.random.default_rng(effective_seed)
    height, width = source.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)

    # Prompt-seeded spatial frequency, phase, orientation, and channel weights.
    frequency = int(rng.integers(8, 25))
    angle = float(rng.uniform(0.0, np.pi))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    coordinate = xx * np.cos(angle) + yy * np.sin(angle)
    sinusoid = np.sin(
        2.0 * np.pi * frequency * coordinate / max(height, width) + phase
    ).astype(np.float32)
    frequency_y = int(rng.integers(1, max(2, height // 3)))
    frequency_x = int(rng.integers(1, max(2, width // 3)))
    dct_like = (
        np.cos(np.pi * (2 * yy + 1) * frequency_y / (2 * height))
        * np.cos(np.pi * (2 * xx + 1) * frequency_x / (2 * width))
    ).astype(np.float32)
    structured_noise = rng.normal(0.0, 0.18, (height, width)).astype(np.float32)
    spatial = 0.62 * sinusoid + 0.30 * dct_like + 0.08 * structured_noise
    spatial /= max(float(np.max(np.abs(spatial))), np.finfo(np.float32).eps)
    channel_weights = rng.uniform(0.65, 1.0, 3).astype(np.float32)
    channel_signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), 3)
    perturbation = spatial[..., None] * channel_weights * channel_signs
    perturbation /= max(
        float(np.max(np.abs(perturbation))), np.finfo(np.float32).eps
    )
    modified = apply_prompt_perturbation(source, perturbation, epsilon)
    metadata = {
        "prompt_hash": prompt_hash,
        "seed": effective_seed,
        "epsilon": float(epsilon),
        "perturbation_method": METHOD,
    }
    return modified, perturbation.astype(np.float32), metadata
