"""Tests for controlled Chapter 3 perturbation simulations."""

import numpy as np
import pytest

from training.generate_perturbations import ATTACK_TYPES, generate_perturbation


@pytest.mark.parametrize("attack_type", ATTACK_TYPES)
def test_perturbations_are_deterministic_and_shape_safe(attack_type: str) -> None:
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    first = generate_perturbation(image, attack_type, epsilon=3.0, seed=42)
    second = generate_perturbation(image, attack_type, epsilon=3.0, seed=42)

    assert first.shape == image.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert np.any(first != image)


@pytest.mark.parametrize("attack_type", ATTACK_TYPES)
def test_perturbations_remain_visually_weak(attack_type: str) -> None:
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    manipulated = generate_perturbation(image, attack_type, epsilon=5.0, seed=7)
    mean_absolute_delta = np.abs(manipulated.astype(float) - image).mean()
    assert mean_absolute_delta <= 8.0
