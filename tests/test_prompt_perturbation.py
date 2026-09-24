"""Unit tests for bounded prompt-conditioned perturbations."""

import numpy as np

from core.prompt_perturbation import generate_prompt_conditioned_perturbation


def test_prompt_perturbation_is_deterministic_bounded_and_non_mutating() -> None:
    image = np.full((48, 52, 3), 127, dtype=np.uint8)
    baseline = image.copy()
    first, perturbation, metadata = generate_prompt_conditioned_perturbation(image, "alpha", 2.0)
    second, second_perturbation, second_metadata = generate_prompt_conditioned_perturbation(image, "alpha", 2.0)
    assert np.array_equal(first, second)
    assert np.array_equal(perturbation, second_perturbation)
    assert metadata == second_metadata
    assert np.array_equal(image, baseline)
    assert np.max(np.abs(first.astype(float) - image.astype(float))) <= 2.0
    assert first.dtype == np.uint8 and first.min() >= 0 and first.max() <= 255


def test_different_prompts_make_different_perturbations() -> None:
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    _, first, _ = generate_prompt_conditioned_perturbation(image, "alpha", 1.0)
    _, second, _ = generate_prompt_conditioned_perturbation(image, "beta", 1.0)
    assert not np.array_equal(first, second)
