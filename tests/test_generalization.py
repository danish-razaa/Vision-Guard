"""Tests for honest cohort metrics and research prompt isolation."""

import numpy as np

from training.evaluate_generalization import calculate_metrics
from training.prompt_banks import TEST_PROMPTS, TRAIN_PROMPTS, VALIDATION_PROMPTS


def test_generalization_metrics_include_error_rates() -> None:
    result = calculate_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.9, 0.8, 0.2]), 0.5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["false_positive_rate"] == 0.5
    assert result["false_negative_rate"] == 0.5
    assert result["roc_auc"] == 0.5


def test_single_class_cohort_has_no_roc_auc() -> None:
    result = calculate_metrics(np.zeros(3), np.array([0.1, 0.2, 0.3]), 0.5)
    assert result["roc_auc"] is None
    assert result["false_negative_rate"] is None


def test_prompt_banks_are_disjoint() -> None:
    train, validation, test = map(set, (TRAIN_PROMPTS, VALIDATION_PROMPTS, TEST_PROMPTS))
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
