"""Leakage tests for group-aware train/validation/test splitting."""

import numpy as np

from training.utils import group_aware_split_indices


def test_clean_manipulated_pairs_never_cross_splits() -> None:
    groups = np.repeat([f"base_{index:03d}" for index in range(100)], 2)
    split = group_aware_split_indices(groups, seed=42)
    train_groups = set(groups[split.train])
    validation_groups = set(groups[split.validation])
    test_groups = set(groups[split.test])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)
    assert len(split.train) == 140
    assert len(split.validation) == 30
    assert len(split.test) == 30
    assert sorted(np.concatenate((split.train, split.validation, split.test))) == list(range(200))


def test_group_split_is_deterministic() -> None:
    groups = np.repeat(np.arange(20), 2)
    first = group_aware_split_indices(groups, seed=9)
    second = group_aware_split_indices(groups, seed=9)
    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.validation, second.validation)
    assert np.array_equal(first.test, second.test)
