"""Shared utilities for leakage-safe VisionGuard model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class GroupSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def binary_class_indices(model: object) -> tuple[int, int]:
    """Return the probability-column indices for semantic labels 0 and 1."""
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.ndim != 1 or classes.size != 2 or set(classes.tolist()) != {0, 1}:
        raise ValueError(
            f"binary classifier classes must contain semantic labels 0 and 1; got {classes.tolist()}"
        )
    clean = int(np.flatnonzero(classes == 0)[0])
    attack = int(np.flatnonzero(classes == 1)[0])
    return clean, attack


def attack_probabilities(model: object, matrix: object) -> np.ndarray:
    """Predict probabilities and select the column whose class label is attack (1)."""
    _, attack_index = binary_class_indices(model)
    probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("binary classifier returned an invalid probability matrix")
    return probabilities[:, attack_index]


def group_aware_split_indices(
    groups: np.ndarray,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> GroupSplit:
    """Create deterministic train/validation/test indices without group overlap."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.size < 3:
        raise ValueError("groups must be a one-dimensional array with at least 3 rows")
    unique_groups = np.unique(groups)
    if unique_groups.size < 3:
        raise ValueError("at least three unique groups are required")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("validation_fraction must fit within the non-training fraction")

    all_indices = np.arange(groups.size)
    first = GroupShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
    train_indices, temporary_indices = next(first.split(all_indices, groups=groups))

    temporary_groups = groups[temporary_indices]
    validation_share = round(validation_fraction / (1.0 - train_fraction), 12)
    second = GroupShuffleSplit(
        n_splits=1,
        train_size=validation_share,
        random_state=seed + 1,
    )
    validation_relative, test_relative = next(
        second.split(temporary_indices, groups=temporary_groups)
    )
    validation_indices = temporary_indices[validation_relative]
    test_indices = temporary_indices[test_relative]

    split = GroupSplit(
        train=np.sort(train_indices),
        validation=np.sort(validation_indices),
        test=np.sort(test_indices),
    )
    assert_no_group_overlap(groups, split)
    return split


def assert_no_group_overlap(groups: np.ndarray, split: GroupSplit) -> None:
    group_sets = [set(np.asarray(groups)[indices]) for indices in (split.train, split.validation, split.test)]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise ValueError("group leakage detected between dataset splits")
