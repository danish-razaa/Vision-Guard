"""Tests for stable multi-domain feature extraction."""

from pathlib import Path

import cv2
import numpy as np

from core.feature_extractor import extract_features, get_feature_names


def _write_test_image(path: Path) -> None:
    yy, xx = np.mgrid[:80, :120]
    image = np.stack(
        ((xx * 2) % 256, (yy * 3) % 256, (xx + yy) % 256), axis=2
    ).astype(np.uint8)
    assert cv2.imwrite(str(path), image)


def test_feature_names_are_unique_and_stable() -> None:
    names = get_feature_names()
    assert len(names) == 65
    assert len(names) == len(set(names))
    assert names[:3] == ["stat_mean", "stat_variance", "stat_std"]
    assert names[-4:] == [
        "residual_mean_abs",
        "residual_std",
        "residual_energy",
        "residual_max_abs",
    ]


def test_extraction_is_finite_ordered_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    _write_test_image(path)
    first = extract_features(path, {"enabled": False})
    second = extract_features(path, {"enabled": False})

    assert list(first) == get_feature_names()
    assert len(first) == 65
    assert np.all(np.isfinite(list(first.values())))
    assert first == second


def test_seeded_smoothing_feature_extraction_is_reproducible(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    _write_test_image(path)
    config = {"enabled": True, "views": 3, "sigma": 0.5, "seed": 9}
    assert extract_features(path, config) == extract_features(path, config)
