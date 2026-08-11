"""Tests for Chapter 7 detection breakdown calculations."""

import pandas as pd

from training.evaluate_models import attack_detection_table, epsilon_detection_table


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "attack_type": ["gaussian", "gaussian", "dct", "fft"],
            "epsilon": [1.5, 2.5, 3.5, 4.5],
            "attack_probability": [0.8, 0.2, 0.7, 0.1],
            "threshold": [0.5, 0.5, 0.5, 0.5],
        }
    )


def test_attack_detection_rates_use_stored_threshold() -> None:
    table = attack_detection_table(_predictions()).set_index("attack_type")
    assert table.loc["gaussian", "sample_count"] == 2
    assert table.loc["gaussian", "detection_rate"] == 0.5
    assert table.loc["dct", "detection_rate"] == 1.0


def test_epsilon_bins_are_fixed_and_complete() -> None:
    table = epsilon_detection_table(_predictions())
    assert table["epsilon_bin"].astype(str).tolist() == ["1-2", "2-3", "3-4", "4-5"]
    assert table["sample_count"].tolist() == [1, 1, 1, 1]
    assert table["detection_rate"].tolist() == [1.0, 0.0, 1.0, 0.0]
