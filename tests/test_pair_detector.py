"""Tests for the trained reference-aware Prompt Lab detector."""

from pathlib import Path

import pandas as pd

from app.pair_inference import PromptPairDetector
from core.feature_extractor import get_feature_names


ROOT = Path(__file__).resolve().parents[1]


def _known_detected_prompt_pair() -> tuple[dict[str, float], dict[str, float]]:
    metadata = pd.read_csv(ROOT/"data"/"metadata"/"development_dataset.csv")
    predictions = pd.read_csv(ROOT/"models"/"pair_test_predictions.csv")
    detected = predictions[(predictions["attack_type"] == "prompt_conditioned") & (predictions["predicted"] == 1)].iloc[0]
    group = metadata[metadata["base_id"] == detected["base_id"]]
    original_path = group[group["label"] == 0].iloc[0]["filepath"]
    modified_path = group[group["attack_type"] == "prompt_conditioned"].iloc[0]["filepath"]
    features = pd.read_csv(ROOT/"data"/"features"/"development_features.csv").set_index("filepath")
    names = get_feature_names()
    return features.loc[original_path, names].to_dict(), features.loc[modified_path, names].to_dict()


def test_pair_detector_does_not_flag_unchanged_reference() -> None:
    detector = PromptPairDetector()
    original, _ = _known_detected_prompt_pair()
    result = detector.predict(original, original)
    assert result["detected"] is False


def test_pair_detector_flags_held_out_prompt_pair_by_model_prediction() -> None:
    detector = PromptPairDetector()
    original, modified = _known_detected_prompt_pair()
    result = detector.predict(original, modified)
    assert result["detected"] is True
    assert result["probability"] >= result["threshold"]
