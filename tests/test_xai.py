"""Tests for safe local model explanations."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.xai import XAIExplainer, feature_domain


def test_feature_domain_mapping() -> None:
    assert feature_domain("stat_mean") == "Statistical"
    assert feature_domain("dct_high_ratio") == "DCT"
    assert feature_domain("fft_high_ratio") == "FFT/Spectral"
    assert feature_domain("spectral_entropy") == "FFT/Spectral"
    assert feature_domain("lbp_uniform_bin_00") == "Texture"
    assert feature_domain("glcm_energy_mean") == "Texture"
    assert feature_domain("residual_energy") == "Residual"


def test_best_model_returns_top_five_finite_contributions(tmp_path: Path) -> None:
    model = joblib.load("models/best_model.pkl")
    names = joblib.load("models/feature_columns.pkl")
    frame = pd.read_csv("data/features/fused_features.csv")
    values = frame.loc[0, names].to_numpy(dtype=float)
    plot_path = tmp_path / "contributions.png"

    result = XAIExplainer(model, names).explain(values, plot_path=plot_path)

    assert result["xai_available"]
    assert result["method"] == "tree_shap"
    assert len(result["top_features"]) == 5
    assert plot_path.is_file() and plot_path.stat().st_size > 0
    assert "not causal" in result["explanation"]
    for item in result["top_features"]:
        assert set(item) == {
            "feature_name",
            "feature_value",
            "contribution",
            "feature_domain",
        }
        assert np.isfinite(item["feature_value"])
        assert np.isfinite(item["contribution"])


def test_explanation_failure_is_isolated() -> None:
    class BrokenModel:
        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            raise RuntimeError("expected test failure")

    result = XAIExplainer(BrokenModel(), ["stat_mean"]).explain([1.0])
    assert not result["xai_available"]
    assert result["top_features"] == []
    assert "prediction is still valid" in result["explanation"]
