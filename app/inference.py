"""Reusable, artifact-backed VisionGuard inference pipeline."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.xai import XAIExplainer
from config import Config
from core.feature_extractor import extract_features, get_feature_names
from core.smoothing import SmoothingConfig
from training.utils import binary_class_indices


LOGGER = logging.getLogger("visionguard.inference")
TRAINING_METADATA_COLUMNS = {
    "label",
    "base_id",
    "attack_type",
    "epsilon",
    "filepath",
    "clean_path",
    "attacked_path",
}


class InferenceConfigurationError(RuntimeError):
    """Raised when saved model artifacts are missing or inconsistent."""


def risk_level(attack_probability: float) -> str:
    """Map a suspicious-class probability percentage to its display risk band."""
    if not np.isfinite(attack_probability) or not 0 <= attack_probability <= 100:
        raise ValueError("attack_probability must be a finite percentage from 0 to 100")
    if attack_probability < 30:
        return "LOW"
    if attack_probability < 60:
        return "MEDIUM"
    if attack_probability < 80:
        return "HIGH"
    return "CRITICAL"


class VisionGuardInference:
    """Load artifacts once and process any number of image predictions."""

    def __init__(
        self,
        model_path: str | Path = Config.MODEL_PATH,
        feature_columns_path: str | Path = Config.FEATURE_COLUMNS_PATH,
        metadata_path: str | Path = Config.MODEL_METADATA_PATH,
        smoothing_config: SmoothingConfig | None = None,
        threshold_override: float | None = None,
        model_name_override: str | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.feature_columns_path = Path(feature_columns_path).expanduser().resolve()
        self.metadata_path = Path(metadata_path).expanduser().resolve()
        for artifact in (self.model_path, self.feature_columns_path, self.metadata_path):
            if not artifact.is_file():
                raise InferenceConfigurationError(f"required model artifact is missing: {artifact}")

        try:
            self.model = joblib.load(self.model_path)
            self.feature_columns = list(joblib.load(self.feature_columns_path))
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise InferenceConfigurationError("could not load saved model artifacts") from exc

        canonical_columns = get_feature_names()
        if not self.feature_columns or len(self.feature_columns) != len(set(self.feature_columns)):
            raise InferenceConfigurationError(
                "saved feature columns must be non-empty and unique"
            )
        metadata_columns = TRAINING_METADATA_COLUMNS.intersection(self.feature_columns)
        if metadata_columns:
            raise InferenceConfigurationError(
                f"training metadata cannot be model input: {sorted(metadata_columns)}"
            )
        unknown_columns = set(self.feature_columns).difference(canonical_columns)
        if unknown_columns:
            raise InferenceConfigurationError(
                f"saved model requires unknown numerical features: {sorted(unknown_columns)}"
            )
        if int(self.metadata.get("feature_count", -1)) != len(self.feature_columns):
            raise InferenceConfigurationError("model metadata feature_count is inconsistent")
        try:
            self.threshold = float(
                self.metadata["threshold"]
                if threshold_override is None
                else threshold_override
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InferenceConfigurationError("model metadata has no valid threshold") from exc
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise InferenceConfigurationError("stored threshold must be between 0 and 1")
        if not hasattr(self.model, "predict_proba"):
            raise InferenceConfigurationError("saved model does not implement predict_proba")
        try:
            self.clean_class_index, self.attack_class_index = binary_class_indices(self.model)
        except ValueError as exc:
            raise InferenceConfigurationError(str(exc)) from exc

        self.smoothing_config = smoothing_config or SmoothingConfig(
            enabled=Config.SMOOTHING_ENABLED,
            views=Config.SMOOTHING_VIEWS,
            sigma=Config.SMOOTHING_SIGMA,
        )
        if model_name_override:
            self.metadata["model_name"] = model_name_override
        self.xai = XAIExplainer(self.model, self.feature_columns)
        LOGGER.info(
            "Loaded %s with %d features and threshold %.6f",
            self.metadata.get("model_name", type(self.model).__name__),
            len(self.feature_columns),
            self.threshold,
        )

    def predict(
        self,
        image_path: str | Path,
        xai_plot_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Produce a security decision and best-effort local explanation.

        Returned probabilities and threshold are percentages in the range
        0–100. The suspicious/clean decision uses the saved classification
        threshold; the display risk band uses fixed probability ranges.
        """
        features = extract_features(image_path, self.smoothing_config)
        missing_features = set(self.feature_columns).difference(features)
        if missing_features:
            raise InferenceConfigurationError(
                f"runtime extraction is missing saved features: {sorted(missing_features)}"
            )
        inference_frame = pd.DataFrame(
            [[features[name] for name in self.feature_columns]],
            columns=self.feature_columns,
            dtype=np.float64,
        )
        if list(inference_frame.columns) != self.feature_columns:
            raise InferenceConfigurationError("runtime feature order differs from saved feature order")
        vector = inference_frame.iloc[0].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            raise ValueError("runtime feature vector contains non-finite values")

        probabilities = np.asarray(self.model.predict_proba(inference_frame), dtype=float)
        if probabilities.shape != (1, 2) or not np.all(np.isfinite(probabilities)):
            raise ValueError("model returned invalid class probabilities")
        attack_probability_fraction = float(probabilities[0, self.attack_class_index])
        if not 0 <= attack_probability_fraction <= 1:
            raise ValueError("model returned a probability outside 0–1")
        attack_percentage = attack_probability_fraction * 100.0
        clean_percentage = float(probabilities[0, self.clean_class_index]) * 100.0
        threshold_percentage = self.threshold * 100.0
        status = (
            "SUSPICIOUS"
            if attack_probability_fraction >= self.threshold
            else "VERIFIED/CLEAN"
        )

        explanation = self.xai.explain(vector, top_k=5, plot_path=xai_plot_path)
        return {
            "status": status,
            "attack_probability": attack_percentage,
            "clean_probability": clean_percentage,
            "threshold": threshold_percentage,
            "risk_level": risk_level(attack_percentage),
            "top_features": explanation["top_features"],
            "explanation": explanation["explanation"],
            "xai_available": explanation["xai_available"],
            "xai_method": explanation["method"],
            "xai_plot_path": explanation["plot_path"],
            "model_name": self.metadata.get("model_name", type(self.model).__name__),
        }


@lru_cache(maxsize=1)
def get_inference_service() -> VisionGuardInference:
    """Return the process-wide inference service, loading artifacts only once."""
    return VisionGuardInference()


def initialize_inference() -> VisionGuardInference:
    """Explicit application-startup hook for the Chapter 10 app factory."""
    return get_inference_service()


def predict_image(
    image_path: str | Path,
    xai_plot_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience API backed by the process-wide cached service."""
    return get_inference_service().predict(image_path, xai_plot_path=xai_plot_path)
