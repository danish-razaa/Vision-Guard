"""Reference-aware inference for Prompt Injection Lab image pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from config import BASE_DIR


class PairDetectorConfigurationError(RuntimeError):
    pass


class PromptPairDetector:
    """Predict controlled change from signed and absolute feature deltas."""

    def __init__(self, model_path: Path = BASE_DIR/"models"/"pair_detector.pkl", columns_path: Path = BASE_DIR/"models"/"pair_feature_columns.pkl", metadata_path: Path = BASE_DIR/"models"/"pair_detector_metadata.json") -> None:
        try:
            self.model = joblib.load(model_path)
            self.columns = list(joblib.load(columns_path))
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.threshold = float(self.metadata["threshold"])
            self.attack_index = list(self.model.classes_).index(1)
        except Exception as exc:
            raise PairDetectorConfigurationError("pair detector artifacts are unavailable or inconsistent") from exc

    def predict(self, original: Mapping[str, float], modified: Mapping[str, float]) -> dict[str, object]:
        feature_names = [name.removeprefix("signed_delta__") for name in self.columns if name.startswith("signed_delta__")]
        if not feature_names or any(name not in original or name not in modified for name in feature_names):
            raise ValueError("pair detector input is missing required image features")
        signed = np.array([float(modified[name])-float(original[name]) for name in feature_names], dtype=float)
        values = np.concatenate((signed, np.abs(signed)))
        frame = pd.DataFrame([values], columns=self.columns)
        probability = float(self.model.predict_proba(frame)[0, self.attack_index])
        detected = probability >= self.threshold
        return {"status": "CONTROLLED PERTURBATION DETECTED" if detected else "NO CONTROLLED CHANGE DETECTED", "detected": detected, "probability": probability*100.0, "threshold": self.threshold*100.0, "model_name": self.metadata.get("model_name", type(self.model).__name__), "scope": "Reference-aware pair analysis; requires the original image and is not a standalone scanner result."}
