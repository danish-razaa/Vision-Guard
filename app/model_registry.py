"""Validated registry of selectable standalone VisionGuard models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from app.inference import InferenceConfigurationError, VisionGuardInference
from config import BASE_DIR


LOGGER = logging.getLogger("visionguard.models")
DEFAULT_MODEL_ID = "production"


@dataclass
class SelectableModel:
    """A validated artifact description that loads its service on first use."""

    model_path: Path | None = None
    columns_path: Path | None = None
    metadata_path: Path | None = None
    threshold: float | None = None
    model_name: str | None = None
    _service: VisionGuardInference | None = None
    _lock: Lock = field(default_factory=Lock)

    def load(self) -> VisionGuardInference:
        if self._service is None:
            with self._lock:
                if self._service is None:
                    self._service = VisionGuardInference(
                        model_path=self.model_path,
                        feature_columns_path=self.columns_path,
                        metadata_path=self.metadata_path,
                        threshold_override=self.threshold,
                        model_name_override=self.model_name,
                    )
        return self._service


def initialize_model_registry(
    production: VisionGuardInference,
) -> tuple[dict[str, SelectableModel], list[dict[str, str]]]:
    """Register valid artifact paths; large candidates load only when selected."""
    services = {DEFAULT_MODEL_ID: SelectableModel(_service=production)}
    options = [{
        "id": DEFAULT_MODEL_ID,
        "label": f"{production.metadata.get('model_name', 'Model')} (Production)",
    }]
    models_root = BASE_DIR / "models"
    results_path = models_root / "candidate_all_model_results.json"
    metadata_path = models_root / "candidate_model_metadata.json"
    columns_path = models_root / "candidate_feature_columns.pkl"
    if not all(path.is_file() for path in (results_path, metadata_path, columns_path)):
        return services, options

    results = json.loads(results_path.read_text(encoding="utf-8"))
    candidates = (
        ("development_randomforest", "RandomForest", "randomforest.pkl"),
        ("development_xgboost", "XGBoost", "xgboost.pkl"),
        ("development_lightgbm", "LightGBM", "lightgbm.pkl"),
        ("development_catboost", "CatBoost", "catboost.pkl"),
        ("development_svm", "SVM", "svm.pkl"),
    )
    for model_id, model_name, filename in candidates:
        try:
            model_path = models_root / "development_candidates" / filename
            threshold = float(results[model_name]["threshold"])
            if not model_path.is_file() or not 0 <= threshold <= 1:
                raise ValueError("artifact or threshold is invalid")
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("Selectable model %s unavailable: %s", model_name, exc)
            continue
        services[model_id] = SelectableModel(
            model_path=model_path,
            columns_path=columns_path,
            metadata_path=metadata_path,
            threshold=threshold,
            model_name=f"{model_name} (Development)",
        )
        options.append({"id": model_id, "label": f"{model_name} (Development)"})
    return services, options


def selected_model(
    services: dict[str, SelectableModel], model_id: str | None
) -> tuple[str, VisionGuardInference]:
    """Resolve a client selection without accepting arbitrary artifact paths."""
    selected = (model_id or DEFAULT_MODEL_ID).strip()
    if selected not in services:
        raise ValueError("Select one of the available VisionGuard models.")
    try:
        return selected, services[selected].load()
    except InferenceConfigurationError as exc:
        raise ValueError("The selected model artifact could not be loaded.") from exc
