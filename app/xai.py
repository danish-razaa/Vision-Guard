"""Non-causal local explanations for VisionGuard predictions."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from training.utils import binary_class_indices


LOGGER = logging.getLogger("visionguard.xai")


def feature_domain(feature_name: str) -> str:
    """Map a canonical feature name to its scientific feature family."""
    if feature_name.startswith("stat_"):
        return "Statistical"
    if feature_name.startswith("dct_"):
        return "DCT"
    if feature_name.startswith("fft_") or feature_name == "spectral_entropy":
        return "FFT/Spectral"
    if feature_name.startswith(("lbp_", "glcm_")):
        return "Texture"
    if feature_name.startswith("residual_"):
        return "Residual"
    if feature_name.startswith(("msres_", "directional_", "color_residual_")):
        return "Residual"
    if feature_name.startswith("spectral_peak_"):
        return "FFT/Spectral"
    return "Unknown"


def _positive_class_probability(
    model: Any, values: np.ndarray, feature_names: Sequence[str] | None = None
) -> np.ndarray:
    model_input: Any = values
    if feature_names is not None:
        model_input = pd.DataFrame(values, columns=list(feature_names))
    probabilities = np.asarray(model.predict_proba(model_input), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("model predict_proba must return probabilities for two classes")
    _, attack_index = binary_class_indices(model)
    return probabilities[:, attack_index]


def _normalize_shap_values(raw_values: Any, feature_count: int) -> np.ndarray:
    """Normalize SHAP's model-dependent binary output shapes to one vector."""
    if isinstance(raw_values, list):
        raw_values = raw_values[-1]
    values = np.asarray(raw_values, dtype=float)
    if values.ndim == 1:
        vector = values
    elif values.ndim == 2:
        vector = values[0]
    elif values.ndim == 3 and values.shape[0] == 1:
        # Newer SHAP versions may return (samples, features, classes)
        vector = values[0, :, -1]
    elif values.ndim == 3 and values.shape[1] == 1:
        # Older multi-output layout: (classes, samples, features)
        vector = values[-1, 0, :]
    else:
        raise ValueError(f"unsupported SHAP output shape: {values.shape}")
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.size != feature_count or not np.all(np.isfinite(vector)):
        raise ValueError("SHAP returned an incomplete or non-finite contribution vector")
    return vector


class XAIExplainer:
    """Reusable explainer initialized once alongside the trained model.

    Tree SHAP is preferred. If the selected estimator is unsupported, the
    fallback measures the change in suspicious-class probability when each
    feature is replaced by a reference value. This is a local sensitivity
    approximation, not a causal attribution.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        baseline: Sequence[float] | None = None,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and unique")
        if baseline is None:
            self.baseline = np.zeros(len(self.feature_names), dtype=float)
        else:
            self.baseline = np.asarray(baseline, dtype=float).reshape(-1)
            if self.baseline.size != len(self.feature_names):
                raise ValueError("baseline length must match feature_names")
            if not np.all(np.isfinite(self.baseline)):
                raise ValueError("baseline must contain finite values")

        self._shap_explainer: Any | None = None
        self.method = "probability_occlusion"
        self._last_method = self.method
        try:
            matplotlib_cache = Path(__file__).resolve().parents[1] / "data" / "raw" / "matplotlib"
            matplotlib_cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
            import shap

            self._shap_explainer = shap.TreeExplainer(model)
            self.method = "tree_shap"
            self._last_method = self.method
        except Exception as exc:
            LOGGER.info("Tree SHAP unavailable; using probability occlusion: %s", exc)

    def _contributions(self, row: np.ndarray) -> np.ndarray:
        if self._shap_explainer is not None:
            try:
                raw = self._shap_explainer.shap_values(row.reshape(1, -1))
                contributions = _normalize_shap_values(raw, len(self.feature_names))
                self._last_method = "tree_shap"
                return contributions
            except Exception as exc:
                LOGGER.warning("Tree SHAP failed; using probability occlusion: %s", exc)

        original_probability = float(
            _positive_class_probability(self.model, row.reshape(1, -1), self.feature_names)[0]
        )
        perturbed = np.repeat(row.reshape(1, -1), len(row), axis=0)
        indices = np.arange(len(row))
        perturbed[indices, indices] = self.baseline
        replaced_probabilities = _positive_class_probability(
            self.model, perturbed, self.feature_names
        )
        contributions = original_probability - replaced_probabilities
        if not np.all(np.isfinite(contributions)):
            raise ValueError("fallback produced non-finite contributions")
        self._last_method = "probability_occlusion"
        return contributions

    def explain(
        self,
        feature_values: Sequence[float] | np.ndarray,
        top_k: int = 5,
        plot_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Return a safe top-feature explanation; errors become a fallback result."""
        try:
            row = np.asarray(feature_values, dtype=float).reshape(-1)
            if row.size != len(self.feature_names):
                raise ValueError("feature value count does not match saved feature names")
            if not np.all(np.isfinite(row)):
                raise ValueError("feature values must be finite")
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
                raise ValueError("top_k must be a positive integer")
            contributions = self._contributions(row)
            limit = min(top_k, len(row))
            order = np.argsort(-np.abs(contributions), kind="stable")[:limit]
            top_features = [
                {
                    "feature_name": self.feature_names[index],
                    "feature_value": float(row[index]),
                    "contribution": float(contributions[index]),
                    "feature_domain": feature_domain(self.feature_names[index]),
                }
                for index in order
            ]
            domains = list(dict.fromkeys(item["feature_domain"] for item in top_features))
            explanation = (
                "These features contributed most strongly to the model prediction. "
                f"The leading signals came from {', '.join(domains)} domains. "
                "Positive contributions support the suspicious class and negative "
                "contributions support the clean class; these are model attributions, "
                "not causal conclusions."
            )
            written_plot: str | None = None
            if plot_path is not None:
                try:
                    written_plot = self._write_plot(top_features, Path(plot_path))
                except Exception as exc:
                    LOGGER.warning("Contribution plot generation failed: %s", exc)
            return {
                "top_features": top_features,
                "explanation": explanation,
                "method": self._last_method,
                "plot_path": written_plot,
                "xai_available": True,
            }
        except Exception as exc:
            LOGGER.exception("XAI explanation failed")
            return {
                "top_features": [],
                "explanation": "Feature attribution is temporarily unavailable; the model prediction is still valid.",
                "method": "unavailable",
                "plot_path": None,
                "xai_available": False,
                "error": str(exc),
            }

    @staticmethod
    def _write_plot(top_features: list[dict[str, Any]], path: Path) -> str:
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        labels = [item["feature_name"] for item in reversed(top_features)]
        values = [item["contribution"] for item in reversed(top_features)]
        colors = ["#e15d64" if value >= 0 else "#3c82d4" for value in values]
        fig, axis = plt.subplots(figsize=(7, 3.6))
        axis.barh(labels, values, color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Model contribution")
        axis.set_title("Top Feature Contributions")
        axis.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path.resolve())


def explain_prediction(
    model: Any,
    feature_values: Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
    baseline: Sequence[float] | None = None,
    plot_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience wrapper; applications should reuse ``XAIExplainer`` instead."""
    try:
        explainer = XAIExplainer(model, feature_names, baseline=baseline)
        return explainer.explain(feature_values, top_k=5, plot_path=plot_path)
    except Exception as exc:
        LOGGER.exception("XAI initialization failed")
        return {
            "top_features": [],
            "explanation": "Feature attribution is temporarily unavailable; the model prediction is still valid.",
            "method": "unavailable",
            "plot_path": None,
            "xai_available": False,
            "error": str(exc),
        }
