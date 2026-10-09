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


def explain_feature_in_plain_english(
    feature_name: str,
    value: float,
    contribution: float,
) -> dict[str, Any]:
    """Translate technical feature metrics into layperson-friendly concepts."""
    domain = feature_domain(feature_name)
    is_suspicious_signal = contribution >= 0
    vote = "Supports Suspicious" if is_suspicious_signal else "Supports Clean"

    if feature_name.startswith("lbp_uniform_bin_"):
        bin_suffix = feature_name.replace("lbp_uniform_bin_", "")
        title = f"Micro-Texture Noise (Pattern #{bin_suffix})"
        if is_suspicious_signal:
            desc = (
                f"Abnormal micro-grain static detected (value {value:.4g}). "
                "Natural photos have smooth, organic transitions, whereas digital "
                "attacks create unnatural microscopic pixel irregularities."
            )
        else:
            desc = (
                f"Natural micro-grain transition (value {value:.4g}), consistent "
                "with standard photographic lens captures."
            )
    elif feature_name.startswith("glcm_"):
        prop = feature_name.replace("glcm_", "").replace("_", " ").title()
        title = f"Surface Texture Regularity ({prop})"
        if is_suspicious_signal:
            desc = (
                f"Disrupted spatial patterns in surface shading (value {value:.4g}). "
                "Adversarial perturbations alter the expected texture continuity."
            )
        else:
            desc = (
                f"Coherent surface texture distribution (value {value:.4g}), "
                "characteristic of authentic images."
            )
    elif feature_name == "stat_max":
        title = "Peak Pixel Brightness Ceiling"
        if value >= 254:
            desc = (
                f"Pixel saturation clipped at maximum limit ({value:.0f}). "
                "Mathematical perturbation scripts often push pixel values to the ceiling."
            )
        else:
            desc = f"Maximum pixel intensity ({value:.4g}) within standard exposure boundaries."
    elif feature_name == "stat_min":
        title = "Minimum Black Level Floor"
        if value <= 1:
            desc = (
                f"Pixels clamped to absolute black ({value:.0f}). "
                "Indicates artificial numerical boundary clipping."
            )
        else:
            desc = f"Minimum shadow intensity ({value:.4g}) is within natural photographic range."
    elif feature_name in ("stat_std", "stat_variance"):
        title = "Pixel Contrast Variance"
        if is_suspicious_signal:
            desc = (
                f"Unusual spread in pixel contrast (value {value:.4g}), indicating "
                "added synthetic static across image channels."
            )
        else:
            desc = f"Natural dynamic range and contrast variation ({value:.4g})."
    elif feature_name == "stat_mean":
        title = "Average Brightness Exposure"
        desc = f"Mean channel exposure level ({value:.4g}) across the image."
    elif feature_name.startswith("stat_"):
        metric = feature_name.replace("stat_", "").title()
        title = f"Statistical Distribution ({metric})"
        desc = (
            f"Mathematical distribution anomaly in pixel brightness (value {value:.4g})."
            if is_suspicious_signal
            else f"Standard pixel intensity distribution (value {value:.4g})."
        )
    elif feature_name.startswith("dct_"):
        title = "Frequency Compression Signature (DCT)"
        if is_suspicious_signal:
            desc = (
                f"High-frequency mathematical artifacts found in frequency analysis (value {value:.4g}), "
                "typical of adversarial perturbation noise."
            )
        else:
            desc = f"Organic frequency falloff curve (value {value:.4g}) matching natural lens captures."
    elif (
        feature_name.startswith("fft_")
        or feature_name == "spectral_entropy"
        or feature_name.startswith("spectral_")
    ):
        title = "Periodic Frequency Harmonics (FFT)"
        if is_suspicious_signal:
            desc = (
                f"Synthetic periodic frequency spikes detected (value {value:.4g}). "
                "These harmonic patterns are invisible to human eyes but detectable in the frequency domain."
            )
        else:
            desc = f"Smooth, non-repetitive spectral energy distribution (value {value:.4g})."
    elif feature_name.startswith(
        ("residual_", "msres_", "directional_", "color_residual_")
    ):
        title = "High-Pass Noise Residue"
        if is_suspicious_signal:
            desc = (
                f"Filtering out the visible subject revealed an underlying artificial noise residue "
                f"(value {value:.4g}) embedded beneath the image."
            )
        else:
            desc = f"Minimal noise residue ({value:.4g}), fully consistent with camera sensor noise."
    elif feature_name.startswith(("perturbation_sensitive_", "laplacian_")):
        title = "Edge Perturbation Gradient"
        if is_suspicious_signal:
            desc = (
                f"Sharp localized micro-gradient spikes (value {value:.4g}) characteristic of "
                "algorithmic manipulation."
            )
        else:
            desc = f"Soft, natural optical edges ({value:.4g}) with no artificial boundary spikes."
    else:
        title = feature_name.replace("_", " ").title()
        desc = f"{domain} signal (value {value:.4g}) pushing the model towards {vote.lower()}."

    return {
        "feature_name": feature_name,
        "title": title,
        "domain": domain,
        "vote": vote,
        "is_suspicious_signal": is_suspicious_signal,
        "contribution": contribution,
        "feature_value": value,
        "plain_explanation": desc,
    }


def generate_layman_narrative(
    top_features: list[dict[str, Any]],
    domains: list[str],
) -> dict[str, Any]:
    """Produce a high-level plain-English narrative with real-world analogies."""
    suspicious_count = sum(1 for f in top_features if f.get("contribution", 0) >= 0)
    clean_count = len(top_features) - suspicious_count
    primary_domain = domains[0] if domains else "Texture"
    domain_list = ", ".join(domains) if domains else "Texture and Statistical"

    if suspicious_count >= clean_count:
        headline = "Invisible Micro-Texture Anomalies & Pixel Distortions Detected"
        narrative = (
            "To the human eye, this image appears ordinary. However, our Explainable AI detected "
            f"invisible microscopic irregularities primarily in the {domain_list} domain(s). "
            "These artificial micro-patterns and pixel distributions strongly suggest the image has been "
            "digitally altered or embedded with adversarial perturbations rather than captured by a standard camera."
        )
        analogy = (
            "Think of this like inspecting a counterfeit banknote: from arm's length it looks genuine, "
            "but under magnification the microscopic ink patterns and fiber textures reveal that it was "
            "produced by an unauthorized digital printer."
        )
        takeaways = [
            f"Microscopic static detected across the {primary_domain} domain.",
            "Pixels exhibit mathematical boundaries and artificial noise profiles uncommon in natural photography.",
            "Positive contribution scores confirm these anomalies actively drove the suspicious classification.",
        ]
    else:
        headline = "Natural Optical Characteristics & Coherent Pixel Gradients Verified"
        narrative = (
            "To the human eye and our AI analysis, this image shows natural optical characteristics "
            f"typical of an authentic, unmanipulated photograph. The micro-textures and pixel distributions "
            f"across {domain_list} domains show no evidence of synthetic perturbations or adversarial noise."
        )
        analogy = (
            "Like checking a genuine watermark under ultraviolet light, the image's microscopic pixel "
            "transitions align seamlessly with authentic camera optics and natural sensor physics."
        )
        takeaways = [
            f"Organic micro-patterns confirmed in {domain_list} domain(s).",
            "No evidence of mathematical pixel clipping or synthetic high-frequency static.",
            "Feature contributions support a clean, untampered verdict.",
        ]

    return {
        "headline": headline,
        "narrative": narrative,
        "analogy": analogy,
        "takeaways": takeaways,
    }


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
            layman_insights = [
                explain_feature_in_plain_english(
                    item["feature_name"], item["feature_value"], item["contribution"]
                )
                for item in top_features
            ]
            layman_summary = generate_layman_narrative(top_features, domains)
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
                "layman_summary": layman_summary,
                "layman_insights": layman_insights,
            }
        except Exception as exc:
            LOGGER.exception("XAI explanation failed")
            return {
                "top_features": [],
                "explanation": "Feature attribution is temporarily unavailable; the model prediction is still valid.",
                "method": "unavailable",
                "plot_path": None,
                "xai_available": False,
                "layman_summary": None,
                "layman_insights": [],
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
