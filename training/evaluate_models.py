"""Generate real smoke-test evaluation plots and feature-domain ablations."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "raw" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from core.feature_extractor import extract_features, get_feature_names
from training.generate_perturbations import ATTACK_TYPES
from training.train_models import select_threshold
from training.utils import attack_probabilities


LOGGER = logging.getLogger("visionguard.evaluate_models")
MODELS_DIR = PROJECT_ROOT / "models"
GENERATED_DIR = PROJECT_ROOT / "static" / "generated"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "fused_features.csv"
SMOOTHED_FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "fused_features_smoothed.csv"
METRICS_PATH = MODELS_DIR / "model_metrics.csv"
PREDICTIONS_PATH = MODELS_DIR / "test_predictions.csv"
SPLITS_PATH = MODELS_DIR / "split_assignments.csv"
METADATA_COLUMNS = ["label", "base_id", "attack_type", "epsilon", "filepath"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate actual model evaluation and ablation artifacts."
    )
    parser.add_argument(
        "--skip-smoothing-ablation",
        action="store_true",
        help="run A-D ablations but skip the optional smoothed E variant",
    )
    return parser


def _save_figure(fig: plt.Figure, filename: str) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(GENERATED_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _predicted_labels(frame: pd.DataFrame) -> np.ndarray:
    return (frame["attack_probability"].to_numpy() >= frame["threshold"].to_numpy()).astype(int)


def epsilon_detection_table(attacked_predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute detection rates in fixed [1,2), [2,3), [3,4), [4,5] bins."""
    bins = [1.0, 2.0, 3.0, 4.0, 5.0000001]
    labels = ["1-2", "2-3", "3-4", "4-5"]
    work = attacked_predictions.copy()
    work["epsilon_bin"] = pd.cut(
        work["epsilon"], bins=bins, labels=labels, right=False, include_lowest=True
    )
    work["detected"] = _predicted_labels(work)
    grouped = work.groupby("epsilon_bin", observed=False)["detected"].agg(
        sample_count="count", detected_count="sum", detection_rate="mean"
    )
    return grouped.reindex(labels).reset_index()


def attack_detection_table(attacked_predictions: pd.DataFrame) -> pd.DataFrame:
    work = attacked_predictions.copy()
    work["detected"] = _predicted_labels(work)
    grouped = work.groupby("attack_type")["detected"].agg(
        sample_count="count", detected_count="sum", detection_rate="mean"
    )
    return grouped.reindex(ATTACK_TYPES).rename_axis("attack_type").reset_index()


def _model_comparison(metrics: pd.DataFrame) -> None:
    metrics.to_csv(GENERATED_DIR / "model_comparison.csv", index=False)
    display_columns = ["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"]
    display = metrics[display_columns].copy()
    for column in display_columns[1:]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    fig, axis = plt.subplots(figsize=(10, 2.8))
    axis.axis("off")
    table = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    axis.set_title("VisionGuard Model Comparison — Smoke Test", pad=16)
    _save_figure(fig, "model_comparison.png")


def _confusion_matrices(predictions: pd.DataFrame) -> None:
    for model_name, group in predictions.groupby("model_name", sort=False):
        matrix = confusion_matrix(group["label"], _predicted_labels(group), labels=[0, 1])
        fig, axis = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(matrix, display_labels=["Clean", "Manipulated"]).plot(
            ax=axis, cmap="Blues", colorbar=False
        )
        axis.set_title(f"{model_name} — Smoke-Test Confusion Matrix")
        safe_name = model_name.lower().replace(" ", "_")
        _save_figure(fig, f"confusion_matrix_{safe_name}.png")


def _roc_and_precision_recall(predictions: pd.DataFrame) -> None:
    roc_fig, roc_axis = plt.subplots(figsize=(7, 5))
    pr_fig, pr_axis = plt.subplots(figsize=(7, 5))
    for model_name, group in predictions.groupby("model_name", sort=False):
        labels = group["label"].to_numpy()
        probabilities = group["attack_probability"].to_numpy()
        if np.unique(labels).size < 2:
            LOGGER.warning("Skipping %s curves: held-out labels contain one class", model_name)
            continue
        false_positive, true_positive, _ = roc_curve(labels, probabilities)
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        roc_axis.plot(false_positive, true_positive, label=f"{model_name} ({auc(false_positive, true_positive):.3f})")
        pr_axis.plot(recall, precision, label=f"{model_name} ({auc(recall, precision):.3f})")
    roc_axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    roc_axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves — Smoke Test")
    roc_axis.legend(fontsize=8)
    roc_axis.grid(alpha=0.2)
    pr_axis.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curves — Smoke Test")
    pr_axis.legend(fontsize=8)
    pr_axis.grid(alpha=0.2)
    _save_figure(roc_fig, "roc_curves.png")
    _save_figure(pr_fig, "precision_recall_curves.png")


def _f1_plot(metrics: pd.DataFrame) -> None:
    ordered = metrics.sort_values("f1", ascending=True)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.barh(ordered["model_name"], ordered["f1"], color="#3977d5")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set(xlim=(0, 1), xlabel="Held-out F1", title="F1 Comparison — Smoke Test")
    axis.grid(axis="x", alpha=0.2)
    _save_figure(fig, "f1_comparison.png")


def _breakdown_artifacts(predictions: pd.DataFrame, best_model: str) -> None:
    best = predictions[predictions["model_name"] == best_model]
    attacked = best[best["label"] == 1]
    attack_table = attack_detection_table(attacked)
    epsilon_table = epsilon_detection_table(attacked)
    attack_table.to_csv(GENERATED_DIR / "attack_type_detection_rates.csv", index=False)
    epsilon_table.to_csv(GENERATED_DIR / "epsilon_detection_rates.csv", index=False)

    present_attacks = attack_table[attack_table["sample_count"].fillna(0) > 0]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(present_attacks["attack_type"], present_attacks["detection_rate"], color="#dc5a5a")
    axis.set(ylim=(0, 1), ylabel="Detection rate", title=f"Attack-Type Detection — {best_model} Smoke Test")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.2)
    _save_figure(fig, "attack_type_detection_rates.png")

    present_bins = epsilon_table[epsilon_table["sample_count"].fillna(0) > 0]
    fig, axis = plt.subplots(figsize=(6, 4.5))
    axis.bar(present_bins["epsilon_bin"].astype(str), present_bins["detection_rate"], color="#815ac7")
    axis.set(ylim=(0, 1), xlabel="Epsilon bin", ylabel="Detection rate", title=f"Epsilon-Level Detection — {best_model} Smoke Test")
    axis.grid(axis="y", alpha=0.2)
    _save_figure(fig, "epsilon_detection_rates.png")


def _smoothed_feature_frame(baseline: pd.DataFrame) -> pd.DataFrame:
    feature_names = get_feature_names()
    expected_columns = METADATA_COLUMNS + feature_names
    if SMOOTHED_FEATURES_PATH.is_file():
        cached = pd.read_csv(SMOOTHED_FEATURES_PATH)
        if list(cached.columns) == expected_columns and set(cached["filepath"]) == set(baseline["filepath"]):
            LOGGER.info("Reusing %s", SMOOTHED_FEATURES_PATH)
            return cached
        LOGGER.warning("Ignoring incompatible smoothed-feature cache")

    rows: list[dict[str, object]] = []
    for index, sample in tqdm(
        baseline.iterrows(), total=len(baseline), desc="Smoothed ablation features"
    ):
        features = extract_features(
            str(sample["filepath"]),
            {"enabled": True, "views": 5, "sigma": 1.0, "seed": 42 + int(index)},
        )
        row = {column: sample[column] for column in METADATA_COLUMNS}
        row.update(features)
        rows.append(row)
    smoothed = pd.DataFrame(rows, columns=expected_columns)
    if not np.isfinite(smoothed[feature_names].to_numpy(dtype=float)).all():
        raise ValueError("smoothed ablation features contain non-finite values")
    SMOOTHED_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    smoothed.to_csv(SMOOTHED_FEATURES_PATH, index=False)
    return smoothed


def _score_ablation(
    name: str,
    feature_frame: pd.DataFrame,
    columns: list[str],
    assignments: pd.DataFrame,
    base_model: object,
) -> dict[str, object]:
    merged = assignments[["filepath", "split"]].merge(feature_frame, on="filepath", how="left", validate="one_to_one")
    if merged[columns].isna().any().any():
        raise ValueError(f"ablation {name} has missing feature values")
    train = merged["split"] == "train"
    validation = merged["split"] == "validation"
    test = merged["split"] == "test"
    model = clone(base_model)
    model.fit(merged.loc[train, columns], merged.loc[train, "label"])
    validation_probabilities = attack_probabilities(model, merged.loc[validation, columns])
    threshold = select_threshold(merged.loc[validation, "label"].to_numpy(), validation_probabilities)
    probabilities = attack_probabilities(model, merged.loc[test, columns])
    labels = merged.loc[test, "label"].to_numpy()
    predicted = (probabilities >= threshold).astype(int)
    return {
        "variant": name,
        "feature_count": len(columns),
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "f1": f1_score(labels, predicted, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
    }


def _ablation_artifacts(skip_smoothing: bool) -> None:
    baseline = pd.read_csv(FEATURES_PATH)
    assignments = pd.read_csv(SPLITS_PATH)
    feature_names = get_feature_names()
    statistical = [name for name in feature_names if name.startswith("stat_")]
    frequency = [name for name in feature_names if name.startswith(("dct_", "fft_")) or name == "spectral_entropy"]
    texture = [name for name in feature_names if name.startswith(("lbp_", "glcm_"))]
    variants = [
        ("A Statistical only", baseline, statistical),
        ("B Statistical + Frequency", baseline, statistical + frequency),
        ("C Statistical + Frequency + Texture", baseline, statistical + frequency + texture),
        ("D All features", baseline, feature_names),
    ]
    if not skip_smoothing:
        variants.append(("E All features + randomized smoothing", _smoothed_feature_frame(baseline), feature_names))
    base_model = joblib.load(MODELS_DIR / "best_model.pkl")
    rows = [_score_ablation(name, frame, columns, assignments, base_model) for name, frame, columns in variants]
    results = pd.DataFrame(rows)
    results.to_csv(GENERATED_DIR / "ablation_results.csv", index=False)
    fig, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.barh(results["variant"], results["f1"], color="#37a878")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set(xlim=(0, 1), xlabel="Held-out F1", title="Feature-Domain Ablation — Smoke Test")
    axis.grid(axis="x", alpha=0.2)
    _save_figure(fig, "ablation_f1.png")


def evaluate(skip_smoothing: bool = False) -> int:
    required = [METRICS_PATH, PREDICTIONS_PATH, SPLITS_PATH, FEATURES_PATH, MODELS_DIR / "best_model.pkl", MODELS_DIR / "model_metadata.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required artifacts missing: {missing}")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    best_model = str(metadata["model_name"])
    _model_comparison(metrics)
    _confusion_matrices(predictions)
    _roc_and_precision_recall(predictions)
    _f1_plot(metrics)
    _breakdown_artifacts(predictions, best_model)
    _ablation_artifacts(skip_smoothing)
    return sum(
        path.is_file() and path.name != ".gitkeep"
        for path in GENERATED_DIR.iterdir()
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        count = evaluate(args.skip_smoothing_ablation)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        return 1
    print(f"Generated artifacts: {count}")
    print(f"Output directory: {GENERATED_DIR}")
    print("Scope: engineering smoke-test evaluation, not final project results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
