"""Evaluate a saved detector by source and perturbation cohort without fitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_extractor import extract_features, get_feature_names
from training.utils import attack_probabilities

DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "generalization_evaluation.json"


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    """Calculate full binary metrics; ROC-AUC is null for single-class cohorts."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "sample_count": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "mean_attack_probability": float(np.mean(probabilities)),
        "confusion_matrix": matrix.tolist(),
    }


def _load_manifest(path: Path, cohort: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"filepath", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["cohort"] = frame.get("cohort", cohort)
    frame["source_dataset"] = frame.get("source_dataset", "unknown")
    frame["attack_type"] = frame.get("attack_type", np.where(frame["label"].astype(int) == 0, "clean", "unknown"))
    frame["epsilon"] = pd.to_numeric(frame.get("epsilon", 0.0), errors="coerce")
    return frame


def evaluate_manifests(manifests: list[tuple[str, Path]], output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    metadata = json.loads((PROJECT_ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(PROJECT_ROOT / "models" / "best_model.pkl")
    columns = list(joblib.load(PROJECT_ROOT / "models" / "feature_columns.pkl"))
    if columns != get_feature_names():
        raise ValueError("saved feature columns differ from the canonical extractor")
    frames = [_load_manifest(path.resolve(), cohort) for cohort, path in manifests]
    samples = pd.concat(frames, ignore_index=True)
    feature_rows = []
    for filepath in samples["filepath"].astype(str):
        feature_rows.append(extract_features(Path(filepath).expanduser().resolve()))
    matrix = pd.DataFrame(feature_rows, columns=columns, dtype=np.float64)
    probabilities = attack_probabilities(model, matrix)
    samples["attack_probability"] = probabilities
    threshold = float(metadata["threshold"])
    report: dict[str, object] = {"model_name": metadata.get("model_name"), "threshold": threshold, "cohorts": {}, "by_source_dataset": {}, "by_attack_type": {}, "by_epsilon_range": {}}
    for field, key in (("cohort", "cohorts"), ("source_dataset", "by_source_dataset"), ("attack_type", "by_attack_type")):
        target = report[key]
        assert isinstance(target, dict)
        for name, group in samples.groupby(field, dropna=False):
            target[str(name)] = calculate_metrics(group["label"].to_numpy(), group["attack_probability"].to_numpy(), threshold)
    attacked = samples[samples["label"].astype(int) == 1].copy()
    attacked["epsilon_range"] = pd.cut(attacked["epsilon"], [-np.inf, 1, 2, 4, np.inf], labels=["very-low", "low", "medium", "high"])
    for name, group in attacked.groupby("epsilon_range", observed=False):
        if len(group):
            report["by_epsilon_range"][str(name)] = calculate_metrics(group["label"].to_numpy(), group["attack_probability"].to_numpy(), threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate current VisionGuard generalization without retraining")
    parser.add_argument("--manifest", action="append", nargs=2, metavar=("COHORT", "CSV"), required=True, help="CSV with filepath,label and optional source_dataset,attack_type,epsilon")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate_manifests([(name, Path(path)) for name, path in args.manifest], args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
