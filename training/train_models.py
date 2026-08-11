"""Train and select five classifiers with leakage-safe grouped splits."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from core.feature_extractor import get_feature_names
from training.utils import GroupSplit, attack_probabilities, binary_class_indices, group_aware_split_indices


LOGGER = logging.getLogger("visionguard.train_models")
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "features" / "fused_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
CANDIDATES_DIR = MODELS_DIR / "candidates"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train five group-aware VisionGuard smoke-test classifiers."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _model_definitions(seed: int) -> dict[str, object]:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        probability=True,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a validation threshold by F1, then recall, then proximity to 0.5."""
    candidates = np.unique(np.concatenate(([0.5], probabilities)))
    candidates = candidates[(candidates >= 0.0) & (candidates <= 1.0)]
    scored: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        scored.append(
            (
                float(f1_score(y_true, predictions, zero_division=0)),
                float(recall_score(y_true, predictions, zero_division=0)),
                -abs(float(threshold) - 0.5),
                float(threshold),
            )
        )
    return max(scored)[3]


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost")
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _save_split_assignments(frame: pd.DataFrame, split: GroupSplit) -> None:
    assignments = frame[["filepath", "base_id", "label", "attack_type", "epsilon"]].copy()
    assignments["split"] = ""
    assignments.loc[split.train, "split"] = "train"
    assignments.loc[split.validation, "split"] = "validation"
    assignments.loc[split.test, "split"] = "test"
    assignments.to_csv(MODELS_DIR / "split_assignments.csv", index=False)


def train_models(features_path: Path, seed: int) -> str:
    frame = pd.read_csv(features_path)
    feature_columns = get_feature_names()
    required = {"label", "base_id", "filepath", *feature_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"feature dataset missing columns: {sorted(missing)}")
    matrix = frame[feature_columns].astype(np.float64)
    labels = frame["label"].to_numpy(dtype=int)
    if not np.all(np.isfinite(matrix.to_numpy())):
        raise ValueError("feature dataset contains NaN or infinite values")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("training requires both labels 0 and 1")

    split = group_aware_split_indices(frame["base_id"].astype(str).to_numpy(), seed=seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    _save_split_assignments(frame, split)

    metrics_rows: list[dict[str, object]] = []
    metrics_json: dict[str, object] = {}
    prediction_rows: list[dict[str, object]] = []
    trained_models: dict[str, object] = {}

    for model_name, model in _model_definitions(seed).items():
        LOGGER.info("Training %s", model_name)
        model.fit(matrix.iloc[split.train], labels[split.train])
        binary_class_indices(model)
        validation_probabilities = attack_probabilities(model, matrix.iloc[split.validation])
        threshold = select_threshold(labels[split.validation], validation_probabilities)
        validation_metrics = _metrics(labels[split.validation], validation_probabilities, threshold)
        test_probabilities = attack_probabilities(model, matrix.iloc[split.test])
        test_metrics = _metrics(labels[split.test], test_probabilities, threshold)

        metrics_rows.append(
            {
                "model_name": model_name,
                "threshold": threshold,
                **{f"validation_{key}": value for key, value in validation_metrics.items() if key != "confusion_matrix"},
                **{key: value for key, value in test_metrics.items() if key != "confusion_matrix"},
            }
        )
        metrics_json[model_name] = {
            "threshold": threshold,
            "validation": validation_metrics,
            "test": test_metrics,
        }
        for row_index, probability in zip(split.test, test_probabilities, strict=True):
            prediction_rows.append(
                {
                    "model_name": model_name,
                    "filepath": frame.iloc[row_index]["filepath"],
                    "base_id": frame.iloc[row_index]["base_id"],
                    "attack_type": frame.iloc[row_index]["attack_type"],
                    "epsilon": frame.iloc[row_index]["epsilon"],
                    "label": int(labels[row_index]),
                    "attack_probability": float(probability),
                    "threshold": threshold,
                }
            )
        trained_models[model_name] = model
        joblib.dump(model, CANDIDATES_DIR / f"{model_name.lower()}.pkl")

    ranked = sorted(
        metrics_rows,
        key=lambda row: (
            float(row["validation_f1"]),
            float(row["validation_roc_auc"]),
            float(row["validation_recall"]),
            float(row["validation_accuracy"]),
        ),
        reverse=True,
    )
    best_name = str(ranked[0]["model_name"])
    best_threshold = float(ranked[0]["threshold"])
    joblib.dump(trained_models[best_name], MODELS_DIR / "best_model.pkl")
    joblib.dump(feature_columns, MODELS_DIR / "feature_columns.pkl")

    pd.DataFrame(metrics_rows).to_csv(MODELS_DIR / "model_metrics.csv", index=False)
    (MODELS_DIR / "model_metrics.json").write_text(
        json.dumps(metrics_json, indent=2), encoding="utf-8"
    )
    pd.DataFrame(prediction_rows).to_csv(MODELS_DIR / "test_predictions.csv", index=False)

    best_details = metrics_json[best_name]
    metadata = {
        "model_name": best_name,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(feature_columns),
        "dataset_size": len(frame),
        "clean_count": int((labels == 0).sum()),
        "attack_count": int((labels == 1).sum()),
        "split_method": "GroupShuffleSplit 70/15/15 by base_id",
        "split_counts": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "metrics": best_details,
        "threshold": best_threshold,
        "software_versions": _package_versions(),
        "seed": seed,
        "class_semantics": {"0": "clean", "1": "attack"},
        "smoothing": {"enabled": False, "views": 1, "sigma": 0.0},
        "result_scope": "engineering smoke-test; not final project results",
    }
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    LOGGER.info("Selected %s with validation threshold %.6f", best_name, best_threshold)
    return best_name


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        best_name = train_models(args.features.resolve(), args.seed)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("Training failed: %s", exc)
        return 1
    print(f"Best model: {best_name}")
    print(f"Metrics: {MODELS_DIR / 'model_metrics.csv'}")
    print("Scope: engineering smoke-test results, not final project results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
