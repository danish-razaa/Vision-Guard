"""Train and evaluate candidate-only models on the corrected development set."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_extractor import extract_features, get_feature_names
from training.train_models import _model_definitions
from training.utils import attack_probabilities, assert_no_group_overlap, GroupSplit

MODELS = PROJECT_ROOT / "models"


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(labels, predicted)), "precision": float(precision_score(labels, predicted, zero_division=0)), "recall": float(recall_score(labels, predicted, zero_division=0)), "f1": float(f1_score(labels, predicted, zero_division=0)), "roc_auc": float(roc_auc_score(labels, probabilities)), "false_positive_rate": float(fp/(fp+tn)) if fp+tn else None, "false_negative_rate": float(fn/(fn+tp)) if fn+tp else None, "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def threshold_analysis(labels: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    candidates = np.unique(np.concatenate((np.linspace(0.05, 0.95, 91), probabilities)))
    rows = []
    for threshold in candidates:
        result = metrics(labels, probabilities, float(threshold))
        rows.append({"threshold": float(threshold), **{key: result[key] for key in ("precision", "recall", "f1", "false_positive_rate", "false_negative_rate")}})
    return pd.DataFrame(rows)


def external_matrix(columns: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "external_holdout_manifest.csv")
    rows = [extract_features(path) for path in manifest["filepath"].astype(str)]
    return pd.DataFrame(rows, columns=columns, dtype=float), np.zeros(len(rows), dtype=int)


def main() -> int:
    metadata = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "development_dataset.csv")
    features = pd.read_csv(PROJECT_ROOT / "data" / "features" / "development_features.csv")
    columns = get_feature_names()
    frame = metadata.merge(features[["filepath", *columns]], on="filepath", validate="one_to_one")
    split_sets = {name: set(frame.loc[frame["split"] == name, "base_id"]) for name in ("train", "validation", "test")}
    if split_sets["train"] & split_sets["validation"] or split_sets["train"] & split_sets["test"] or split_sets["validation"] & split_sets["test"]:
        raise ValueError("base_id leakage detected")
    matrix, labels = frame[columns].astype(float), frame["label"].to_numpy(int)
    masks = {name: (frame["split"] == name).to_numpy() for name in ("train", "validation", "test")}
    external_x, external_y = external_matrix(columns)
    model_results, prediction_rows, threshold_rows = {}, [], []
    candidate_dir = MODELS / "development_candidates"; candidate_dir.mkdir(parents=True, exist_ok=True)
    trained = {}
    definitions = _model_definitions(42)
    definitions["XGBoost"].set_params(scale_pos_weight=1/6)
    definitions["CatBoost"].set_params(auto_class_weights="Balanced")
    for name, model in definitions.items():
        model.fit(matrix.loc[masks["train"]], labels[masks["train"]])
        validation_prob = attack_probabilities(model, matrix.loc[masks["validation"]])
        analysis = threshold_analysis(labels[masks["validation"]], validation_prob)
        feasible = analysis[analysis["false_positive_rate"] <= 0.10]
        if feasible.empty:
            minimum_fpr = analysis["false_positive_rate"].min()
            feasible = analysis[analysis["false_positive_rate"] == minimum_fpr]
        chosen = feasible.sort_values(["f1", "recall", "threshold"], ascending=[False, False, True]).iloc[0]
        threshold = float(chosen["threshold"])
        test_prob = attack_probabilities(model, matrix.loc[masks["test"]])
        test_frame = frame.loc[masks["test"]].copy(); test_frame["attack_probability"] = test_prob
        external_prob = attack_probabilities(model, external_x)
        result = {"threshold": threshold, "validation": metrics(labels[masks["validation"]], validation_prob, threshold), "test": metrics(labels[masks["test"]], test_prob, threshold), "held_out_clean_coco_fpr": float(np.mean(test_frame.loc[test_frame["label"] == 0, "attack_probability"] >= threshold)), "held_out_manipulated_recall": float(np.mean(test_frame.loc[test_frame["label"] == 1, "attack_probability"] >= threshold)), "prompt_conditioned_recall": float(np.mean(test_frame.loc[test_frame["attack_type"] == "prompt_conditioned", "attack_probability"] >= threshold)), "unseen_prompt_recall": float(np.mean(test_frame.loc[test_frame["attack_type"] == "prompt_conditioned", "attack_probability"] >= threshold)), "external_clean_false_positive_rate": float(np.mean(external_prob >= threshold)), "external_clean_mean_attack_probability": float(np.mean(external_prob))}
        model_results[name] = result
        for _, row in analysis.iterrows(): threshold_rows.append({"model_name": name, **row.to_dict()})
        joblib.dump(model, candidate_dir / f"{name.lower()}.pkl"); trained[name] = model
    ranking = sorted(model_results, key=lambda name: (model_results[name]["validation"]["f1"], model_results[name]["validation"]["recall"], -model_results[name]["validation"]["false_positive_rate"]), reverse=True)
    best = ranking[0]; best_result = model_results[best]
    joblib.dump(trained[best], MODELS / "candidate_best_model.pkl"); joblib.dump(columns, MODELS / "candidate_feature_columns.pkl")
    pd.DataFrame(threshold_rows).to_csv(MODELS / "candidate_threshold_analysis.csv", index=False)
    (MODELS / "candidate_all_model_results.json").write_text(json.dumps(model_results, indent=2), encoding="utf-8")
    candidate_metadata = {"model_name": best, "training_date": datetime.now(timezone.utc).isoformat(), "feature_count": len(columns), "threshold": best_result["threshold"], "dataset_size": len(frame), "base_image_count": int(frame["base_id"].nunique()), "split_method": "preassigned group-aware 70/15/15 by base_id", "split_sample_counts": {name: int(mask.sum()) for name, mask in masks.items()}, "metrics": best_result, "class_semantics": {"0": "clean", "1": "controlled_manipulated"}, "result_scope": "development candidate; production artifacts not replaced"}
    (MODELS / "candidate_model_metadata.json").write_text(json.dumps(candidate_metadata, indent=2), encoding="utf-8")
    print(json.dumps({"best_candidate": best, "result": best_result, "all_models": model_results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
