"""Train a reference-aware detector on original-to-candidate feature changes."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_extractor import get_feature_names


def pair_columns(feature_names: list[str]) -> list[str]:
    return [f"signed_delta__{name}" for name in feature_names] + [f"absolute_delta__{name}" for name in feature_names]


def build_pair_frame() -> tuple[pd.DataFrame, list[str]]:
    metadata = pd.read_csv(ROOT / "data" / "metadata" / "development_dataset.csv")
    features = pd.read_csv(ROOT / "data" / "features" / "development_features.csv")
    names = get_feature_names()
    frame = metadata.merge(features[["filepath", *names]], on="filepath", validate="one_to_one")
    rows = []
    for base_id, group in frame.groupby("base_id", sort=False):
        clean = group[group["label"] == 0]
        if len(clean) != 1:
            raise ValueError(f"base_id {base_id} does not have exactly one clean sample")
        reference = clean.iloc[0][names].to_numpy(float)
        common = {"base_id": base_id, "split": clean.iloc[0]["split"]}
        zero = np.zeros(len(names), dtype=float)
        rows.append({**common, "attack_type": "clean", "label": 0, **dict(zip(pair_columns(names), np.concatenate((zero, zero)), strict=True))})
        for _, candidate in group[group["label"] == 1].iterrows():
            signed = candidate[names].to_numpy(float) - reference
            vector = np.concatenate((signed, np.abs(signed)))
            rows.append({**common, "attack_type": candidate["attack_type"], "label": 1, **dict(zip(pair_columns(names), vector, strict=True))})
    return pd.DataFrame(rows), pair_columns(names)


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.unique(np.concatenate((np.linspace(0.05, 0.95, 91), probabilities))):
        predicted = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
        rows.append({"threshold": float(threshold), "precision": float(precision_score(labels, predicted, zero_division=0)), "recall": float(recall_score(labels, predicted, zero_division=0)), "f1": float(f1_score(labels, predicted, zero_division=0)), "false_positive_rate": float(fp/(fp+tn)), "false_negative_rate": float(fn/(fn+tp))})
    analysis = pd.DataFrame(rows)
    feasible = analysis[analysis["false_positive_rate"] <= 0.01]
    chosen = feasible.sort_values(["f1", "recall"], ascending=False).iloc[0]
    return float(chosen["threshold"]), analysis


def evaluate(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"precision": float(precision_score(labels, predicted, zero_division=0)), "recall": float(recall_score(labels, predicted, zero_division=0)), "f1": float(f1_score(labels, predicted, zero_division=0)), "roc_auc": float(roc_auc_score(labels, probabilities)), "false_positive_rate": float(fp/(fp+tn)), "false_negative_rate": float(fn/(fn+tp)), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def main() -> int:
    frame, columns = build_pair_frame()
    train, validation, test = (frame["split"] == name for name in ("train", "validation", "test"))
    model = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1, min_samples_leaf=2)
    model.fit(frame.loc[train, columns], frame.loc[train, "label"])
    validation_probabilities = model.predict_proba(frame.loc[validation, columns])[:, list(model.classes_).index(1)]
    threshold, analysis = choose_threshold(frame.loc[validation, "label"].to_numpy(), validation_probabilities)
    test_probabilities = model.predict_proba(frame.loc[test, columns])[:, list(model.classes_).index(1)]
    test_rows = frame.loc[test, ["base_id", "attack_type", "label"]].copy(); test_rows["attack_probability"] = test_probabilities; test_rows["predicted"] = (test_probabilities >= threshold).astype(int)
    prompt = test_rows[test_rows["attack_type"] == "prompt_conditioned"]
    result = {"threshold": threshold, "feature_count": len(columns), "validation": evaluate(frame.loc[validation, "label"].to_numpy(), validation_probabilities, threshold), "test": evaluate(frame.loc[test, "label"].to_numpy(), test_probabilities, threshold), "unseen_prompt_recall": float(prompt["predicted"].mean()), "unseen_prompt_count": len(prompt), "model_name": "RandomForestPairDetector"}
    models = ROOT / "models"; joblib.dump(model, models / "pair_detector.pkl"); joblib.dump(columns, models / "pair_feature_columns.pkl")
    analysis.to_csv(models / "pair_threshold_analysis.csv", index=False); test_rows.to_csv(models / "pair_test_predictions.csv", index=False)
    metadata = {**result, "training_date": datetime.now(timezone.utc).isoformat(), "class_semantics": {"0": "no measured controlled change", "1": "controlled perturbation detected"}, "scope": "reference-aware Prompt Injection Lab comparison; not a standalone image detector"}
    (models / "pair_detector_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
