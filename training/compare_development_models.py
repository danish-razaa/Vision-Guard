"""Compare production and candidate detectors on identical development cohorts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_extractor import get_feature_names
from training.utils import attack_probabilities

def summarize(model_path: Path, metadata_path: Path, frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    model = joblib.load(model_path)
    threshold = float(json.loads(metadata_path.read_text(encoding="utf-8"))["threshold"])
    test = frame[frame["split"] == "test"].copy()
    probabilities = attack_probabilities(model, test[columns])
    test["probability"] = probabilities
    clean = test[test["label"] == 0]
    prompt = test[test["attack_type"] == "prompt_conditioned"]
    return {"threshold": threshold, "held_out_clean_coco_fpr": float(np.mean(clean["probability"] >= threshold)), "prompt_conditioned_recall": float(np.mean(prompt["probability"] >= threshold)), "unseen_prompt_recall": float(np.mean(prompt["probability"] >= threshold)), "prompt_mean_attack_probability": float(prompt["probability"].mean())}


def main() -> int:
    metadata = pd.read_csv(ROOT / "data" / "metadata" / "development_dataset.csv")
    features = pd.read_csv(ROOT / "data" / "features" / "development_features.csv")
    columns = get_feature_names(); frame = metadata.merge(features[["filepath", *columns]], on="filepath", validate="one_to_one")
    old_columns = list(joblib.load(ROOT / "models" / "feature_columns.pkl"))
    candidate_columns = list(joblib.load(ROOT / "models" / "candidate_feature_columns.pkl"))
    report = {"old": summarize(ROOT/"models"/"best_model.pkl", ROOT/"models"/"model_metadata.json", frame, old_columns), "candidate": summarize(ROOT/"models"/"candidate_best_model.pkl", ROOT/"models"/"candidate_model_metadata.json", frame, candidate_columns)}
    old_external = json.loads((ROOT/"models"/"old_external_generalization.json").read_text())["cohorts"]["external_clean"]["false_positive_rate"]
    candidate_results = json.loads((ROOT/"models"/"candidate_model_metadata.json").read_text())["metrics"]
    report["old"]["external_clean_fpr"] = old_external; report["candidate"]["external_clean_fpr"] = candidate_results["external_clean_false_positive_rate"]
    (ROOT/"models"/"development_old_vs_candidate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
