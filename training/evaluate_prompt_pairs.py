"""Evaluate prompt-conditioned pairs on base images from one saved split."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = PROJECT_ROOT / "models"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import VisionGuardInference, initialize_inference
from core.prompt_perturbation import generate_prompt_conditioned_perturbation
from training.prompt_banks import TEST_PROMPTS


def evaluate(split: str, epsilon: float, output: Path, candidate: bool = False) -> dict[str, object]:
    assignments = pd.read_csv(
        PROJECT_ROOT / "data" / "metadata" / "development_dataset.csv"
        if candidate
        else PROJECT_ROOT / "models" / "split_assignments.csv"
    )
    clean = assignments[(assignments["split"] == split) & (assignments["label"] == 0)].drop_duplicates("base_id")
    if clean.empty:
        raise ValueError(f"no clean base images found in split {split!r}")
    service = VisionGuardInference(MODELS / "candidate_best_model.pkl", MODELS / "candidate_feature_columns.pkl", MODELS / "candidate_model_metadata.json") if candidate else initialize_inference()
    rows = []
    with tempfile.TemporaryDirectory(prefix="visionguard_pair_evaluation_") as temporary:
        for index, (_, sample) in enumerate(clean.iterrows()):
            source = cv2.imread(str(sample["filepath"]), cv2.IMREAD_COLOR)
            if source is None:
                raise ValueError(f"could not decode {sample['filepath']}")
            prompt = TEST_PROMPTS[index % len(TEST_PROMPTS)]
            modified_rgb, _, metadata = generate_prompt_conditioned_perturbation(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), prompt, epsilon)
            modified_path = Path(temporary) / f"{sample['base_id']}.png"
            cv2.imwrite(str(modified_path), cv2.cvtColor(modified_rgb, cv2.COLOR_RGB2BGR))
            original = service.predict(sample["filepath"])
            modified = service.predict(modified_path)
            original_suspicious = original["status"] == "SUSPICIOUS"
            modified_suspicious = modified["status"] == "SUSPICIOUS"
            transition = ("C" if original_suspicious and modified_suspicious else "D" if original_suspicious else "A" if modified_suspicious else "B")
            rows.append({"base_id": sample["base_id"], "prompt_hash": metadata["prompt_hash"], "epsilon": epsilon, "original_attack_probability": original["attack_probability"], "modified_attack_probability": modified["attack_probability"], "risk_delta": modified["attack_probability"] - original["attack_probability"], "original_prediction": original["status"], "modified_prediction": modified["status"], "transition": transition})
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    counts = frame["transition"].value_counts().reindex(list("ABCD"), fill_value=0)
    deltas = frame["risk_delta"].to_numpy(float)
    summary = {
        "split": split, "number_tested": len(frame), "epsilon": epsilon,
        "predicted_clean_original": int((frame["original_prediction"] != "SUSPICIOUS").sum()),
        "predicted_suspicious_original": int((frame["original_prediction"] == "SUSPICIOUS").sum()),
        "clean_false_positive_rate": float((frame["original_prediction"] == "SUSPICIOUS").mean()),
        "original_probability_mean": float(frame["original_attack_probability"].mean()),
        "original_probability_median": float(frame["original_attack_probability"].median()),
        "original_probability_min": float(frame["original_attack_probability"].min()),
        "original_probability_max": float(frame["original_attack_probability"].max()),
        "transitions": {key: {"count": int(counts[key]), "percentage": float(100 * counts[key] / len(frame))} for key in "ABCD"},
        "mean_risk_delta": float(np.mean(deltas)), "median_risk_delta": float(np.median(deltas)),
        "risk_delta_positive_percentage": float(100 * np.mean(deltas > 0)),
        "risk_delta_negative_percentage": float(100 * np.mean(deltas < 0)),
        "modified_probability_greater_percentage": float(100 * np.mean(frame["modified_attack_probability"] > frame["original_attack_probability"])),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate held-out Prompt Lab pairs")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "prompt_pair_evaluation.csv")
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.split, args.epsilon, args.output, args.candidate), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
