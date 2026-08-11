"""Audit VisionGuard predictions without changing model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.utils import attack_probabilities


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write auditable clean/attack predictions.")
    parser.add_argument("--n-clean", type=_positive_int, default=20)
    parser.add_argument("--n-attack", type=_positive_int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "debug_predictions.csv",
    )
    return parser


def run(n_clean: int, n_attack: int, output: Path) -> pd.DataFrame:
    features = pd.read_csv(PROJECT_ROOT / "data" / "features" / "fused_features.csv")
    columns = list(joblib.load(PROJECT_ROOT / "models" / "feature_columns.pkl"))
    model = joblib.load(PROJECT_ROOT / "models" / "best_model.pkl")
    metadata = json.loads(
        (PROJECT_ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8")
    )
    selected = pd.concat(
        [
            features[features.label == 0].head(n_clean),
            features[features.label == 1].head(n_attack),
        ],
        ignore_index=True,
    )
    matrix = selected[columns].astype(float)
    if list(matrix.columns) != columns:
        raise ValueError("inference feature order differs from feature_columns.pkl")
    probabilities = attack_probabilities(model, matrix)
    threshold = float(metadata["threshold"])
    result = selected[["filepath", "base_id", "label", "attack_type", "epsilon"]].copy()
    result["attack_probability"] = probabilities
    result["threshold"] = threshold
    result["predicted_label"] = (probabilities >= threshold).astype(int)
    result["correct"] = result["predicted_label"] == result["label"]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    y = result.label.to_numpy()
    predicted = result.predicted_label.to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    print(f"model={metadata.get('model_name')} classes={list(model.classes_)} threshold={threshold:.6f}")
    print(f"TN={tn} FP={fp} FN={fn} TP={tp} FPR={fp / (tn + fp):.6f}")
    print(
        f"precision={precision_score(y, predicted, zero_division=0):.6f} "
        f"recall={recall_score(y, predicted, zero_division=0):.6f} "
        f"f1={f1_score(y, predicted, zero_division=0):.6f}"
    )
    print(f"output={output.resolve()}")
    return result


def main() -> int:
    args = build_parser().parse_args()
    run(args.n_clean, args.n_attack, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
