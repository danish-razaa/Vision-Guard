"""Run one prompt-conditioned simulation against existing model artifacts."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import initialize_inference
from core.feature_comparison import compare_feature_vectors
from core.feature_extractor import extract_features
from core.prompt_perturbation import generate_prompt_conditioned_perturbation

STRENGTHS = {"very_low": 0.5, "low": 1.0, "medium": 2.0, "high": 4.0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostic controlled perturbation comparison")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--strength", choices=STRENGTHS, default="low")
    args = parser.parse_args()
    source = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if source is None:
        parser.error("image could not be decoded")
    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    modified, _, _ = generate_prompt_conditioned_perturbation(rgb, args.prompt, STRENGTHS[args.strength])
    service = initialize_inference()
    with tempfile.TemporaryDirectory(prefix="visionguard_prompt_diagnostic_") as directory:
        original_path, modified_path = Path(directory)/"original.png", Path(directory)/"modified.png"
        cv2.imwrite(str(original_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(modified_path), cv2.cvtColor(modified, cv2.COLOR_RGB2BGR))
        original, changed = service.predict(original_path), service.predict(modified_path)
        deltas = compare_feature_vectors(extract_features(original_path, service.smoothing_config), extract_features(modified_path, service.smoothing_config))
    print(f"Original attack probability: {original['attack_probability']:.3f}%")
    print(f"Modified attack probability: {changed['attack_probability']:.3f}%")
    print(f"Risk delta: {changed['attack_probability'] - original['attack_probability']:+.3f} percentage points")
    print(f"Prediction before: {original['status']}")
    print(f"Prediction after: {changed['status']}")
    print("Top changed feature domains: " + ", ".join(dict.fromkeys(row['feature_domain'] for row in deltas)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
