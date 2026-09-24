"""Reproduce and explain one original/modified Prompt Lab pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import initialize_inference
from app.xai import feature_domain
from core.feature_extractor import extract_features


def diagnose(original_path: Path, modified_path: Path, output_dir: Path) -> dict[str, object]:
    service = initialize_inference()
    original_features = extract_features(original_path, service.smoothing_config)
    modified_features = extract_features(modified_path, service.smoothing_config)
    columns = service.feature_columns
    if list(original_features) != columns or list(modified_features) != columns:
        raise ValueError("runtime feature column order differs from the saved model order")
    original_vector = np.array([original_features[name] for name in columns], dtype=float)
    modified_vector = np.array([modified_features[name] for name in columns], dtype=float)
    original_result = service.predict(original_path)
    modified_result = service.predict(modified_path)

    delta_rows = []
    for index, name in enumerate(columns):
        original = float(original_vector[index])
        modified = float(modified_vector[index])
        signed = modified - original
        relative = abs(signed) / abs(original) if abs(original) > 1e-12 else (0.0 if signed == 0 else None)
        delta_rows.append({"feature_name": name, "feature_domain": feature_domain(name), "original_value": original, "modified_value": modified, "absolute_delta": abs(signed), "signed_delta": signed, "relative_delta": relative})
    delta_frame = pd.DataFrame(delta_rows).sort_values("absolute_delta", ascending=False)

    original_shap = service.xai._contributions(original_vector)
    modified_shap = service.xai._contributions(modified_vector)
    shap_frame = pd.DataFrame({
        "feature_name": columns,
        "feature_domain": [feature_domain(name) for name in columns],
        "original_value": original_vector,
        "modified_value": modified_vector,
        "original_shap_contribution": original_shap,
        "modified_shap_contribution": modified_shap,
        "shap_delta": modified_shap - original_shap,
    })
    shap_frame["absolute_shap_delta"] = shap_frame["shap_delta"].abs()
    shap_frame = shap_frame.sort_values("absolute_shap_delta", ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_frame.to_csv(output_dir / "prompt_experiment_feature_delta.csv", index=False)
    shap_frame.to_csv(output_dir / "prompt_experiment_shap_delta.csv", index=False)

    classes = np.asarray(service.model.classes_).tolist()
    report = {
        "model_classes": classes,
        "class_semantics": {"0": "CLEAN", "1": "MANIPULATED"},
        "clean_class_index": service.clean_class_index,
        "attack_class_index": service.attack_class_index,
        "selected_threshold_percent": service.threshold * 100.0,
        "feature_count": len(columns),
        "feature_column_order": columns,
        "smoothing_configuration": service.smoothing_config.__dict__,
        "original": {key: original_result[key] for key in ("status", "clean_probability", "attack_probability")},
        "modified": {key: modified_result[key] for key in ("status", "clean_probability", "attack_probability")},
        "risk_delta_percentage_points": modified_result["attack_probability"] - original_result["attack_probability"],
        "top_15_feature_changes": delta_frame.head(15).to_dict(orient="records"),
        "top_15_shap_changes": shap_frame.head(15).to_dict(orient="records"),
    }
    (output_dir / "prompt_experiment_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose a Prompt Injection Lab image pair")
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--modified", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models")
    args = parser.parse_args()
    print(json.dumps(diagnose(args.original.resolve(), args.modified.resolve(), args.output_dir.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
