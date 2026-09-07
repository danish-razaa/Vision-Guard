"""Record four actual candidate predictions without changing production artifacts."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import VisionGuardInference
from core.prompt_perturbation import generate_prompt_conditioned_perturbation
from training.prompt_banks import TEST_PROMPTS

def main() -> int:
    service = VisionGuardInference(ROOT/"models"/"candidate_best_model.pkl", ROOT/"models"/"candidate_feature_columns.pkl", ROOT/"models"/"candidate_model_metadata.json")
    development = pd.read_csv(ROOT/"data"/"metadata"/"development_dataset.csv")
    coco = development[(development["split"] == "test") & (development["label"] == 0)].iloc[0]
    external = pd.read_csv(ROOT/"data"/"metadata"/"external_holdout_manifest.csv").iloc[0]
    report = {}
    with tempfile.TemporaryDirectory(prefix="visionguard_candidate_validation_") as temporary:
        for name, path in (("clean_coco", coco["filepath"]), ("external_clean", external["filepath"])):
            source = cv2.imread(str(path), cv2.IMREAD_COLOR)
            normalized = cv2.resize(source, (256, 256), interpolation=cv2.INTER_AREA)
            clean_path = Path(temporary)/f"{name}.png"; cv2.imwrite(str(clean_path), normalized)
            modified, _, _ = generate_prompt_conditioned_perturbation(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB), TEST_PROMPTS[0], 4.0)
            modified_path = Path(temporary)/f"{name}_prompt.png"; cv2.imwrite(str(modified_path), cv2.cvtColor(modified, cv2.COLOR_RGB2BGR))
            report[name] = service.predict(clean_path)
            report[f"{name}_prompt_conditioned"] = service.predict(modified_path)
    for result in report.values(): result.pop("xai_plot_path", None)
    (ROOT/"models"/"candidate_manual_scenarios.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: {"status": value["status"], "attack_probability": value["attack_probability"]} for key, value in report.items()}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
