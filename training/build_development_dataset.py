"""Build the leakage-safe 500-base development dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from core.preprocessing import preprocess_image
from core.prompt_perturbation import generate_prompt_conditioned_perturbation
from training.generate_perturbations import ATTACK_TYPES, generate_perturbation
from training.prompt_banks import TEST_PROMPTS, TRAIN_PROMPTS, VALIDATION_PROMPTS
from training.utils import group_aware_split_indices

OUTPUT_ROOT = PROJECT_ROOT / "data" / "development"
METADATA = PROJECT_ROOT / "data" / "metadata" / "development_dataset.csv"
STRENGTHS = (("very-low", 0.5), ("low", 1.0), ("medium", 2.0), ("high", 4.0))
FIELDS = ("base_id", "source_dataset", "split", "filepath", "clean_path", "attack_type", "strength_band", "epsilon", "prompt_hash", "seed", "label")


def build(manifest_path: Path, seed: int = 42) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    if len(manifest) < 3 or {"base_id", "clean_path"} - set(manifest.columns):
        raise ValueError("clean manifest requires base_id and clean_path with at least three rows")
    manifest = manifest.drop_duplicates("base_id").reset_index(drop=True)
    split_indices = group_aware_split_indices(manifest["base_id"].astype(str).to_numpy(), seed=seed)
    split_for_index = {}
    for name, indices in (("train", split_indices.train), ("validation", split_indices.validation), ("test", split_indices.test)):
        split_for_index.update({int(index): name for index in indices})
    clean_dir, attacked_dir = OUTPUT_ROOT / "clean", OUTPUT_ROOT / "attacked"
    clean_dir.mkdir(parents=True, exist_ok=True); attacked_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    prompt_banks = {"train": TRAIN_PROMPTS, "validation": VALIDATION_PROMPTS, "test": TEST_PROMPTS}
    for index, sample in manifest.iterrows():
        base_id, split = str(sample["base_id"]), split_for_index[index]
        source_dataset = str(sample.get("source", "coco-2017"))
        normalized = preprocess_image(Path(str(sample["clean_path"])), Config.IMAGE_SIZE).bgr
        clean_path = clean_dir / f"{base_id}.png"
        if not cv2.imwrite(str(clean_path), normalized):
            raise OSError(f"could not write {clean_path}")
        rows.append({"base_id": base_id, "source_dataset": source_dataset, "split": split, "filepath": str(clean_path.resolve()), "clean_path": str(clean_path.resolve()), "attack_type": "clean", "strength_band": "none", "epsilon": 0.0, "prompt_hash": "", "seed": "", "label": 0})
        for attack_index, attack_type in enumerate((*ATTACK_TYPES, "prompt_conditioned")):
            strength_band, epsilon = STRENGTHS[(index + attack_index) % len(STRENGTHS)]
            sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            prompt_hash = ""
            if attack_type == "prompt_conditioned":
                prompt = prompt_banks[split][index % len(prompt_banks[split])]
                rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
                modified_rgb, _, prompt_metadata = generate_prompt_conditioned_perturbation(rgb, prompt, epsilon)
                manipulated = cv2.cvtColor(modified_rgb, cv2.COLOR_RGB2BGR)
                prompt_hash, sample_seed = prompt_metadata["prompt_hash"], prompt_metadata["seed"]
            else:
                manipulated = generate_perturbation(normalized, attack_type, epsilon, sample_seed)
            attacked_path = attacked_dir / f"{base_id}_{attack_type}.png"
            if not cv2.imwrite(str(attacked_path), manipulated):
                raise OSError(f"could not write {attacked_path}")
            rows.append({"base_id": base_id, "source_dataset": source_dataset, "split": split, "filepath": str(attacked_path.resolve()), "clean_path": str(clean_path.resolve()), "attack_type": attack_type, "strength_band": strength_band, "epsilon": epsilon, "prompt_hash": prompt_hash, "seed": sample_seed, "label": 1})
    frame = pd.DataFrame(rows, columns=FIELDS)
    temporary = METADATA.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False); temporary.replace(METADATA)
    composition = {
        "total_base_images": int(frame["base_id"].nunique()),
        "bases_by_split": frame.drop_duplicates("base_id").groupby("split")["base_id"].nunique().to_dict(),
        "clean_samples": int((frame["label"] == 0).sum()), "manipulated_samples": int((frame["label"] == 1).sum()),
        "by_attack_type": frame.groupby("attack_type").size().to_dict(), "by_strength_band": frame.groupby("strength_band").size().to_dict(),
    }
    (PROJECT_ROOT / "models" / "development_dataset_composition.json").write_text(__import__("json").dumps(composition, indent=2), encoding="utf-8")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Build corrected development dataset")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "metadata" / "clean_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); frame = build(args.manifest.resolve(), args.seed)
    print(frame.groupby(["split", "label"]).size())
    print(frame.groupby("attack_type").size())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
