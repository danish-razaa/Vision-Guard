"""Acquire an evaluation-only Open Images clean holdout."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fiftyone"
DB_DIR = PROJECT_ROOT / "data" / "raw" / "fiftyone-db"
OUTPUT_DIR = PROJECT_ROOT / "data" / "external_holdout"
MANIFEST = PROJECT_ROOT / "data" / "metadata" / "external_holdout_manifest.csv"


def acquire(max_samples: int, seed: int) -> int:
    compatibility = PROJECT_ROOT / "training" / "_compat"
    sys.path.insert(0, str(compatibility))
    os.environ["PYTHONPATH"] = str(compatibility) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["FIFTYONE_DATASET_ZOO_DIR"] = str(RAW_DIR)
    os.environ["FIFTYONE_DATABASE_DIR"] = str(DB_DIR)
    import fiftyone.zoo as foz

    dataset = foz.load_zoo_dataset(
        "open-images-v7", split="validation", max_samples=max_samples,
        shuffle=True, seed=seed, dataset_name="visionguard_external_holdout",
        drop_existing_dataset=True,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sample in enumerate(dataset.iter_samples(progress=True)):
        source = Path(sample.filepath).resolve()
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            continue
        destination = OUTPUT_DIR / f"openimages_validation_{index:06d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        rows.append({"base_id": f"openimages_validation_{index:06d}", "filepath": str(destination.resolve()), "label": 0, "source_dataset": "open-images-v7", "attack_type": "clean", "epsilon": 0.0, "holdout_only": True})
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ("base_id", "filepath", "label", "source_dataset", "attack_type", "epsilon", "holdout_only"))
        writer.writeheader(); writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download evaluation-only Open Images holdout")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()
    count = acquire(args.max_samples, args.seed)
    print(f"External clean holdout images: {count}")
    print(f"Manifest: {MANIFEST}")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
