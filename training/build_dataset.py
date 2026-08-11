"""Build the paired clean/manipulated VisionGuard smoke dataset."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from core.preprocessing import preprocess_image

try:
    from training.generate_perturbations import ATTACK_TYPES, generate_perturbation
except ModuleNotFoundError:  # Supports direct execution from training/
    from generate_perturbations import ATTACK_TYPES, generate_perturbation


LOGGER = logging.getLogger("visionguard.build_dataset")
CLEAN_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "clean_manifest.csv"
DATASET_METADATA = PROJECT_ROOT / "data" / "metadata" / "dataset_metadata.csv"
FAILURE_LOG = PROJECT_ROOT / "data" / "metadata" / "perturbation_failures.csv"
ATTACKED_DIR = PROJECT_ROOT / "data" / "attacked"
NORMALIZED_CLEAN_DIR = PROJECT_ROOT / "data" / "clean_normalized"
METADATA_FIELDS = (
    "base_id",
    "filepath",
    "clean_path",
    "attacked_path",
    "attack_type",
    "epsilon",
    "label",
    "seed",
)


@dataclass(frozen=True)
class BuildStats:
    clean: int
    attacked: int
    failed: int


def finite_positive(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paired clean and controlled manipulated image metadata."
    )
    parser.add_argument("--epsilon-min", type=finite_positive, default=1.0)
    parser.add_argument("--epsilon-max", type=finite_positive, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _write_csv(path: Path, rows: list[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_dataset(epsilon_min: float, epsilon_max: float, seed: int) -> BuildStats:
    if epsilon_min > epsilon_max:
        raise ValueError("epsilon-min must be less than or equal to epsilon-max")
    if not CLEAN_MANIFEST.is_file():
        raise FileNotFoundError(f"clean manifest not found: {CLEAN_MANIFEST}")

    clean_manifest = pd.read_csv(CLEAN_MANIFEST)
    required = {"base_id", "clean_path"}
    missing = required.difference(clean_manifest.columns)
    if missing:
        raise ValueError(f"clean manifest missing columns: {sorted(missing)}")
    if clean_manifest["base_id"].duplicated().any():
        raise ValueError("clean manifest contains duplicate base_id values")

    ATTACKED_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(seed)
    metadata_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for index, row in clean_manifest.reset_index(drop=True).iterrows():
        base_id = str(row["base_id"])
        source_clean_path = Path(str(row["clean_path"])).expanduser().resolve()
        clean_path = (NORMALIZED_CLEAN_DIR / f"{base_id}.png").resolve()
        attack_type = ATTACK_TYPES[index % len(ATTACK_TYPES)]
        epsilon = float(master_rng.uniform(epsilon_min, epsilon_max))
        sample_seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
        attacked_path = (ATTACKED_DIR / f"{base_id}_{attack_type}.png").resolve()

        try:
            # Both labels must start from the same decoded/resized pixels and use
            # the same lossless encoding. Otherwise the model can learn JPEG vs
            # PNG artifacts instead of manipulation evidence.
            clean_image = preprocess_image(source_clean_path, Config.IMAGE_SIZE).bgr
            if not cv2.imwrite(str(clean_path), clean_image):
                raise OSError("OpenCV failed to write normalized clean PNG")
            clean_image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
            if clean_image is None or clean_image.shape[:2] != (Config.IMAGE_SIZE, Config.IMAGE_SIZE):
                raise ValueError("normalized clean PNG failed validation")
            manipulated = generate_perturbation(
                clean_image, attack_type, epsilon, sample_seed
            )
            if not cv2.imwrite(str(attacked_path), manipulated):
                raise OSError("OpenCV failed to write manipulated PNG")
            verified = cv2.imread(str(attacked_path), cv2.IMREAD_COLOR)
            if verified is None or verified.shape != clean_image.shape:
                raise ValueError("written manipulated image failed validation")
        except (OSError, ValueError, cv2.error) as exc:
            LOGGER.error("Failed base_id=%s: %s", base_id, exc)
            failures.append(
                {
                    "base_id": base_id,
                    "clean_path": str(clean_path),
                    "attack_type": attack_type,
                    "epsilon": epsilon,
                    "seed": sample_seed,
                    "error": str(exc),
                }
            )
            continue

        common = {
            "base_id": base_id,
            "clean_path": str(clean_path),
            "attacked_path": str(attacked_path),
            "seed": sample_seed,
        }
        metadata_rows.append(
            {
                **common,
                "filepath": str(clean_path),
                "attack_type": "clean",
                "epsilon": 0.0,
                "label": 0,
            }
        )
        metadata_rows.append(
            {
                **common,
                "filepath": str(attacked_path),
                "attack_type": attack_type,
                "epsilon": epsilon,
                "label": 1,
            }
        )

    _write_csv(DATASET_METADATA, metadata_rows, METADATA_FIELDS)
    if failures:
        _write_csv(
            FAILURE_LOG,
            failures,
            ("base_id", "clean_path", "attack_type", "epsilon", "seed", "error"),
        )
    elif FAILURE_LOG.exists():
        FAILURE_LOG.unlink()

    stats = BuildStats(
        clean=sum(row["label"] == 0 for row in metadata_rows),
        attacked=sum(row["label"] == 1 for row in metadata_rows),
        failed=len(failures),
    )
    LOGGER.info(
        "Actual build statistics: clean=%d attacked=%d failed=%d metadata=%s",
        stats.clean,
        stats.attacked,
        stats.failed,
        DATASET_METADATA,
    )
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        stats = build_dataset(args.epsilon_min, args.epsilon_max, args.seed)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("Dataset build failed: %s", exc)
        return 1
    print(f"Clean samples: {stats.clean}")
    print(f"Manipulated samples: {stats.attacked}")
    print(f"Failures: {stats.failed}")
    print(f"Metadata: {DATASET_METADATA}")
    return 0 if stats.clean > 0 and stats.attacked > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
