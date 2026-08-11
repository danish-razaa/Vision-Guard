"""Download and validate a deterministic COCO-2017 clean-image subset."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2


LOGGER = logging.getLogger("visionguard.download_dataset")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "clean_manifest.csv"
DEFAULT_FIFTYONE_DIR = PROJECT_ROOT / "data" / "raw" / "fiftyone"
DEFAULT_FIFTYONE_DATABASE_DIR = PROJECT_ROOT / "data" / "raw" / "fiftyone-db"
DATASET_NAME = "visionguard_coco"
MANIFEST_FIELDS = ("base_id", "original_path", "clean_path", "source", "split")


@dataclass(frozen=True)
class AcquisitionStats:
    """Actual results from clean-image acquisition."""

    requested: int
    dataset_samples: int
    valid: int
    failed: int


def positive_int(value: str) -> int:
    """Argparse type that accepts positive integer sample counts only."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a deterministic COCO-2017 training subset via FiftyOne."
    )
    parser.add_argument(
        "--max-samples",
        type=positive_int,
        default=100,
        help="maximum number of COCO train images to request (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="deterministic shuffle seed (default: 42)",
    )
    return parser


def image_is_decodable(path: Path) -> bool:
    """Return whether OpenCV can decode a non-empty image at ``path``."""
    if not path.is_file():
        return False
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image is not None and image.size > 0


def _load_coco_dataset(max_samples: int, seed: int):
    """Load the requested zoo subset, reusing FiftyOne's download cache."""
    DEFAULT_FIFTYONE_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_FIFTYONE_DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FIFTYONE_DATASET_ZOO_DIR", str(DEFAULT_FIFTYONE_DIR))
    os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(DEFAULT_FIFTYONE_DATABASE_DIR))
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "raw" / "matplotlib"))

    import fiftyone as fo
    import fiftyone.zoo as foz

    if fo.dataset_exists(DATASET_NAME):
        existing = fo.load_dataset(DATASET_NAME)
        if len(existing) == max_samples:
            LOGGER.info(
                "Reusing existing FiftyOne dataset '%s' with %d samples",
                DATASET_NAME,
                len(existing),
            )
            return existing

        LOGGER.info(
            "Existing dataset '%s' has %d samples; rebuilding its database view "
            "for a request of %d (downloaded files remain cached)",
            DATASET_NAME,
            len(existing),
            max_samples,
        )

    return foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        max_samples=max_samples,
        shuffle=True,
        seed=seed,
        dataset_name=DATASET_NAME,
        drop_existing_dataset=True,
    )


def _base_id(original_path: Path) -> str:
    """Create a stable COCO identifier from the source filename."""
    return f"coco_train_{original_path.stem}"


def _copy_valid_samples(dataset, clean_dir: Path) -> tuple[list[dict[str, str]], int]:
    clean_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    failures = 0
    seen_ids: set[str] = set()

    for sample in dataset.iter_samples(progress=True):
        original_path = Path(sample.filepath).expanduser().resolve()
        base_id = _base_id(original_path)

        if base_id in seen_ids:
            failures += 1
            LOGGER.error("Duplicate base_id %s from %s", base_id, original_path)
            continue
        if not image_is_decodable(original_path):
            failures += 1
            LOGGER.error("Could not decode source image: %s", original_path)
            continue

        suffix = original_path.suffix.lower() or ".jpg"
        clean_path = (clean_dir / f"{base_id}{suffix}").resolve()
        try:
            if not clean_path.exists():
                shutil.copy2(original_path, clean_path)
            if not image_is_decodable(clean_path):
                raise ValueError("copied image could not be decoded")
        except (OSError, ValueError) as exc:
            failures += 1
            LOGGER.error("Failed to prepare %s: %s", original_path, exc)
            continue

        seen_ids.add(base_id)
        rows.append(
            {
                "base_id": base_id,
                "original_path": str(original_path),
                "clean_path": str(clean_path),
                "source": "coco-2017",
                "split": "train",
            }
        )

    return rows, failures


def _write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(manifest_path)


def acquire_clean_images(max_samples: int, seed: int) -> AcquisitionStats:
    """Acquire, validate, copy, and record a clean COCO subset."""
    LOGGER.info(
        "Requesting up to %d COCO-2017 train samples with seed %d",
        max_samples,
        seed,
    )
    dataset = _load_coco_dataset(max_samples=max_samples, seed=seed)
    dataset_samples = len(dataset)
    rows, failures = _copy_valid_samples(dataset, DEFAULT_CLEAN_DIR)
    _write_manifest(rows, DEFAULT_MANIFEST)

    stats = AcquisitionStats(
        requested=max_samples,
        dataset_samples=dataset_samples,
        valid=len(rows),
        failed=failures,
    )
    LOGGER.info("Manifest written to %s", DEFAULT_MANIFEST)
    LOGGER.info(
        "Actual statistics: requested=%d dataset_samples=%d valid=%d failed=%d",
        stats.requested,
        stats.dataset_samples,
        stats.valid,
        stats.failed,
    )
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        stats = acquire_clean_images(args.max_samples, args.seed)
    except Exception:
        LOGGER.exception("COCO acquisition failed")
        return 1

    print(f"Requested samples: {stats.requested}")
    print(f"FiftyOne dataset samples: {stats.dataset_samples}")
    print(f"Valid clean images: {stats.valid}")
    print(f"Failed images: {stats.failed}")
    print(f"Manifest: {DEFAULT_MANIFEST}")
    return 0 if stats.valid > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
