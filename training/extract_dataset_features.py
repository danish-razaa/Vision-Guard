"""Extract resumable fused features for the paired VisionGuard dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_extractor import extract_features, get_feature_names


LOGGER = logging.getLogger("visionguard.extract_dataset_features")
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata" / "dataset_metadata.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "features" / "fused_features.csv"
DEFAULT_FAILURES = PROJECT_ROOT / "data" / "metadata" / "feature_failures.csv"
METADATA_COLUMNS = ("label", "base_id", "attack_type", "epsilon", "filepath")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract fused multi-domain image features.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true", help="resume from an existing CSV")
    parser.add_argument("--checkpoint-every", type=positive_int, default=25)
    return parser


def _atomic_csv(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)


def extract_dataset(
    metadata_path: Path,
    output_path: Path,
    resume: bool,
    checkpoint_every: int,
) -> tuple[int, int]:
    metadata = pd.read_csv(metadata_path)
    required = set(METADATA_COLUMNS)
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    if metadata["filepath"].duplicated().any():
        raise ValueError("metadata contains duplicate filepath values")

    feature_names = get_feature_names()
    output_columns = list(METADATA_COLUMNS) + feature_names
    completed: dict[str, dict[str, object]] = {}
    if resume and output_path.is_file():
        previous = pd.read_csv(output_path)
        if list(previous.columns) != output_columns:
            raise ValueError("existing feature CSV columns do not match canonical ordering")
        completed = {
            str(row["filepath"]): row.to_dict() for _, row in previous.iterrows()
        }
        LOGGER.info("Resuming with %d completed rows", len(completed))

    rows = list(completed.values())
    failures: list[dict[str, str]] = []
    pending = metadata[~metadata["filepath"].astype(str).isin(completed)]
    for processed, (_, sample) in enumerate(
        tqdm(pending.iterrows(), total=len(pending), desc="Extracting features"), start=1
    ):
        filepath = str(sample["filepath"])
        try:
            features = extract_features(filepath)
        except Exception as exc:
            LOGGER.error("Feature extraction failed for %s: %s", filepath, exc)
            failures.append({"filepath": filepath, "error": str(exc)})
            continue
        row = {column: sample[column] for column in METADATA_COLUMNS}
        row.update(features)
        rows.append(row)
        if processed % checkpoint_every == 0:
            _atomic_csv(pd.DataFrame(rows, columns=output_columns), output_path)

    frame = pd.DataFrame(rows, columns=output_columns)
    if not frame.empty:
        numeric = frame[feature_names].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("output contains NaN or infinite feature values")
        order = {str(path): index for index, path in enumerate(metadata["filepath"])}
        frame["_order"] = frame["filepath"].astype(str).map(order)
        frame = frame.sort_values("_order").drop(columns="_order")
    _atomic_csv(frame, output_path)

    if failures:
        pd.DataFrame(failures).to_csv(DEFAULT_FAILURES, index=False)
    elif DEFAULT_FAILURES.exists():
        DEFAULT_FAILURES.unlink()

    parquet_path = output_path.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet_path, index=False)
        LOGGER.info("Parquet written to %s", parquet_path)
    except (ImportError, ModuleNotFoundError) as exc:
        LOGGER.info("Parquet skipped because no engine is installed: %s", exc)

    return len(frame), len(failures)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        rows, failures = extract_dataset(
            args.metadata.resolve(),
            args.output.resolve(),
            args.resume,
            args.checkpoint_every,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("Dataset feature extraction failed: %s", exc)
        return 1
    print(f"Feature rows: {rows}")
    print(f"Failures: {failures}")
    print(f"Output: {args.output.resolve()}")
    return 0 if rows > 0 and failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
