"""Audit label-correlated encoding and source shortcuts in dataset metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def audit(metadata_path: Path) -> dict[str, object]:
    frame = pd.read_csv(metadata_path)
    required = {"filepath", "label", "base_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    rows = []
    for _, sample in frame.iterrows():
        path = Path(str(sample["filepath"])).expanduser().resolve()
        with Image.open(path) as opened:
            transposed = ImageOps.exif_transpose(opened)
            rows.append({"label": int(sample["label"]), "source_dataset": sample.get("source_dataset", "unknown"), "extension": path.suffix.lower(), "format": opened.format, "mode": transposed.mode, "width": transposed.width, "height": transposed.height, "has_exif": bool(opened.getexif())})
    details = pd.DataFrame(rows)
    def grouped(columns: list[str]) -> list[dict[str, object]]:
        return details.groupby(columns, dropna=False).size().rename("count").reset_index().to_dict(orient="records")

    report = {
        "sample_count": len(details),
        "by_label_and_format": grouped(["label", "format"]),
        "by_label_and_dimensions": grouped(["label", "width", "height"]),
        "by_label_and_mode": grouped(["label", "mode"]),
        "by_label_and_source": grouped(["label", "source_dataset"]),
        "exif_by_label": details.groupby("label")["has_exif"].mean().to_dict(),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit VisionGuard dataset shortcuts")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data" / "metadata" / "dataset_metadata.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "dataset_shortcut_audit.json")
    args = parser.parse_args()
    result = audit(args.metadata)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
