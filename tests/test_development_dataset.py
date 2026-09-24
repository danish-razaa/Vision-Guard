"""Integrity checks for the corrected development experiment."""

from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def test_development_base_ids_never_cross_splits() -> None:
    frame = pd.read_csv(ROOT / "data" / "metadata" / "development_dataset.csv")
    memberships = frame.groupby("base_id")["split"].nunique()
    assert len(memberships) == 500
    assert memberships.eq(1).all()


def test_development_attack_families_are_balanced() -> None:
    frame = pd.read_csv(ROOT / "data" / "metadata" / "development_dataset.csv")
    attacked = frame[frame["label"] == 1]
    counts = attacked["attack_type"].value_counts()
    assert set(counts.index) == {"gaussian", "sinusoidal", "checkerboard", "dct", "fft", "prompt_conditioned"}
    assert counts.nunique() == 1


def test_clean_and_manipulated_outputs_use_same_encoding() -> None:
    frame = pd.read_csv(ROOT / "data" / "metadata" / "development_dataset.csv")
    for filepath in frame.groupby("label").head(5)["filepath"]:
        with Image.open(filepath) as image:
            assert image.format == "PNG"
            assert image.mode == "RGB"
            assert image.size == (256, 256)


def test_external_holdout_is_not_in_development_dataset() -> None:
    development = pd.read_csv(ROOT / "data" / "metadata" / "development_dataset.csv")
    external = pd.read_csv(ROOT / "data" / "metadata" / "external_holdout_manifest.csv")
    assert set(development["base_id"]).isdisjoint(external["base_id"])
    assert external["holdout_only"].astype(bool).all()
