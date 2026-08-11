"""Unit tests for Chapter 2 downloader input and image validation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from training.download_dataset import build_parser, image_is_decodable


def test_parser_rejects_non_positive_sample_count() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--max-samples", "0"])


def test_image_decode_validation(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.png"
    corrupt_path = tmp_path / "corrupt.png"
    assert cv2.imwrite(str(valid_path), np.zeros((8, 8, 3), dtype=np.uint8))
    corrupt_path.write_bytes(b"not an image")

    assert image_is_decodable(valid_path)
    assert not image_is_decodable(corrupt_path)
    assert not image_is_decodable(tmp_path / "missing.png")
