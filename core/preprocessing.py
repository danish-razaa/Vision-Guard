"""Shared image decoding and preprocessing for training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from config import Config


ImageArray: TypeAlias = NDArray[np.uint8]
ImageInput: TypeAlias = str | Path | ImageArray


class ImagePreprocessingError(ValueError):
    """Raised when an image cannot be validated or decoded safely."""


@dataclass(frozen=True)
class PreprocessedImage:
    """Canonical representations created by the shared pipeline.

    ``original_bgr`` preserves decoded pixels for later display or auditing.
    ``bgr``, ``rgb``, and ``grayscale`` are resized to the configured square
    dimensions and are ready for downstream smoothing and feature extraction.
    """

    original_bgr: ImageArray
    bgr: ImageArray
    rgb: ImageArray
    grayscale: ImageArray
    source_path: Path | None = None

    @property
    def image_size(self) -> tuple[int, int]:
        return self.bgr.shape[:2]


def _decode_path(path_value: str | Path) -> tuple[ImageArray, Path]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ImagePreprocessingError(f"image file does not exist: {path}")
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ImagePreprocessingError(f"could not read image file: {path}") from exc
    if encoded.size == 0:
        raise ImagePreprocessingError(f"image file is empty: {path}")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise ImagePreprocessingError(f"image could not be decoded: {path}")
    return decoded, path


def _validate_array(image: np.ndarray) -> ImageArray:
    if not isinstance(image, np.ndarray):
        raise ImagePreprocessingError("image must be a path or NumPy array")
    if image.size == 0:
        raise ImagePreprocessingError("image array is empty")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ImagePreprocessingError(
            "image array must have shape (height, width, 3) or (height, width, 4)"
        )
    if image.dtype != np.uint8:
        raise ImagePreprocessingError("image array must use uint8 pixels")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def preprocess_image(
    image: ImageInput,
    image_size: int = Config.IMAGE_SIZE,
) -> PreprocessedImage:
    """Decode and convert an image into canonical 8-bit representations.

    Array inputs are interpreted as OpenCV-style BGR/BGRA arrays. Both training
    and inference must call this function to prevent preprocessing drift.
    """
    if isinstance(image, (str, Path)):
        decoded, source_path = _decode_path(image)
    else:
        decoded = _validate_array(image)
        source_path = None

    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 1:
        raise ImagePreprocessingError("image_size must be a positive integer")

    original_bgr = np.ascontiguousarray(decoded.copy())
    interpolation = (
        cv2.INTER_AREA
        if decoded.shape[0] > image_size or decoded.shape[1] > image_size
        else cv2.INTER_LINEAR
    )
    resized_bgr = cv2.resize(
        decoded,
        (image_size, image_size),
        interpolation=interpolation,
    )
    resized_bgr = np.ascontiguousarray(resized_bgr, dtype=np.uint8)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    grayscale = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    return PreprocessedImage(
        original_bgr=original_bgr,
        bgr=resized_bgr,
        rgb=resized_rgb,
        grayscale=grayscale,
        source_path=source_path,
    )
