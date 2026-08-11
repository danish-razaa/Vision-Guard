"""Strict image upload validation and safe storage."""

from __future__ import annotations

import io
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage

from config import Config


class UploadValidationError(ValueError):
    """Raised when an uploaded file is unsafe or not a supported image."""


@dataclass(frozen=True)
class ValidatedUpload:
    data: bytes
    extension: str
    mime_type: str
    width: int
    height: int


FORMAT_DETAILS = {
    "JPEG": ("jpg", "image/jpeg"),
    "PNG": ("png", "image/png"),
}


def _read_limited(stream: BinaryIO, max_bytes: int) -> bytes:
    data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise UploadValidationError(f"image exceeds the {max_bytes // (1024 * 1024)} MB limit")
    if not data:
        raise UploadValidationError("no image data was uploaded")
    return data


def validate_image_upload(
    upload: FileStorage | None,
    max_bytes: int = Config.MAX_CONTENT_LENGTH,
    allowed_extensions: Iterable[str] = Config.ALLOWED_EXTENSIONS,
) -> ValidatedUpload:
    """Validate extension, actual format, full decoding, and pixel dimensions."""
    if upload is None or not upload.filename:
        raise UploadValidationError("select a JPG, JPEG, or PNG image")
    supplied_extension = Path(upload.filename).suffix.casefold().lstrip(".")
    allowed = {extension.casefold().lstrip(".") for extension in allowed_extensions}
    if supplied_extension not in allowed:
        raise UploadValidationError("only JPG, JPEG, and PNG files are accepted")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")

    data = _read_limited(upload.stream, max_bytes)
    try:
        with Image.open(io.BytesIO(data)) as probe:
            detected_format = probe.format
            width, height = probe.size
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise UploadValidationError("uploaded file is not a valid decodable image") from exc
    if detected_format not in FORMAT_DETAILS:
        raise UploadValidationError("decoded image format must be JPEG or PNG")
    canonical_extension, mime_type = FORMAT_DETAILS[detected_format]
    # A browser or download source may give a genuine JPEG a .png suffix (or
    # vice versa). Both the supplied extension and the detected format must be
    # independently allowed, but the detected bytes are authoritative. The
    # server stores the upload using this canonical detected extension.
    if width < 1 or height < 1:
        raise UploadValidationError("image dimensions are invalid")

    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise UploadValidationError("uploaded image failed full pixel decoding")
    return ValidatedUpload(data, canonical_extension, mime_type, width, height)


def save_validated_upload(upload: ValidatedUpload, folder: str | Path) -> Path:
    """Store validated bytes under a server-generated UUID filename."""
    destination = Path(folder).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{uuid.uuid4().hex}.{upload.extension}"
    path.write_bytes(upload.data)
    if path.parent != destination:
        raise UploadValidationError("unsafe upload destination")
    return path


def cleanup_old_files(
    folder: str | Path,
    max_age_seconds: int = 3600,
    name_prefixes: tuple[str, ...] | None = None,
) -> int:
    """Remove generated upload files older than the retention window."""
    directory = Path(folder)
    if not directory.is_dir():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for path in directory.iterdir():
        matches_scope = name_prefixes is None or path.name.startswith(name_prefixes)
        if (
            path.is_file()
            and path.name != ".gitkeep"
            and matches_scope
            and path.stat().st_mtime < cutoff
        ):
            path.unlink(missing_ok=True)
            removed += 1
    return removed
