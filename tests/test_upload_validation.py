"""Security tests for actual image upload validation."""

import io
from pathlib import Path

from PIL import Image
import pytest
from werkzeug.datastructures import FileStorage

from app.validators import UploadValidationError, save_validated_upload, validate_image_upload


def _image_upload(
    filename: str = "sample.png",
    image_format: str = "PNG",
    content_type: str = "application/octet-stream",
) -> FileStorage:
    data = io.BytesIO()
    Image.new("RGB", (16, 12), color=(20, 80, 160)).save(data, format=image_format)
    data.seek(0)
    return FileStorage(stream=data, filename=filename, content_type=content_type)


@pytest.mark.parametrize(
    ("filename", "image_format", "expected_extension"),
    [
        ("external.jpg", "JPEG", "jpg"),
        ("external.jpeg", "JPEG", "jpg"),
        ("external.png", "PNG", "png"),
    ],
)
def test_external_jpg_jpeg_and_png_are_accepted_without_trusting_mime(
    filename: str, image_format: str, expected_extension: str
) -> None:
    validated = validate_image_upload(
        _image_upload(filename, image_format, content_type="application/octet-stream")
    )
    assert validated.extension == expected_extension


def test_valid_image_uses_detected_format_and_uuid_name(tmp_path: Path) -> None:
    validated = validate_image_upload(_image_upload())
    path = save_validated_upload(validated, tmp_path)
    assert validated.extension == "png"
    assert validated.mime_type == "image/png"
    assert validated.width == 16 and validated.height == 12
    assert path.parent == tmp_path.resolve()
    assert path.suffix == ".png"
    assert path.name != "sample.png"
    assert len(path.stem) == 32


def test_path_traversal_filename_is_never_used(tmp_path: Path) -> None:
    path = save_validated_upload(
        validate_image_upload(_image_upload("../../escape.png")), tmp_path
    )
    assert path.parent == tmp_path.resolve()
    assert "escape" not in path.name


def test_corrupt_spoofed_and_oversized_uploads_are_rejected() -> None:
    corrupt = FileStorage(stream=io.BytesIO(b"not an image"), filename="fake.png")
    with pytest.raises(UploadValidationError, match="valid decodable"):
        validate_image_upload(corrupt)
    with pytest.raises(UploadValidationError, match="exceeds"):
        validate_image_upload(_image_upload(), max_bytes=10)


def test_allowed_extension_with_different_supported_encoding_is_canonicalized(
    tmp_path: Path,
) -> None:
    # Matches real-world downloads like apple.PNG whose bytes are JPEG.
    validated = validate_image_upload(_image_upload("apple.PNG", "JPEG"))
    saved = save_validated_upload(validated, tmp_path)
    assert validated.extension == "jpg"
    assert validated.mime_type == "image/jpeg"
    assert saved.suffix == ".jpg"


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(UploadValidationError, match="only JPG"):
        validate_image_upload(_image_upload("external.gif", "PNG"))
