"""Tests for stored-threshold reusable inference."""

from pathlib import Path
import io

from PIL import Image
import pytest
from werkzeug.datastructures import FileStorage

from app.inference import get_inference_service, risk_level
from app.validators import save_validated_upload, validate_image_upload
from training.utils import binary_class_indices


class ReversedClassModel:
    classes_ = [1, 0]


def test_probability_class_indices_follow_semantic_labels() -> None:
    assert binary_class_indices(ReversedClassModel()) == (1, 0)


def test_probability_class_indices_reject_wrong_semantics() -> None:
    class WrongClassModel:
        classes_ = [0, 2]

    with pytest.raises(ValueError, match="semantic labels 0 and 1"):
        binary_class_indices(WrongClassModel())


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0, "LOW"), (29.999, "LOW"), (30, "MEDIUM"), (59.999, "MEDIUM"), (60, "HIGH"), (79.999, "HIGH"), (80, "CRITICAL"), (100, "CRITICAL")],
)
def test_risk_level_boundaries(probability: float, expected: str) -> None:
    assert risk_level(probability) == expected


def test_inference_service_is_loaded_once() -> None:
    assert get_inference_service() is get_inference_service()


def test_clean_and_manipulated_smoke_images_are_processed(tmp_path: Path) -> None:
    clean = next(Path("data/clean_normalized").glob("*.png"))
    attacked = next(Path("data/attacked").glob("*.png"))
    service = get_inference_service()

    clean_result = service.predict(clean, xai_plot_path=tmp_path / "clean_xai.png")
    attacked_result = service.predict(attacked, xai_plot_path=tmp_path / "attacked_xai.png")

    for result in (clean_result, attacked_result):
        assert result["status"] in {"VERIFIED/CLEAN", "SUSPICIOUS"}
        assert 0 <= result["attack_probability"] <= 100
        assert 0 <= result["clean_probability"] <= 100
        assert result["attack_probability"] + result["clean_probability"] == pytest.approx(100)
        assert result["threshold"] == pytest.approx(service.threshold * 100)
        assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert len(result["top_features"]) == 5
        assert result["xai_available"]
        expected_status = (
            "SUSPICIOUS"
            if result["attack_probability"] >= result["threshold"]
            else "VERIFIED/CLEAN"
        )
        assert result["status"] == expected_status

    assert (tmp_path / "clean_xai.png").is_file()
    assert (tmp_path / "attacked_xai.png").is_file()


def test_new_external_png_upload_reaches_feature_extraction_and_prediction(
    tmp_path: Path,
) -> None:
    """The image is generated in-test and has no dataset IDs or metadata rows."""
    buffer = io.BytesIO()
    Image.new("RGB", (73, 51), color=(17, 113, 209)).save(buffer, format="PNG")
    buffer.seek(0)
    upload = FileStorage(
        stream=buffer,
        filename="never_seen_external.png",
        content_type="application/octet-stream",
    )
    validated = validate_image_upload(upload)
    saved_path = save_validated_upload(validated, tmp_path / "uploads")

    assert "never_seen_external" not in saved_path.name
    result = get_inference_service().predict(
        saved_path, xai_plot_path=tmp_path / "external_xai.png"
    )

    assert result["status"] in {"VERIFIED/CLEAN", "SUSPICIOUS"}
    assert 0 <= result["attack_probability"] <= 100
    assert len(result["top_features"]) == 5
    assert (tmp_path / "external_xai.png").is_file()
