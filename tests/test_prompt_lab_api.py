"""Prompt Injection Lab API validation and detector-reuse tests."""

import io

from PIL import Image

from app import create_app


def image_bytes(format_name: str) -> io.BytesIO:
    data = io.BytesIO()
    Image.new("RGB", (24, 24), (80, 110, 140)).save(data, format=format_name)
    data.seek(0)
    return data


def generate(client, filename="sample.png", format_name="PNG", prompt="research text", strength="low"):
    return client.post("/api/prompt-perturb", data={"image": (image_bytes(format_name), filename), "prompt": prompt, "strength": strength}, content_type="multipart/form-data")


def test_lab_page_and_validation(tmp_path) -> None:
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": tmp_path, "SECRET_KEY": "test"})
    client = app.test_client()
    assert client.get("/prompt-lab").status_code == 200
    assert generate(client, prompt="").status_code == 400
    assert generate(client, strength="unlimited").status_code == 400
    response = client.post("/api/prompt-perturb", data={"image": (io.BytesIO(b"bad"), "x.png"), "prompt": "safe", "strength": "low"}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_jpg_jpeg_png_and_new_request_identity(tmp_path) -> None:
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": tmp_path, "SECRET_KEY": "test"})
    client = app.test_client()
    ids = []
    for filename, fmt in (("a.jpg", "JPEG"), ("b.jpeg", "JPEG"), ("c.png", "PNG")):
        response = generate(client, filename, fmt)
        assert response.status_code == 200
        ids.append(response.get_json()["request_id"])
    assert len(set(ids)) == 3
    assert client.post("/api/compare-scan", json={"request_id": ids[0]}).status_code == 404


def test_compare_uses_configured_existing_detector(tmp_path) -> None:
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": tmp_path, "SECRET_KEY": "test"})
    service = app.extensions["visionguard_inference"]
    response = generate(app.test_client())
    assert response.status_code == 200
    request_id = response.get_json()["request_id"]
    compared = app.test_client()  # a different browser session must not access it
    assert compared.post("/api/compare-scan", json={"request_id": request_id}).status_code == 404
    assert service is app.extensions["visionguard_inference"]


def test_prompt_is_data_not_code(tmp_path) -> None:
    marker = tmp_path / "executed"
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": tmp_path / "uploads", "SECRET_KEY": "test"})
    response = generate(app.test_client(), prompt=f"__import__('pathlib').Path('{marker}').touch()")
    assert response.status_code == 200
    assert not marker.exists()
