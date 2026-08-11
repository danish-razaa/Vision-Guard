"""Flask route smoke tests using the real cached inference service."""

import io

from PIL import Image

from app import create_app


def _png_bytes() -> io.BytesIO:
    data = io.BytesIO()
    Image.new("RGB", (32, 32), color=(70, 100, 140)).save(data, format="PNG")
    data.seek(0)
    return data


def test_all_get_pages_render() -> None:
    app = create_app({"TESTING": True})
    client = app.test_client()
    for path in ("/", "/scanner", "/metrics", "/architecture"):
        response = client.get(path)
        assert response.status_code == 200
        assert b"VISIONGUARD" in response.data
        assert "no-store" in response.headers["Cache-Control"]


def test_valid_scan_renders_real_result(tmp_path) -> None:
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": tmp_path})
    response = app.test_client().post(
        "/scan",
        data={"image": (_png_bytes(), "test.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"ANALYSIS COMPLETE" in response.data
    assert b"Why did the model make this prediction?" in response.data
    assert b"data:image/png;base64," in response.data
    assert list(tmp_path.iterdir()) == []


def test_invalid_scan_returns_400() -> None:
    app = create_app({"TESTING": True})
    response = app.test_client().post(
        "/scan",
        data={"image": (io.BytesIO(b"bad"), "bad.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert b"not a valid decodable image" in response.data
