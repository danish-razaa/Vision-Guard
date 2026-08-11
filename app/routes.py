"""VisionGuard web routes."""

from __future__ import annotations

import json
import logging
import base64
import tempfile
from pathlib import Path

import pandas as pd
from flask import Blueprint, current_app, render_template, request

from app.validators import (
    UploadValidationError,
    save_validated_upload,
    validate_image_upload,
)
from config import BASE_DIR


LOGGER = logging.getLogger("visionguard.routes")
web = Blueprint("web", __name__)


@web.get("/")
def home():
    return render_template("index.html")


@web.get("/scanner")
def scanner():
    return render_template(
        "scanner.html",
        model_available=current_app.extensions["visionguard_inference"] is not None,
        error=current_app.extensions.get("visionguard_model_error"),
    )


@web.post("/scan")
def scan():
    service = current_app.extensions["visionguard_inference"]
    if service is None:
        return render_template(
            "scanner.html",
            model_available=False,
            error="Model has not been trained yet.",
        ), 503

    try:
        if "image" not in request.files:
            raise UploadValidationError(
                "the multipart request did not contain an image field"
            )
        validated = validate_image_upload(
            request.files["image"],
            max_bytes=int(current_app.config["MAX_CONTENT_LENGTH"]),
            allowed_extensions=current_app.config["ALLOWED_EXTENSIONS"],
        )
        with tempfile.TemporaryDirectory(prefix="visionguard_scan_") as temporary:
            temporary_folder = Path(temporary)
            saved_path = save_validated_upload(validated, temporary_folder)
            plot_path = temporary_folder / f"xai_{saved_path.stem}.png"
            result = service.predict(saved_path, xai_plot_path=plot_path)
            image_url = (
                f"data:{validated.mime_type};base64,"
                + base64.b64encode(validated.data).decode("ascii")
            )
            xai_plot_url = None
            if result["xai_plot_path"] and plot_path.is_file():
                xai_plot_url = (
                    "data:image/png;base64,"
                    + base64.b64encode(plot_path.read_bytes()).decode("ascii")
                )
    except UploadValidationError as exc:
        return render_template(
            "scanner.html", error=str(exc), model_available=True
        ), 400
    except Exception:
        LOGGER.exception("Image scan failed")
        return render_template(
            "scanner.html",
            error="The image could not be scanned. Please try another valid image.",
            model_available=True,
        ), 500

    return render_template(
        "result.html",
        result=result,
        image_url=image_url,
        xai_plot_url=xai_plot_url,
    )


@web.get("/metrics")
def metrics():
    metrics_path = BASE_DIR / "models" / "model_metrics.csv"
    metadata_path = BASE_DIR / "models" / "model_metadata.json"
    if not metrics_path.is_file() or not metadata_path.is_file():
        return render_template("metrics.html", metrics=None, metadata=None, figures=[])
    try:
        frame = pd.read_csv(metrics_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = frame.to_dict(orient="records")
    except Exception:
        LOGGER.exception("Could not read saved model metrics")
        return render_template("metrics.html", metrics=None, metadata=None, figures=[])
    generated = BASE_DIR / "static" / "generated"
    figure_names = [
        name
        for name in ("model_comparison.png", "roc_curves.png", "precision_recall_curves.png", "f1_comparison.png", "ablation_f1.png")
        if (generated / name).is_file()
    ]
    return render_template(
        "metrics.html", metrics=rows, metadata=metadata, figures=figure_names
    )


@web.get("/architecture")
def architecture():
    return render_template("architecture.html")
