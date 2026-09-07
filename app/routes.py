"""VisionGuard web routes."""

from __future__ import annotations

import json
import logging
import base64
import io
import tempfile
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, abort, current_app, jsonify, render_template, request, send_file, session, url_for
from PIL import Image

from app.model_registry import DEFAULT_MODEL_ID, selected_model
from app.validators import (
    UploadValidationError,
    save_validated_upload,
    validate_image_upload,
)
from config import BASE_DIR
from core.feature_comparison import compare_feature_vectors
from core.feature_extractor import extract_features
from core.prompt_perturbation import generate_prompt_conditioned_perturbation


LOGGER = logging.getLogger("visionguard.routes")
web = Blueprint("web", __name__)
PROMPT_LIMIT = 1000
STRENGTHS = {"very_low": 0.5, "low": 1.0, "medium": 2.0, "high": 4.0}


def _prompt_root() -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"]).resolve() / "prompt_lab"


def _cleanup_prompt_experiments(max_age: int = 3600) -> None:
    root = _prompt_root()
    root.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - max_age
    for candidate in root.iterdir():
        if candidate.is_dir() and candidate.stat().st_mtime < cutoff:
            shutil.rmtree(candidate, ignore_errors=True)


def _experiment(request_id: str) -> Path:
    if request_id != session.get("prompt_lab_request_id"):
        abort(404)
    try:
        parsed = uuid.UUID(request_id)
    except (ValueError, TypeError):
        abort(404)
    folder = (_prompt_root() / parsed.hex).resolve()
    if folder.parent != _prompt_root() or not folder.is_dir():
        abort(404)
    return folder


@web.get("/")
def home():
    return render_template("index.html")


@web.get("/scanner")
def scanner():
    return render_template(
        "scanner.html",
        model_available=current_app.extensions["visionguard_inference"] is not None,
        error=current_app.extensions.get("visionguard_model_error"),
        model_options=current_app.extensions.get("visionguard_model_options", []),
        selected_model_id=DEFAULT_MODEL_ID,
    )


@web.get("/prompt-lab")
def prompt_lab():
    return render_template(
        "prompt_lab.html",
        model_available=current_app.extensions["visionguard_inference"] is not None,
        model_options=current_app.extensions.get("visionguard_model_options", []),
        selected_model_id=DEFAULT_MODEL_ID,
    )


@web.post("/api/prompt-perturb")
def prompt_perturb():
    try:
        if "image" not in request.files:
            raise UploadValidationError("the multipart request did not contain an image field")
        prompt = request.form.get("prompt", "").strip()
        if not prompt:
            raise UploadValidationError("prompt must not be empty")
        if len(prompt) > PROMPT_LIMIT:
            raise UploadValidationError(f"prompt must be at most {PROMPT_LIMIT} characters")
        strength = request.form.get("strength", "")
        if strength not in STRENGTHS:
            raise UploadValidationError("strength must be very_low, low, medium, or high")
        validated = validate_image_upload(
            request.files["image"],
            max_bytes=int(current_app.config["MAX_CONTENT_LENGTH"]),
            allowed_extensions=current_app.config["ALLOWED_EXTENSIONS"],
        )
        _cleanup_prompt_experiments()
        old_id = session.get("prompt_lab_request_id")
        if old_id:
            old = _prompt_root() / str(old_id).replace("-", "")
            if old.parent == _prompt_root() and old.is_dir():
                shutil.rmtree(old, ignore_errors=True)
        request_id = str(uuid.uuid4())
        folder = _prompt_root() / uuid.UUID(request_id).hex
        folder.mkdir(parents=True)
        with Image.open(io.BytesIO(validated.data)) as opened:
            original = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        modified, _perturbation, metadata = generate_prompt_conditioned_perturbation(
            original, prompt, STRENGTHS[strength]
        )
        Image.fromarray(original).save(folder / "original.png", format="PNG")
        Image.fromarray(modified).save(folder / "modified.png", format="PNG")
        actual_delta = np.abs(modified.astype(np.int16) - original.astype(np.int16))
        amplified = np.clip(actual_delta * 16, 0, 255).astype(np.uint8)
        Image.fromarray(amplified).save(folder / "difference.png", format="PNG")
        record = {"request_id": request_id, "strength": strength, **metadata}
        (folder / "metadata.json").write_text(json.dumps(record), encoding="utf-8")
        session["prompt_lab_request_id"] = request_id
        return jsonify({
            "success": True,
            "request_id": request_id,
            "original_image_url": url_for("web.prompt_image", request_id=request_id, kind="original"),
            "modified_image_url": url_for("web.prompt_image", request_id=request_id, kind="modified"),
            "difference_image_url": url_for("web.prompt_image", request_id=request_id, kind="difference"),
            "perturbation": {
                "method": metadata["perturbation_method"], "epsilon": metadata["epsilon"],
                "seed": metadata["seed"], "prompt_hash": metadata["prompt_hash"],
            },
        })
    except UploadValidationError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception:
        LOGGER.exception("Prompt perturbation generation failed")
        return jsonify({"success": False, "error": "The controlled perturbation could not be generated."}), 500


@web.get("/api/prompt-image/<request_id>/<kind>")
def prompt_image(request_id: str, kind: str):
    if kind not in {"original", "modified", "difference"}:
        abort(404)
    return send_file(_experiment(request_id) / f"{kind}.png", mimetype="image/png", conditional=False)


@web.post("/api/compare-scan")
def compare_scan():
    services = current_app.extensions.get("visionguard_models", {})
    if not services:
        return jsonify({"success": False, "error": "Model has not been trained yet."}), 503
    payload = request.get_json(silent=True) or request.form
    request_id = payload.get("request_id", "")
    try:
        selected_model_id, service = selected_model(services, payload.get("model_id"))
        folder = _experiment(request_id)
        original_path, modified_path = folder / "original.png", folder / "modified.png"
        original_result = service.predict(original_path)
        modified_result = service.predict(modified_path)
        smoothing = getattr(service, "smoothing_config", None)
        original_features = extract_features(original_path, smoothing)
        modified_features = extract_features(modified_path, smoothing)
        deltas = compare_feature_vectors(original_features, modified_features)
        pair_service = current_app.extensions.get("visionguard_pair_detector")
        pair_detection = (
            pair_service.predict(original_features, modified_features)
            if pair_service is not None
            else None
        )
        risk_delta = modified_result["attack_probability"] - original_result["attack_probability"]
        if pair_detection is not None and pair_detection["detected"]:
            interpretation = (
                "The reference-aware model detected a prompt-conditioned change and "
                "raised concern for the modified image."
            )
            modified_assessment = "PROMPTED / SUSPICIOUS"
        elif original_result["status"] != "SUSPICIOUS" and modified_result["status"] == "SUSPICIOUS":
            interpretation = "The standalone detector raised concern for the modified image."
            modified_assessment = "SUSPICIOUS"
        elif original_result["status"] != "SUSPICIOUS" and modified_result["status"] != "SUSPICIOUS":
            interpretation = "Controlled perturbation was not flagged."
            modified_assessment = "VERIFIED/CLEAN"
        elif original_result["status"] == "SUSPICIOUS" and modified_result["status"] == "SUSPICIOUS":
            interpretation = "Baseline was already suspicious."
            modified_assessment = "SUSPICIOUS"
        else:
            interpretation = "Detector score reversed after modification; investigate this sample."
            modified_assessment = "REVIEW REQUIRED"
        score_direction = (
            "Detector score decreased after modification."
            if risk_delta < 0
            else "Detector score increased after modification."
            if risk_delta > 0
            else "Detector score was unchanged after modification."
        )
        interpretation = f"{score_direction} {interpretation}"
        if pair_detection is not None:
            interpretation += (
                f" Reference-aware Pair Detector: {pair_detection['status']} "
                f"({pair_detection['probability']:.1f}% probability; "
                f"{pair_detection['threshold']:.1f}% threshold)."
            )
        original_result["lab_assessment"] = original_result["status"]
        modified_result["lab_assessment"] = modified_assessment
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        session["prompt_lab_record"] = {
            "request_id": request_id, "prompt_hash": metadata["prompt_hash"],
            "epsilon": metadata["epsilon"], "method": metadata["perturbation_method"],
            "original_probability": original_result["attack_probability"],
            "modified_probability": modified_result["attack_probability"], "risk_delta": risk_delta,
        }
        for result in (original_result, modified_result):
            result.pop("xai_plot_path", None)
        return jsonify({"success": True, "request_id": request_id, "original": original_result,
                        "modified": modified_result, "risk_delta": risk_delta,
                        "selected_model_id": selected_model_id,
                        "interpretation": interpretation, "score_direction": score_direction,
                        "pair_detection": pair_detection,
                        "feature_deltas": deltas})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        if getattr(exc, "code", None) == 404:
            raise
        LOGGER.exception("Prompt lab comparison failed")
        return jsonify({"success": False, "error": "The experiment could not be scanned."}), 500


@web.post("/scan")
def scan():
    services = current_app.extensions.get("visionguard_models", {})
    model_options = current_app.extensions.get("visionguard_model_options", [])
    if not services:
        return render_template(
            "scanner.html",
            model_available=False,
            error="Model has not been trained yet.",
            model_options=model_options,
            selected_model_id=DEFAULT_MODEL_ID,
        ), 503

    try:
        selected_model_id, service = selected_model(services, request.form.get("model_id"))
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
    except (UploadValidationError, ValueError) as exc:
        return render_template(
            "scanner.html", error=str(exc), model_available=True,
            model_options=model_options,
            selected_model_id=request.form.get("model_id", DEFAULT_MODEL_ID),
        ), 400
    except Exception:
        LOGGER.exception("Image scan failed")
        return render_template(
            "scanner.html",
            error="The image could not be scanned. Please try another valid image.",
            model_available=True,
            model_options=model_options,
            selected_model_id=request.form.get("model_id", DEFAULT_MODEL_ID),
        ), 500

    return render_template(
        "result.html",
        result=result,
        image_url=image_url,
        xai_plot_url=xai_plot_url,
        selected_model_id=selected_model_id,
    )


@web.get("/metrics")
def metrics():
    metrics_path = BASE_DIR / "models" / "model_metrics.csv"
    metadata_path = BASE_DIR / "models" / "model_metadata.json"
    development_path = BASE_DIR / "models" / "candidate_all_model_results.json"
    development_metadata_path = BASE_DIR / "models" / "candidate_model_metadata.json"
    pair_path = BASE_DIR / "models" / "pair_detector_metadata.json"
    required = (metrics_path, metadata_path, development_path, development_metadata_path)
    if not all(path.is_file() for path in required):
        return render_template("metrics.html", production_rows=None)
    try:
        frame = pd.read_csv(metrics_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        production_rows = frame.to_dict(orient="records")
        development_results = json.loads(development_path.read_text(encoding="utf-8"))
        development_metadata = json.loads(
            development_metadata_path.read_text(encoding="utf-8")
        )
        development_rows = [
            {"model_name": name, "threshold": values["threshold"], **values["test"]}
            for name, values in development_results.items()
        ]
        pair_metadata = (
            json.loads(pair_path.read_text(encoding="utf-8"))
            if pair_path.is_file() else None
        )
    except Exception:
        LOGGER.exception("Could not read saved model metrics")
        return render_template("metrics.html", production_rows=None)
    generated = BASE_DIR / "static" / "generated"
    figure_names = [
        name
        for name in ("model_comparison.png", "roc_curves.png", "precision_recall_curves.png", "f1_comparison.png", "ablation_f1.png")
        if (generated / name).is_file()
    ]
    return render_template(
        "metrics.html", production_rows=production_rows,
        production_metadata=metadata, development_rows=development_rows,
        development_metadata=development_metadata, pair_metadata=pair_metadata,
        figures=figure_names,
    )


@web.get("/architecture")
def architecture():
    return render_template("architecture.html")
