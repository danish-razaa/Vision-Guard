"""VisionGuard Flask application factory."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask

from app.inference import InferenceConfigurationError, initialize_inference
from config import BASE_DIR, Config


def create_app(test_config: dict[str, object] | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.config.from_object(Config)
    app.config.update(SECRET_KEY="visionguard-local-development")
    if test_config:
        app.config.update(test_config)

    upload_folder = Path(app.config["UPLOAD_FOLDER"])
    upload_folder.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "static" / "generated").mkdir(parents=True, exist_ok=True)

    try:
        app.extensions["visionguard_inference"] = initialize_inference()
        app.extensions["visionguard_model_error"] = None
    except InferenceConfigurationError as exc:
        logging.getLogger("visionguard.app").warning("Model unavailable: %s", exc)
        app.extensions["visionguard_inference"] = None
        app.extensions["visionguard_model_error"] = str(exc)

    from app.routes import web

    app.register_blueprint(web)

    @app.after_request
    def disable_local_cache(response):
        """Prevent stale scanner/result pages and JavaScript during local use."""
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.errorhandler(413)
    def request_too_large(_error):
        from flask import render_template

        return render_template(
            "scanner.html",
            error="The uploaded image exceeds the configured size limit.",
            model_available=app.extensions["visionguard_inference"] is not None,
        ), 413

    return app
