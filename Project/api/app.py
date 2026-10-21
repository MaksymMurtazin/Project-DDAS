"""Flask REST API для оцінки вартості нерухомості.

Ендпоїнти:
    GET  /health                 — стан сервісу та моделі
    GET  /model/info             — схема входу, метрики, версія моделі
    POST /predict/price          — один об'єкт
    POST /predict/price/batch    — {"instances": [...]}

Запуск (dev):   python -m api.app
Запуск (prod):  gunicorn "api.app:create_app()"
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from flask import Flask, abort, jsonify, request
from werkzeug.exceptions import HTTPException

from src.config import MAX_BATCH_SIZE, setup_logging
from src.predictor import InputValidationError, ModelNotReadyError, PricePredictor

logger = logging.getLogger("api")


def create_app(predictor: PricePredictor | None = None) -> Flask:
    """Application factory: зручна для тестів і gunicorn."""
    setup_logging()
    app = Flask(__name__)
    app.json.sort_keys = False
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 МБ — захист від надто великих тіл запиту

    predictor = predictor or PricePredictor()
    try:
        predictor.load()
    except ModelNotReadyError as exc:
        # Сервіс стартує й без моделі: /health повертатиме 503, а запити спробують lazy-load
        logger.error("Model not loaded at startup: %s", exc)

    def get_json_object() -> dict[str, Any]:
        if not request.is_json:
            abort(415, description="Content-Type must be application/json.")
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise InputValidationError(["Request body must be a valid JSON object."])
        return payload

    # ------------------------------------------------------------------ #
    # Маршрути
    # ------------------------------------------------------------------ #
    @app.get("/health")
    def health():
        try:
            predictor.ensure_ready()
        except ModelNotReadyError as exc:
            logger.warning("Health check: %s", exc)
            return jsonify(status="degraded", model_loaded=False), 503
        return jsonify(status="ok", model_loaded=True, model_version=predictor.model_version)

    @app.get("/model/info")
    def model_info():
        return jsonify(predictor.model_info())

    @app.post("/predict/price")
    def predict_price():
        payload = get_json_object()
        started = time.perf_counter()
        result = predictor.predict_one(payload)
        logger.info("Prediction served in %.1f ms", (time.perf_counter() - started) * 1000)
        return jsonify(result)

    @app.post("/predict/price/batch")
    def predict_price_batch():
        payload = get_json_object()
        instances = payload.get("instances")
        if not isinstance(instances, list) or not instances:
            raise InputValidationError(["'instances' must be a non-empty list of objects."])
        if len(instances) > MAX_BATCH_SIZE:
            raise InputValidationError([f"Too many instances: {len(instances)} (max {MAX_BATCH_SIZE})."])
        predictions = predictor.predict_batch(instances)
        return jsonify(count=len(predictions), predictions=predictions)

    # ------------------------------------------------------------------ #
    # Обробка помилок: клієнтські -> 4xx, серверні -> 5xx (без витоку деталей)
    # ------------------------------------------------------------------ #
    @app.errorhandler(InputValidationError)
    def handle_validation_error(exc: InputValidationError):
        return jsonify(error="validation_error", details=exc.errors), 400

    @app.errorhandler(ModelNotReadyError)
    def handle_model_not_ready(exc: ModelNotReadyError):
        logger.error("Model not ready: %s", exc)
        return jsonify(error="model_not_ready", message="Model is not available yet."), 503

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        name = (exc.name or "error").lower().replace(" ", "_")
        return jsonify(error=name, message=exc.description), exc.code

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        logger.exception("Unhandled error")
        return jsonify(error="internal_error", message="Unexpected server error."), 500

    return app


if __name__ == "__main__":
    # Лише для розробки. host=127.0.0.1 за замовчуванням: debug-сервер не можна виставляти назовні.
    create_app().run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
