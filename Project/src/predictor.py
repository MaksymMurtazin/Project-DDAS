"""Сервісний шар: завантаження артефактів, валідація вхідних даних, прогнозування.

Відокремлений від Flask, щоб його можна було тестувати й перевикористовувати
(CLI, batch-job, інші фреймворки).
"""
from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import METADATA_FILENAME, MODEL_FILENAME, MODELS_DIR
from src.features import normalize_text

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Артефакти моделі відсутні або пошкоджені."""


class InputValidationError(ValueError):
    """Некоректні вхідні дані клієнта (-> HTTP 400). Містить список проблем."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class PricePredictor:
    """Обгортка над навченим Pipeline + metadata зі схемою та інтервалом."""

    def __init__(self, model_path: str | Path | None = None, metadata_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else MODELS_DIR / MODEL_FILENAME
        self.metadata_path = Path(metadata_path) if metadata_path else MODELS_DIR / METADATA_FILENAME
        self._model: Any = None
        self._metadata: dict[str, Any] | None = None
        self._lock = threading.Lock()  # захист від одночасного lazy-load у кількох потоках

    # ------------------------------------------------------------------ #
    # Життєвий цикл
    # ------------------------------------------------------------------ #
    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._metadata is not None

    @property
    def model_version(self) -> str | None:
        return self._metadata.get("model_version") if self._metadata else None

    def load(self) -> None:
        """Завантажує модель і metadata. Кидає ModelNotReadyError, якщо не вдалося."""
        with self._lock:
            if not self.model_path.is_file() or not self.metadata_path.is_file():
                raise ModelNotReadyError(
                    f"Model artifacts not found ({self.model_path}, {self.metadata_path}). "
                    "Run `python -m src.train` first."
                )
            try:
                model = joblib.load(self.model_path)
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 — pickle може впасти з будь-яким типом помилки
                raise ModelNotReadyError(f"Failed to load model artifacts: {exc}") from exc

            trained_with = metadata.get("environment", {}).get("scikit_learn")
            if trained_with and trained_with != sklearn.__version__:
                logger.warning(
                    "scikit-learn version mismatch: trained with %s, serving with %s",
                    trained_with, sklearn.__version__,
                )
            self._model, self._metadata = model, metadata
            logger.info("Model loaded: %s (version %s)", metadata.get("model_name"), metadata.get("model_version"))

    def ensure_ready(self) -> None:
        """Lazy-load: дозволяє API стартувати до того, як модель натреновано."""
        if not self.is_ready:
            self.load()

    def model_info(self) -> dict[str, Any]:
        """Публічна частина metadata (для /model/info та UI)."""
        self.ensure_ready()
        meta = self._metadata or {}
        keys = (
            "model_version", "model_name", "trained_at", "target", "n_rows", "schema",
            "prediction_interval", "metrics", "model_comparison", "feature_importance",
        )
        return {key: meta.get(key) for key in keys}

    # ------------------------------------------------------------------ #
    # Валідація
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_number(value: Any) -> float | None:
        """Перетворює значення на скінченне число або повертає None."""
        if isinstance(value, (bool, np.bool_)):
            return None  # True/False — не число
        if isinstance(value, str):
            try:
                value = float(value.strip())
            except ValueError:
                return None
        elif not isinstance(value, (int, float, np.integer, np.floating)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    def _validate_record(self, record: Any, prefix: str) -> tuple[dict[str, Any], list[str], list[str]]:
        """Повертає (очищений запис, помилки, попередження)."""
        if not isinstance(record, dict):
            return {}, [f"{prefix}must be a JSON object"], []

        schema = (self._metadata or {})["schema"]
        numeric: dict[str, dict[str, Any]] = schema["numeric"]
        categorical: dict[str, list[str] | None] = schema["categorical"]
        expected = set(numeric) | set(categorical)

        errors: list[str] = []
        warnings: list[str] = []

        missing = sorted(expected - record.keys())
        if missing:
            errors.append(f"{prefix}missing required fields: {missing}")
        extra = sorted(set(record) - expected)
        if extra:
            warnings.append(f"{prefix}ignored unknown fields: {extra}")

        clean: dict[str, Any] = {}
        for name, stats in numeric.items():
            if name not in record:
                continue
            number = self._to_number(record[name])
            if number is None:
                errors.append(f"{prefix}'{name}' must be a finite number, got {record[name]!r}")
                continue
            if not stats["min"] <= number <= stats["max"]:
                warnings.append(
                    f"{prefix}'{name}'={number:g} is outside the training range "
                    f"[{stats['min']:g}, {stats['max']:g}]; the estimate may be unreliable"
                )
            clean[name] = number

        for name, allowed in categorical.items():
            if name not in record:
                continue
            raw = record[name]
            if raw is None or isinstance(raw, (dict, list, tuple, set)):
                errors.append(f"{prefix}'{name}' must be a string, got {raw!r}")
                continue
            value = normalize_text(raw)
            if allowed is not None and value not in allowed:
                errors.append(f"{prefix}'{name}' must be one of {allowed[:20]}, got {raw!r}")
                continue
            clean[name] = value

        return clean, errors, warnings

    # ------------------------------------------------------------------ #
    # Прогнозування
    # ------------------------------------------------------------------ #
    def predict_batch(self, records: list[Any]) -> list[dict[str, Any]]:
        """Валідує й прогнозує список об'єктів. Усі помилки валідації збираються разом."""
        self.ensure_ready()
        metadata = self._metadata or {}

        cleaned: list[dict[str, Any]] = []
        all_errors: list[str] = []
        all_warnings: list[list[str]] = []
        for idx, record in enumerate(records):
            prefix = f"instances[{idx}]: " if len(records) > 1 else ""
            clean, errors, warnings = self._validate_record(record, prefix)
            cleaned.append(clean)
            all_errors.extend(errors)
            all_warnings.append(warnings)
        if all_errors:
            raise InputValidationError(all_errors)

        # Колонки в тому самому порядку, що й при навчанні
        frame = pd.DataFrame(cleaned, columns=metadata["schema"]["feature_order"])
        predictions = np.asarray(self._model.predict(frame), dtype=float)
        if not np.all(np.isfinite(predictions)):
            raise RuntimeError("Model produced a non-finite prediction.")

        interval = metadata["prediction_interval"]
        results = []
        for pred, warnings in zip(predictions, all_warnings):
            results.append(
                {
                    "predicted_price": round(float(pred), 2),
                    "price_range": {
                        "min": round(float(pred * interval["lower_ratio"]), 2),
                        "max": round(float(pred * interval["upper_ratio"]), 2),
                        "confidence_level": interval["level"],
                    },
                    "model_version": metadata.get("model_version"),
                    "warnings": warnings,
                }
            )
        return results

    def predict_one(self, record: Any) -> dict[str, Any]:
        return self.predict_batch([record])[0]
