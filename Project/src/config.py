"""Централізована конфігурація проєкту: шляхи, константи, логування.

Усі шляхи можна перевизначити змінними оточення — це зручно для Docker.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Шляхи ------------------------------------------------------------------
DATA_PATH = Path(os.getenv("DATA_PATH", BASE_DIR / "data" / "raw" / "Housing.csv"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", BASE_DIR / "reports"))
MODEL_FILENAME = "house_price_model.joblib"
METADATA_FILENAME = "model_metadata.json"

# --- Параметри навчання -----------------------------------------------------
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "price")
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
INTERVAL_LEVEL = 0.90  # номінальне покриття інтервалу прогнозу (90 %)

# --- Параметри сервісу ------------------------------------------------------
MAX_BATCH_SIZE = 1000  # максимум об'єктів в одному batch-запиті
MAX_CATEGORIES_IN_SCHEMA = 200  # якщо унікальних категорій більше — не валідуємо список


def setup_logging(level: str | None = None) -> None:
    """Налаштовує єдиний формат логів для навчання та API."""
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
