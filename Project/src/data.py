"""Завантаження, валідація й очищення датасету + генератор синтетичних даних."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_ROWS = 50  # менше — навчання не має сенсу
ID_LIKE_COLUMNS = {"id", "index", "unnamed: 0"}


class DataError(Exception):
    """Проблема з вхідними даними (немає файлу, немає target, замало рядків...)."""


def load_dataset(path: str | Path, target: str) -> pd.DataFrame:
    """Читає CSV і повертає очищений DataFrame."""
    path = Path(path)
    if not path.is_file():
        raise DataError(
            f"Dataset file not found: {path}. Download it from Kaggle "
            "(see data/README.md) or run training with --synthetic."
        )
    try:
        raw = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataError(f"Cannot read CSV '{path}': {exc}") from exc

    logger.info("Loaded %s: %d rows x %d columns", path.name, *raw.shape)
    return clean_dataset(raw, target)


def clean_dataset(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Базове очищення: дублікати, службові колонки, некоректний target."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if target not in df.columns:
        raise DataError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    # Службові колонки-ідентифікатори не несуть сигналу й шкодять моделі
    id_cols = [c for c in df.columns if c.lower() in ID_LIKE_COLUMNS and c != target]
    if id_cols:
        logger.info("Dropping id-like columns: %s", id_cols)
        df = df.drop(columns=id_cols)

    rows_before = len(df)
    df = df.drop_duplicates()

    # Target має бути додатним скінченним числом
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df[np.isfinite(df[target]) & (df[target] > 0)].copy()
    logger.info("Removed %d duplicate/invalid rows", rows_before - len(df))

    # Повністю порожні колонки видаляємо; bool -> int для єдиного представлення
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        logger.info("Dropping fully empty columns: %s", empty_cols)
        df = df.drop(columns=empty_cols)
    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    if df.shape[1] < 2:
        raise DataError("Dataset has no feature columns after cleaning.")
    if len(df) < MIN_ROWS:
        raise DataError(f"Too few rows after cleaning: {len(df)} (need at least {MIN_ROWS}).")

    return df.reset_index(drop=True)


def generate_synthetic_dataset(n_rows: int = 800, seed: int = 42) -> pd.DataFrame:
    """Синтетичний датасет зі схемою Kaggle 'Housing Prices' (Housing.csv).

    Потрібен для швидкої перевірки пайплайна без доступу до Kaggle, а також для
    юніт-тестів.
    """
    rng = np.random.default_rng(seed)

    area = np.clip(rng.lognormal(mean=8.5, sigma=0.35, size=n_rows).round(), 1500, 16000)
    bedrooms = rng.integers(1, 7, n_rows)
    bathrooms = rng.integers(1, 5, n_rows)
    stories = rng.integers(1, 5, n_rows)
    parking = rng.integers(0, 4, n_rows)
    flags = {
        name: (rng.random(n_rows) < p).astype(int)
        for name, p in {
            "mainroad": 0.85,
            "guestroom": 0.18,
            "basement": 0.35,
            "hotwaterheating": 0.05,
            "airconditioning": 0.32,
            "prefarea": 0.23,
        }.items()
    }
    furnishing = rng.choice(["furnished", "semi-furnished", "unfurnished"], n_rows, p=[0.25, 0.42, 0.33])

    base = (
        1.2e6
        + 550 * area
        + 6.5e5 * bathrooms
        + 2.5e5 * bedrooms
        + 3.5e5 * stories
        + 4.5e5 * parking
        + 7.0e5 * flags["airconditioning"]
        + 5.0e5 * flags["prefarea"]
        + 4.0e5 * flags["basement"]
        + 4.5e5 * flags["hotwaterheating"]
        + 3.0e5 * flags["guestroom"]
        + 3.5e5 * flags["mainroad"]
    )
    multiplier = pd.Series(furnishing).map({"furnished": 1.08, "semi-furnished": 1.0, "unfurnished": 0.93}).to_numpy()
    price = (base * multiplier * rng.lognormal(0.0, 0.12, n_rows)).round(-3)

    to_yes_no = lambda arr: np.where(arr == 1, "yes", "no")  # noqa: E731
    return pd.DataFrame(
        {
            "price": price.astype(int),
            "area": area.astype(int),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": to_yes_no(flags["mainroad"]),
            "guestroom": to_yes_no(flags["guestroom"]),
            "basement": to_yes_no(flags["basement"]),
            "hotwaterheating": to_yes_no(flags["hotwaterheating"]),
            "airconditioning": to_yes_no(flags["airconditioning"]),
            "parking": parking,
            "prefarea": to_yes_no(flags["prefarea"]),
            "furnishingstatus": furnishing,
        }
    )
