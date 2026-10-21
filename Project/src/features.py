"""Інженерія ознак та опис схеми вхідних даних.

Головний принцип: ОДНА й та сама логіка перетворень використовується і при
навчанні, і в API. Тому feature engineering — це sklearn-трансформер усередині
Pipeline, який зберігається разом з моделлю.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.config import MAX_CATEGORIES_IN_SCHEMA

YES_NO_MAP = {"yes": 1.0, "no": 0.0}


def normalize_text(value: Any) -> Any:
    """Нормалізує категоріальне значення: trim + lower-case; bool -> yes/no."""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if pd.isna(value):
        return np.nan
    return str(value).strip().lower()


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Повертає копію DataFrame з нормалізованими нечисловими колонками."""
    out = df.copy()
    for col in out.select_dtypes(exclude=[np.number]).columns:
        out[col] = out[col].astype(object).map(normalize_text)
    return out


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Перетворення сирих ознак житла в модельні.

    * нормалізує текст (регістр, пробіли);
    * колонки зі значеннями yes/no (визначаються на train) -> 0/1;
    * додає похідні ознаки, якщо потрібні колонки присутні:
        - total_rooms     = bedrooms + bathrooms
        - area_per_room   = area / total_rooms
        - amenity_count   = кількість yes-прапорців (зручностей)

    Усі похідні ознаки обчислюються ЛИШЕ з вхідних колонок (без target),
    тому витоку даних (data leakage) немає.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "HousingFeatureEngineer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HousingFeatureEngineer expects a pandas DataFrame.")
        normalized = normalize_categoricals(X)
        text_cols = normalized.select_dtypes(exclude=[np.number]).columns
        self.binary_columns_ = [
            col
            for col in text_cols
            if 0 < normalized[col].dropna().nunique() and set(normalized[col].dropna().unique()) <= set(YES_NO_MAP)
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "binary_columns_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HousingFeatureEngineer expects a pandas DataFrame.")

        df = normalize_categoricals(X)
        for col in self.binary_columns_:
            if col in df.columns:
                df[col] = df[col].map(YES_NO_MAP).astype(float)

        columns = set(df.columns)
        if {"bedrooms", "bathrooms"} <= columns:
            df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
            if "area" in columns:
                # 0 кімнат -> NaN, далі заповнить SimpleImputer (без ділення на нуль)
                df["area_per_room"] = df["area"] / df["total_rooms"].replace(0, np.nan)

        present_flags = [c for c in self.binary_columns_ if c in df.columns]
        if present_flags:
            df["amenity_count"] = df[present_flags].sum(axis=1)
        return df


def build_schema(X: pd.DataFrame) -> dict[str, Any]:
    """Описує очікувані вхідні поля — зберігається в metadata і використовується API/UI."""
    normalized = normalize_categoricals(X)

    numeric: dict[str, dict[str, Any]] = {}
    for col in normalized.select_dtypes(include=[np.number]).columns:
        values = normalized[col].dropna()
        numeric[col] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "is_integer": bool(np.allclose(values, np.round(values))),
        }

    categorical: dict[str, list[str] | None] = {}
    for col in normalized.select_dtypes(exclude=[np.number]).columns:
        uniques = sorted(normalized[col].dropna().unique().tolist())
        # Для високої кардинальності список не зберігаємо (None = не валідувати)
        categorical[col] = uniques if len(uniques) <= MAX_CATEGORIES_IN_SCHEMA else None

    return {"feature_order": list(X.columns), "numeric": numeric, "categorical": categorical}
