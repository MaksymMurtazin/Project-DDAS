"""Побудова моделей, крос-валідація, тюнінг та калібрування інтервалу прогнозу."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.features import HousingFeatureEngineer

logger = logging.getLogger(__name__)

CV_SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "mape": "neg_mean_absolute_percentage_error",
    "r2": "r2",
}


# --------------------------------------------------------------------------- #
# Побудова пайплайнів
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Числові: median-imputer + StandardScaler; категоріальні: mode-imputer + OneHot.

    Колонки вибираються за dtype ПІСЛЯ feature engineering, тому код не залежить
    від конкретних назв колонок.
    """
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
    )


def _wrap(regressor: BaseEstimator) -> TransformedTargetRegressor:
    """Pipeline(features -> preprocess -> model) + log-перетворення цільової змінної.

    Ціни мають скошений розподіл, тому модель навчається на log1p(price),
    а прогнози автоматично повертаються в початковий масштаб (expm1).
    """
    pipeline = Pipeline(
        [
            ("features", HousingFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", regressor),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )


def _make_xgb(random_state: int, n_jobs: int, fast: bool) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=40 if fast else 500,
        max_depth=4,
        learning_rate=0.1 if fast else 0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
    )


def _make_lgbm(random_state: int, n_jobs: int, fast: bool) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=40 if fast else 500,
        learning_rate=0.1 if fast else 0.03,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,
    )


def build_models(random_state: int = 42, n_jobs: int = -1, fast: bool = False) -> dict[str, BaseEstimator]:
    """Кандидати: бейзлайн, Random Forest, XGBoost, LightGBM та їхній ансамбль.

    fast=True зменшує кількість дерев — лише для юніт-тестів.
    """
    random_forest = RandomForestRegressor(
        n_estimators=40 if fast else 300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    ensemble = VotingRegressor(
        [
            ("xgb", _make_xgb(random_state, n_jobs, fast)),
            ("lgbm", _make_lgbm(random_state, n_jobs, fast)),
        ]
    )
    return {
        "baseline_median": _wrap(DummyRegressor(strategy="median")),
        "random_forest": _wrap(random_forest),
        "xgboost": _wrap(_make_xgb(random_state, n_jobs, fast)),
        "lightgbm": _wrap(_make_lgbm(random_state, n_jobs, fast)),
        "ensemble": _wrap(ensemble),
    }


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #
def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """RMSE/MAE/MAPE/R² + частка прогнозів у межах ±5 % і ±7 %."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_true - y_pred) / np.abs(y_true)  # y_true > 0 гарантовано очищенням
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(ape.mean()),
        "median_ape": float(np.median(ape)),
        "r2": float(r2_score(y_true, y_pred)),
        "within_5pct": float((ape <= 0.05).mean()),
        "within_7pct": float((ape <= 0.07).mean()),
    }


def cross_validate_model(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, cv: Any) -> dict[str, float]:
    """K-fold крос-валідація; метрики — в масштабі цін (завдяки TransformedTargetRegressor)."""
    scores = cross_validate(model, X, y, cv=cv, scoring=CV_SCORING, n_jobs=1, error_score="raise")
    return {
        "rmse": float(-scores["test_rmse"].mean()),
        "rmse_std": float(scores["test_rmse"].std()),
        "mae": float(-scores["test_mae"].mean()),
        "mape": float(-scores["test_mape"].mean()),
        "r2": float(scores["test_r2"].mean()),
    }


# --------------------------------------------------------------------------- #
# Тюнінг і невизначеність
# --------------------------------------------------------------------------- #
def tuning_space() -> dict[str, Any]:
    """Простір пошуку для ансамблю XGBoost + LightGBM (імена параметрів — через вкладені префікси)."""
    prefix = "regressor__model__"
    return {
        f"{prefix}xgb__max_depth": randint(3, 8),
        f"{prefix}xgb__learning_rate": loguniform(0.01, 0.2),
        f"{prefix}xgb__n_estimators": randint(200, 1000),
        f"{prefix}xgb__subsample": uniform(0.6, 0.4),
        f"{prefix}xgb__colsample_bytree": uniform(0.6, 0.4),
        f"{prefix}lgbm__num_leaves": randint(8, 64),
        f"{prefix}lgbm__learning_rate": loguniform(0.01, 0.2),
        f"{prefix}lgbm__n_estimators": randint(200, 1000),
        f"{prefix}lgbm__min_child_samples": randint(5, 40),
        f"{prefix}lgbm__colsample_bytree": uniform(0.6, 0.4),
    }


def tune_ensemble(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, cv: Any, n_iter: int, random_state: int):
    """RandomizedSearchCV для ансамблю. Повертає найкращий (вже навчений) estimator.

    Увага: CV-метрики налаштованої моделі трохи оптимістичні (selection bias),
    тому чесною оцінкою вважайте метрики на hold-out тесті.
    """
    search = RandomizedSearchCV(
        model,
        param_distributions=tuning_space(),
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    logger.info("Best CV RMSE during tuning: %.2f", -search.best_score_)
    return search.best_estimator_


def calibrate_interval(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, cv: Any, level: float
) -> dict[str, float]:
    """Емпіричний інтервал прогнозу на основі out-of-fold залишків.

    Рахуємо відношення y_true / y_pred на крос-валідації й беруть квантилі
    (1-level)/2 та (1+level)/2. На проді: [pred * lower_ratio, pred * upper_ratio].
    Ширина інтервалу відображає реальну похибку моделі на цих даних.
    """
    oof_pred = cross_val_predict(clone(model), X, y, cv=cv, n_jobs=1)
    ratios = y.to_numpy(dtype=float) / np.clip(oof_pred, 1e-9, None)
    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(ratios, [alpha, 1.0 - alpha])
    return {
        "level": float(level),
        "lower_ratio": float(min(lower, 1.0)),  # інтервал завжди містить точковий прогноз
        "upper_ratio": float(max(upper, 1.0)),
    }
