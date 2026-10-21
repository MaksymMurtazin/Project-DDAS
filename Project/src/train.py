"""Навчання моделі: порівняння кандидатів, оцінка, калібрування інтервалу, збереження артефактів.

Запуск:
    python -m src.train --data data/raw/Housing.csv
    python -m src.train --synthetic            # без Kaggle-датасету
    python -m src.train --tune --n-iter 30     # тюнінг ансамблю XGBoost+LightGBM
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split

from src import config
from src.data import DataError, generate_synthetic_dataset, load_dataset
from src.features import build_schema
from src.modeling import (
    build_models,
    calibrate_interval,
    cross_validate_model,
    regression_metrics,
    tune_ensemble,
)

logger = logging.getLogger("train")


def _json_default(obj: Any) -> Any:
    """Серіалізація numpy-типів у JSON."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _atomic_dump(path: Path, writer) -> None:
    """Пише у тимчасовий файл і атомарно підміняє — API ніколи не прочитає напівзаписаний артефакт."""
    tmp_path = path.with_name(path.name + ".tmp")
    writer(tmp_path)
    os.replace(tmp_path, path)


def compute_feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series, seed: int) -> pd.DataFrame:
    """Permutation importance на сирих ознаках (зрозуміло бізнесу, враховує весь пайплайн)."""
    try:
        result = permutation_importance(
            model, X_test, y_test, scoring="neg_mean_absolute_error", n_repeats=10, random_state=seed, n_jobs=1
        )
    except Exception:  # noqa: BLE001 — другорядний крок, не повинен валити навчання
        logger.exception("Feature importance failed; continuing without it")
        return pd.DataFrame(columns=["feature", "importance", "std"])
    frame = pd.DataFrame(
        {"feature": X_test.columns, "importance": result.importances_mean, "std": result.importances_std}
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def run_training(
    df: pd.DataFrame,
    *,
    target: str = config.TARGET_COLUMN,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_STATE,
    tune: bool = False,
    n_iter: int = 20,
    final_model: str = "ensemble",
    fast: bool = False,
    models_dir: Path = config.MODELS_DIR,
    reports_dir: Path = config.REPORTS_DIR,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Повний цикл навчання. Повертає metadata (те саме, що зберігається у JSON)."""
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=[target])
    y = df[target].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    logger.info("Split: train=%d, test=%d, features=%d", len(X_train), len(X_test), X.shape[1])

    schema = build_schema(X_train)
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    models = build_models(random_state=seed, fast=fast)

    if tune:
        logger.info("Tuning ensemble with RandomizedSearchCV (n_iter=%d)...", n_iter)
        models["ensemble"] = tune_ensemble(models["ensemble"], X_train, y_train, cv, n_iter, seed)

    # --- Порівняння кандидатів: CV на train + оцінка на hold-out test ---------
    comparison: dict[str, dict[str, dict[str, float]]] = {}
    for name, model in models.items():
        logger.info("Evaluating '%s'...", name)
        cv_metrics = cross_validate_model(model, X_train, y_train, cv)
        model.fit(X_train, y_train)
        test_metrics = regression_metrics(y_test, model.predict(X_test))
        comparison[name] = {"cv": cv_metrics, "test": test_metrics}
        logger.info(
            "%-16s CV RMSE=%.0f | test RMSE=%.0f MAPE=%.1f%% R2=%.3f within±5%%=%.0f%% within±7%%=%.0f%%",
            name, cv_metrics["rmse"], test_metrics["rmse"], test_metrics["mape"] * 100,
            test_metrics["r2"], test_metrics["within_5pct"] * 100, test_metrics["within_7pct"] * 100,
        )

    # --- Вибір фінальної моделі ----------------------------------------------
    if final_model == "best":
        candidates = {k: v for k, v in comparison.items() if k != "baseline_median"}
        final_model = min(candidates, key=lambda k: candidates[k]["cv"]["rmse"])
    if final_model not in models:
        raise ValueError(f"Unknown final model '{final_model}'. Options: {list(models)} or 'best'.")
    model = models[final_model]  # вже навчена на X_train
    logger.info("Final model: %s", final_model)

    # --- Інтервал прогнозу, важливість ознак, перевірка покриття -------------
    interval = calibrate_interval(model, X_train, y_train, cv, config.INTERVAL_LEVEL)
    y_pred_test = model.predict(X_test)
    inside = (y_test.to_numpy() >= y_pred_test * interval["lower_ratio"]) & (
        y_test.to_numpy() <= y_pred_test * interval["upper_ratio"]
    )
    interval["empirical_test_coverage"] = float(inside.mean())
    logger.info(
        "Interval ratios [%.3f, %.3f]; empirical test coverage=%.0f%% (nominal %.0f%%)",
        interval["lower_ratio"], interval["upper_ratio"], inside.mean() * 100, interval["level"] * 100,
    )

    importance = compute_feature_importance(model, X_test, y_test, seed)
    if not importance.empty:
        importance.to_csv(reports_dir / "feature_importance.csv", index=False)

    if make_plots:
        try:
            from src.reporting import save_feature_importance_plot, save_pred_vs_actual_plot

            if not importance.empty:
                save_feature_importance_plot(importance, reports_dir / "feature_importance.png")
            save_pred_vs_actual_plot(y_test, y_pred_test, reports_dir / "pred_vs_actual.png")
        except Exception:  # noqa: BLE001 — графіки другорядні
            logger.exception("Could not save plots; continuing")

    # --- Збереження артефактів -----------------------------------------------
    now = datetime.now(timezone.utc)
    metadata: dict[str, Any] = {
        "model_version": now.strftime("%Y%m%d-%H%M%S"),
        "model_name": final_model,
        "trained_at": now.isoformat(),
        "target": target,
        "n_rows": {"train": int(len(X_train)), "test": int(len(X_test))},
        "schema": schema,
        "prediction_interval": interval,
        "metrics": {"test": comparison[final_model]["test"], "cv": comparison[final_model]["cv"]},
        "model_comparison": comparison,
        "feature_importance": importance.head(15).to_dict(orient="records"),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    model_path = models_dir / config.MODEL_FILENAME
    metadata_path = models_dir / config.METADATA_FILENAME
    _atomic_dump(model_path, lambda p: joblib.dump(model, p))
    _atomic_dump(
        metadata_path,
        lambda p: p.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"),
    )
    logger.info("Saved model -> %s", model_path)
    logger.info("Saved metadata -> %s", metadata_path)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the house price prediction model.")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH, help="CSV з даними (за замовч. data/raw/Housing.csv)")
    parser.add_argument("--target", default=config.TARGET_COLUMN, help="назва цільової колонки")
    parser.add_argument("--synthetic", action="store_true", help="використати синтетичний датасет")
    parser.add_argument("--tune", action="store_true", help="налаштувати гіперпараметри ансамблю")
    parser.add_argument("--n-iter", type=int, default=20, help="кількість ітерацій RandomizedSearchCV")
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument(
        "--final-model",
        default="ensemble",
        choices=["ensemble", "random_forest", "xgboost", "lightgbm", "best"],
        help="яку модель зберегти ('best' — найменший CV RMSE)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    config.setup_logging()
    args = parse_args(argv)
    try:
        if args.synthetic:
            logger.warning("Using SYNTHETIC data — metrics are not meaningful for real-world conclusions.")
            df = generate_synthetic_dataset(seed=args.seed)
            args.target = "price"
        else:
            df = load_dataset(args.data, args.target)
        metadata = run_training(
            df,
            target=args.target,
            test_size=args.test_size,
            seed=args.seed,
            tune=args.tune,
            n_iter=args.n_iter,
            final_model=args.final_model,
        )
    except DataError as exc:
        logger.error("Data error: %s", exc)
        return 2
    except (ValueError, OSError) as exc:
        logger.error("Training failed: %s", exc)
        return 1
    except Exception:  # noqa: BLE001 — верхній рівень CLI: логуємо traceback і виходимо з кодом ≠ 0
        logger.exception("Unexpected error during training")
        return 1

    test = metadata["metrics"]["test"]
    print(
        f"\nГотово: model={metadata['model_name']} version={metadata['model_version']}\n"
        f"  Test MAPE={test['mape']:.1%}  R2={test['r2']:.3f}  "
        f"у межах ±5%: {test['within_5pct']:.0%}  у межах ±7%: {test['within_7pct']:.0%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
