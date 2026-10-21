"""Збереження графіків (feature importance, predicted vs actual)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без GUI — працює в Docker/CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)


def save_feature_importance_plot(importance: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    """Горизонтальна діаграма permutation importance."""
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 1.5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#3b6ea5")
    ax.set_xlabel("Зростання MAE після перемішування ознаки")
    ax.set_title("Permutation importance (hold-out)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_pred_vs_actual_plot(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> None:
    """Scatter «факт vs прогноз» з діагоналлю ідеального прогнозу."""
    y_true = np.asarray(y_true, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=18, color="#3b6ea5")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Фактична ціна")
    ax.set_ylabel("Прогноз")
    ax.set_title("Predicted vs Actual (hold-out)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
