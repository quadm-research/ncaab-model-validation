"""Deterministic, publication-ready result charts."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import polars as pl
from matplotlib import pyplot as plt

from ncaab_model_validation.models import MODEL_NAMES

COLORS: Final = {
    "baseline": "#8C96A3",
    "elo": "#55A5A6",
    "ridge": "#4C78A8",
    "boost": "#D97732",
}
LABELS: Final = {
    "baseline": "Venue mean",
    "elo": "Elo-only ridge",
    "ridge": "Full ridge",
    "boost": "Gradient boost",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "svg.hashsalt": "ncaab-model-validation",
        }
    )


def plot_fold_mae(fold_metrics: pl.DataFrame, path: Path) -> None:
    """Plot out-of-sample MAE for each expanding-season fold."""

    _style()
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    for model in MODEL_NAMES:
        model_rows = fold_metrics.filter(pl.col("model") == model).sort("season")
        axis.plot(
            model_rows["season"].to_list(),
            model_rows["mae"].to_list(),
            color=COLORS[model],
            linewidth=2.1,
            marker="o",
            markersize=4.5,
            label=LABELS[model],
        )
    axis.set_title("Walk-forward error by held-out season")
    axis.set_xlabel("Held-out season")
    axis.set_ylabel("Mean absolute error (points, lower is better)")
    axis.set_xticks(sorted(fold_metrics["season"].unique().to_list()))
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.8)
    axis.legend(frameon=False, ncols=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", metadata={"Date": None, "Creator": "QuadM Research"})
    plt.close(figure)


def plot_pooled_mae(pooled_metrics: pl.DataFrame, path: Path) -> None:
    """Plot pooled out-of-sample MAE in the declared model order."""

    _style()
    rows = [
        pooled_metrics.filter(pl.col("model") == model).row(0, named=True) for model in MODEL_NAMES
    ]
    values = [float(row["mae"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    bars = axis.bar(
        [LABELS[model] for model in MODEL_NAMES],
        values,
        color=[COLORS[model] for model in MODEL_NAMES],
        width=0.68,
    )
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.08,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_title("Pooled walk-forward error, 2019-2026")
    axis.set_ylabel("Mean absolute error (points, lower is better)")
    axis.set_ylim(0.0, max(values) * 1.16)
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.8)
    axis.tick_params(axis="x", rotation=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", metadata={"Date": None, "Creator": "QuadM Research"})
    plt.close(figure)
