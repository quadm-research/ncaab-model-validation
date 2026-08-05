"""Frozen model specifications for expanding-season evaluation."""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ncaab_model_validation.features import POINT_IN_TIME_FEATURES

MODEL_NAMES: Final = ("baseline", "elo", "ridge", "boost")
ELO_FEATURES: Final = ("rating_diff", "neutral")
RIDGE_ALPHA: Final = 10.0
BOOST_SEED: Final = 42

MODEL_SPECIFICATIONS: Final = {
    "baseline": "training-only mean home margin, split by neutral-site status",
    "elo": "standardized Ridge(alpha=10) on pregame Elo difference and neutral flag",
    "ridge": "standardized Ridge(alpha=10) on the frozen point-in-time feature set",
    "boost": (
        "HistGradientBoostingRegressor(learning_rate=0.05, max_iter=200, "
        "max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=10, random_state=42)"
    ),
}


def _matrix(frame: pl.DataFrame, columns: tuple[str, ...]) -> NDArray[np.float64]:
    return frame.select(columns).to_numpy().astype(np.float64, copy=False)


def _target(frame: pl.DataFrame) -> NDArray[np.float64]:
    return frame["home_margin"].to_numpy().astype(np.float64, copy=False)


def _baseline(train: pl.DataFrame, test: pl.DataFrame) -> NDArray[np.float64]:
    overall = float(train.select(pl.col("home_margin").mean()).item())
    non_neutral_mean = (
        train.filter(pl.col("neutral") == 0.0).select(pl.col("home_margin").mean()).item()
    )
    neutral_mean = (
        train.filter(pl.col("neutral") == 1.0).select(pl.col("home_margin").mean()).item()
    )
    non_neutral = overall if non_neutral_mean is None else float(non_neutral_mean)
    neutral = overall if neutral_mean is None else float(neutral_mean)
    return np.where(test["neutral"].to_numpy() == 1.0, neutral, non_neutral).astype(np.float64)


def _ridge(
    train: pl.DataFrame,
    test: pl.DataFrame,
    columns: tuple[str, ...],
) -> NDArray[np.float64]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(_matrix(train, columns), _target(train))
    return np.asarray(model.predict(_matrix(test, columns)), dtype=np.float64)


def predict_fold(train: pl.DataFrame, test: pl.DataFrame) -> dict[str, NDArray[np.float64]]:
    """Fit each frozen specification on prior seasons and predict one season."""

    if train.is_empty() or test.is_empty():
        raise ValueError("both train and test frames must contain games")
    boost = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=10.0,
        random_state=BOOST_SEED,
    )
    boost.fit(_matrix(train, POINT_IN_TIME_FEATURES), _target(train))
    return {
        "baseline": _baseline(train, test),
        "elo": _ridge(train, test, ELO_FEATURES),
        "ridge": _ridge(train, test, POINT_IN_TIME_FEATURES),
        "boost": np.asarray(boost.predict(_matrix(test, POINT_IN_TIME_FEATURES)), dtype=np.float64),
    }
