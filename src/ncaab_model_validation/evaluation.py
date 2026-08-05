"""Expanding-season evaluation and date-clustered uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ncaab_model_validation.models import MODEL_NAMES, predict_fold

PRIMARY_METRIC: Final = "mean absolute error"
DEFAULT_TEST_SEASONS: Final = tuple(range(2019, 2027))
DEFAULT_BOOTSTRAP_REPETITIONS: Final = 5_000
DEFAULT_BOOTSTRAP_SEED: Final = 20_260_805
COMPARISONS: Final = (
    ("elo", "baseline"),
    ("ridge", "baseline"),
    ("boost", "baseline"),
    ("ridge", "elo"),
    ("boost", "elo"),
    ("boost", "ridge"),
)


@dataclass(frozen=True)
class EvaluationResult:
    """All deterministic tables produced by the public evaluation."""

    predictions: pl.DataFrame
    fold_metrics: pl.DataFrame
    pooled_metrics: pl.DataFrame
    comparisons: pl.DataFrame


def _metrics(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def clustered_mae_comparison(
    predictions: pl.DataFrame,
    candidate: str,
    reference: str,
    *,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Paired bootstrap by game date, stratified within test season."""

    if repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    scored = predictions.with_columns(
        (
            (pl.col("home_margin") - pl.col(reference)).abs()
            - (pl.col("home_margin") - pl.col(candidate)).abs()
        ).alias("mae_improvement")
    )
    rng = np.random.default_rng(seed)
    bootstrap_total = np.zeros(repetitions, dtype=np.float64)
    bootstrap_count = np.zeros(repetitions, dtype=np.int64)
    for season in sorted(scored["season"].unique().to_list()):
        clusters = (
            scored.filter(pl.col("season") == season)
            .group_by("game_date")
            .agg(
                pl.col("mae_improvement").sum().alias("total"),
                pl.len().alias("games"),
            )
            .sort("game_date")
        )
        totals = clusters["total"].to_numpy().astype(np.float64, copy=False)
        counts = clusters["games"].to_numpy().astype(np.int64, copy=False)
        draws = rng.integers(0, len(totals), size=(repetitions, len(totals)))
        bootstrap_total += totals[draws].sum(axis=1)
        bootstrap_count += counts[draws].sum(axis=1)
    distribution = bootstrap_total / bootstrap_count
    point = float(scored.select(pl.col("mae_improvement").mean()).item())
    low, high = np.quantile(distribution, [0.025, 0.975])
    return point, float(low), float(high)


def run_evaluation(
    features: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...] = DEFAULT_TEST_SEASONS,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> EvaluationResult:
    """Run expanding-season folds without refitting specifications to outcomes."""

    predictions: list[pl.DataFrame] = []
    fold_rows: list[dict[str, float | int | str]] = []
    for season in test_seasons:
        train = features.filter(pl.col("season") < season)
        test = features.filter(pl.col("season") == season)
        fold_predictions = predict_fold(train, test)
        actual = test["home_margin"].to_numpy().astype(np.float64, copy=False)
        for model in MODEL_NAMES:
            fold_rows.append(
                {
                    "season": season,
                    "games": test.height,
                    "model": model,
                    **_metrics(actual, fold_predictions[model]),
                }
            )
        predictions.append(
            test.select(
                "game_id",
                "game_datetime",
                "game_date",
                "season",
                "home_team",
                "away_team",
                "home_margin",
            ).with_columns([pl.Series(model, fold_predictions[model]) for model in MODEL_NAMES])
        )

    prediction_frame = pl.concat(predictions).sort(["game_datetime", "game_id"])
    actual = prediction_frame["home_margin"].to_numpy().astype(np.float64, copy=False)
    pooled_rows = []
    for model in MODEL_NAMES:
        predicted = prediction_frame[model].to_numpy().astype(np.float64, copy=False)
        pooled_rows.append(
            {
                "model": model,
                "games": prediction_frame.height,
                **_metrics(actual, predicted),
            }
        )

    fold_metrics = pl.DataFrame(fold_rows).sort(["season", "model"])
    comparison_rows: list[dict[str, float | int | str]] = []
    for candidate, reference in COMPARISONS:
        point, low, high = clustered_mae_comparison(
            prediction_frame,
            candidate,
            reference,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        )
        candidate_folds = fold_metrics.filter(pl.col("model") == candidate).select(
            "season", pl.col("mae").alias("candidate_mae")
        )
        reference_folds = fold_metrics.filter(pl.col("model") == reference).select(
            "season", pl.col("mae").alias("reference_mae")
        )
        paired = candidate_folds.join(reference_folds, on="season")
        comparison_rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "mae_improvement": point,
                "ci_95_low": low,
                "ci_95_high": high,
                "seasons_better": paired.filter(
                    pl.col("candidate_mae") < pl.col("reference_mae")
                ).height,
                "seasons_total": paired.height,
            }
        )
    return EvaluationResult(
        predictions=prediction_frame,
        fold_metrics=fold_metrics,
        pooled_metrics=pl.DataFrame(pooled_rows).sort("model"),
        comparisons=pl.DataFrame(comparison_rows),
    )
