from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

import ncaab_model_validation.evaluation as evaluation
from ncaab_model_validation.evaluation import clustered_mae_comparison
from ncaab_model_validation.models import MODEL_NAMES


def _predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2024, 2024, 2024, 2025, 2025, 2025],
            "game_date": [
                date(2023, 11, 1),
                date(2023, 11, 1),
                date(2023, 11, 2),
                date(2024, 11, 1),
                date(2024, 11, 2),
                date(2024, 11, 2),
            ],
            "home_margin": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "candidate": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "reference": [4.0, 7.0, 6.0, 2.0, 8.0, 7.0],
        }
    )


def test_clustered_comparison_is_paired_stratified_and_deterministic() -> None:
    first = clustered_mae_comparison(
        _predictions(), "candidate", "reference", repetitions=500, seed=7
    )
    second = clustered_mae_comparison(
        _predictions(), "candidate", "reference", repetitions=500, seed=7
    )
    reversed_rows = clustered_mae_comparison(
        _predictions().reverse(), "candidate", "reference", repetitions=500, seed=7
    )

    assert first == second
    assert first == reversed_rows
    assert first[0] == pytest.approx(22.0 / 6.0)
    assert first[1] > 0.0
    assert first[2] >= first[1]


def test_walk_forward_training_seasons_are_strictly_earlier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[int, tuple[int, ...]]] = []

    def fake_predict_fold(train: pl.DataFrame, test: pl.DataFrame) -> dict[str, np.ndarray]:
        test_season = int(test.select(pl.col("season").first()).item())
        train_seasons = tuple(sorted(train["season"].unique().to_list()))
        seen.append((test_season, train_seasons))
        return {model: np.zeros(test.height) for model in MODEL_NAMES}

    monkeypatch.setattr(evaluation, "predict_fold", fake_predict_fold)
    features = pl.DataFrame(
        {
            "game_id": ["a", "b", "c"],
            "game_datetime": [
                "2017-11-01T00:00:00Z",
                "2018-11-01T00:00:00Z",
                "2019-11-01T00:00:00Z",
            ],
            "game_date": [date(2017, 11, 1), date(2018, 11, 1), date(2019, 11, 1)],
            "season": [2018, 2019, 2020],
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "B", "B"],
            "home_margin": [1.0, 2.0, 3.0],
        }
    )

    evaluation.run_evaluation(
        features,
        test_seasons=(2019, 2020),
        bootstrap_repetitions=100,
        bootstrap_seed=1,
    )

    assert seen == [(2019, (2018,)), (2020, (2018, 2019))]
