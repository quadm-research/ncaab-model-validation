from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl

from ncaab_model_validation.features import POINT_IN_TIME_FEATURES, build_features


def _games(last_home_score: int = 75) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "season": [2025, 2025, 2025],
            "game_datetime": [
                datetime(2024, 11, 1, 20, tzinfo=UTC),
                datetime(2024, 11, 3, 20, tzinfo=UTC),
                datetime(2024, 11, 5, 20, tzinfo=UTC),
            ],
            "game_date": [date(2024, 11, 1), date(2024, 11, 3), date(2024, 11, 5)],
            "home_id": ["a", "a", "b"],
            "home_team": ["A", "A", "B"],
            "away_id": ["b", "c", "a"],
            "away_team": ["B", "C", "A"],
            "home_score": [80, 77, last_home_score],
            "away_score": [70, 72, 74],
            "neutral_site": [False, False, True],
        }
    )


def test_current_outcome_cannot_change_current_features() -> None:
    original = build_features(_games(last_home_score=75))
    revised = build_features(_games(last_home_score=5))

    assert original.select(POINT_IN_TIME_FEATURES).equals(revised.select(POINT_IN_TIME_FEATURES))
    assert original["home_margin"].to_list() != revised["home_margin"].to_list()


def test_full_timestamp_orders_games_with_same_calendar_date() -> None:
    frame = (
        _games()
        .head(2)
        .with_columns(
            pl.lit(date(2024, 11, 1)).alias("game_date"),
            pl.Series(
                "game_datetime",
                [
                    datetime(2024, 11, 1, 23, tzinfo=UTC),
                    datetime(2024, 11, 1, 2, tzinfo=UTC),
                ],
            ),
            pl.Series("home_id", ["a", "b"]),
            pl.Series("away_id", ["c", "a"]),
        )
    )

    features = build_features(frame)

    first = features.row(0, named=True)
    second = features.row(1, named=True)
    assert first["game_id"] == "g2"
    assert first["away_games"] == 0.0
    assert second["game_id"] == "g1"
    assert second["home_games"] == 1.0
