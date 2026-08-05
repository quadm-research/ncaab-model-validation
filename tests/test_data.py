from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from ncaab_model_validation.data import normalize_sources, validate_games


def _source_frame() -> pl.DataFrame:
    eastern = ZoneInfo("America/New_York")
    return pl.DataFrame(
        {
            "game_id": ["final", "forfeit", "inactive"],
            "season": [2023, 2023, 2023],
            "game_date_time": [
                datetime(2023, 1, 1, 12, tzinfo=eastern),
                datetime(2023, 1, 2, 12, tzinfo=eastern),
                datetime(2023, 1, 3, 12, tzinfo=eastern),
            ],
            "game_date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "home_id": ["a", "c", "e"],
            "home_display_name": ["A", "C", "E"],
            "away_id": ["b", "d", "f"],
            "away_display_name": ["B", "D", "F"],
            "home_score": [80, 2, 70],
            "away_score": [70, 0, 60],
            "neutral_site": [False, False, False],
            "status_type_name": ["STATUS_FINAL", "STATUS_FORFEIT", "STATUS_FINAL"],
            "status_type_completed": [True, True, True],
            "home_is_active": [True, True, True],
            "away_is_active": [True, True, False],
            "away_non_div1_team": [False, False, False],
        }
    )


def test_normalization_excludes_forfeits_and_inactive_teams(tmp_path: Path) -> None:
    source = tmp_path / "mbb_schedule_2023.parquet"
    _source_frame().write_parquet(source)

    games, report = normalize_sources([source])

    assert games["game_id"].to_list() == ["final"]
    assert report.raw_rows == 3
    assert report.completed_rows == 3
    assert report.forfeit_rows == 1
    assert report.final_rows == 2
    assert report.inactive_team_rows == 1
    assert report.eligible_rows == 1


def test_validation_rejects_simultaneous_team_appearances(tmp_path: Path) -> None:
    source = tmp_path / "mbb_schedule_2023.parquet"
    frame = _source_frame().filter(pl.col("game_id") == "final")
    duplicate_time = frame.with_columns(
        pl.lit("second").alias("game_id"),
        pl.lit("c").alias("home_id"),
        pl.lit("A").alias("away_id"),
    )
    pl.concat([frame, duplicate_time]).write_parquet(source)
    games, _ = normalize_sources([source])
    broken = games.with_columns(
        pl.when(pl.col("game_id") == "second")
        .then(pl.lit("a"))
        .otherwise(pl.col("away_id"))
        .alias("away_id")
    )

    try:
        validate_games(broken)
    except ValueError as error:
        assert "same source timestamp" in str(error)
    else:
        raise AssertionError("simultaneous team appearances should fail validation")
