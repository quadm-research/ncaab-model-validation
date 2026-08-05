"""Strictly point-in-time team-state features."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, cast

import polars as pl

POINT_IN_TIME_FEATURES: Final = (
    "rating_diff",
    "margin_ewm_diff",
    "last5_margin_diff",
    "home_off_ewm",
    "home_def_ewm",
    "away_off_ewm",
    "away_def_ewm",
    "rest_diff",
    "home_games",
    "away_games",
    "neutral",
)

ELO_K: Final = 24.0
ELO_HOME_ADVANTAGE: Final = 55.0
EWM_ALPHA: Final = 0.18
SEASON_RETENTION: Final = 0.60
DEFAULT_RATING: Final = 1500.0
DEFAULT_POINTS: Final = 70.0


@dataclass
class TeamState:
    """State available immediately before a team's next game."""

    rating: float = DEFAULT_RATING
    off_ewm: float = DEFAULT_POINTS
    def_ewm: float = DEFAULT_POINTS
    margin_ewm: float = 0.0
    last_margins: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    games: int = 0
    season: int | None = None
    last_datetime: datetime | None = None

    def advance_season(self, season: int) -> None:
        """Shrink persistent state and reset within-season counters."""

        if self.season is None:
            self.season = season
            return
        if season < self.season:
            raise ValueError("team state cannot move backward across seasons")
        if season == self.season:
            return
        gap = season - self.season
        retain = SEASON_RETENTION**gap
        self.rating = DEFAULT_RATING + retain * (self.rating - DEFAULT_RATING)
        self.off_ewm = DEFAULT_POINTS + retain * (self.off_ewm - DEFAULT_POINTS)
        self.def_ewm = DEFAULT_POINTS + retain * (self.def_ewm - DEFAULT_POINTS)
        self.margin_ewm *= retain
        self.last_margins.clear()
        self.games = 0
        self.last_datetime = None
        self.season = season


def _rest_days(state: TeamState, game_datetime: datetime) -> float:
    if state.last_datetime is None:
        return 7.0
    elapsed = (game_datetime - state.last_datetime).total_seconds() / 86_400.0
    if elapsed < 0:
        raise ValueError("games must be processed in timestamp order")
    return min(elapsed, 14.0)


def _recent_margin(state: TeamState) -> float:
    if not state.last_margins:
        return 0.0
    return sum(state.last_margins) / len(state.last_margins)


def _update_states(
    home: TeamState,
    away: TeamState,
    *,
    home_score: float,
    away_score: float,
    neutral: bool,
    game_datetime: datetime,
) -> None:
    margin = home_score - away_score
    home_advantage = 0.0 if neutral else ELO_HOME_ADVANTAGE
    expected_home = 1.0 / (1.0 + 10.0 ** (-((home.rating + home_advantage) - away.rating) / 400.0))
    actual_home = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
    change = ELO_K * (actual_home - expected_home)
    home.rating += change
    away.rating -= change

    home.off_ewm = (1.0 - EWM_ALPHA) * home.off_ewm + EWM_ALPHA * home_score
    home.def_ewm = (1.0 - EWM_ALPHA) * home.def_ewm + EWM_ALPHA * away_score
    home.margin_ewm = (1.0 - EWM_ALPHA) * home.margin_ewm + EWM_ALPHA * margin
    away.off_ewm = (1.0 - EWM_ALPHA) * away.off_ewm + EWM_ALPHA * away_score
    away.def_ewm = (1.0 - EWM_ALPHA) * away.def_ewm + EWM_ALPHA * home_score
    away.margin_ewm = (1.0 - EWM_ALPHA) * away.margin_ewm - EWM_ALPHA * margin
    home.last_margins.append(margin)
    away.last_margins.append(-margin)
    home.games += 1
    away.games += 1
    home.last_datetime = game_datetime
    away.last_datetime = game_datetime


def build_features(games: pl.DataFrame) -> pl.DataFrame:
    """Emit each feature row before updating either team with that game's result."""

    ordered = games.sort(["game_datetime", "game_id"])
    states: dict[str, TeamState] = {}
    rows: list[dict[str, object]] = []

    for game in ordered.iter_rows(named=True):
        season = int(game["season"])
        game_datetime = cast(datetime, game["game_datetime"])
        home_id = str(game["home_id"])
        away_id = str(game["away_id"])
        home = states.setdefault(home_id, TeamState())
        away = states.setdefault(away_id, TeamState())
        home.advance_season(season)
        away.advance_season(season)

        neutral = bool(game["neutral_site"])
        home_rest = _rest_days(home, game_datetime)
        away_rest = _rest_days(away, game_datetime)
        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        rows.append(
            {
                "game_id": game["game_id"],
                "game_datetime": game_datetime,
                "game_date": game["game_date"],
                "season": season,
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_margin": home_score - away_score,
                "rating_diff": (home.rating - away.rating) / 100.0,
                "margin_ewm_diff": home.margin_ewm - away.margin_ewm,
                "last5_margin_diff": _recent_margin(home) - _recent_margin(away),
                "home_off_ewm": home.off_ewm,
                "home_def_ewm": home.def_ewm,
                "away_off_ewm": away.off_ewm,
                "away_def_ewm": away.def_ewm,
                "rest_diff": home_rest - away_rest,
                "home_games": float(home.games),
                "away_games": float(away.games),
                "neutral": float(neutral),
            }
        )
        _update_states(
            home,
            away,
            home_score=home_score,
            away_score=away_score,
            neutral=neutral,
            game_datetime=game_datetime,
        )

    return pl.DataFrame(rows).sort(["game_datetime", "game_id"])
