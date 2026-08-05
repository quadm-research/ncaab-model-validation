"""Normalize and validate the public NCAA basketball source snapshot."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.request import Request, urlopen

import polars as pl

SOURCE_SNAPSHOT_DATE: Final = "2026-08-05"
SOURCE_SEASONS: Final = tuple(range(2015, 2027))
SOURCE_TAG: Final = "espn_mens_college_basketball_schedules"
SOURCE_URL_TEMPLATE: Final = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    f"{SOURCE_TAG}/mbb_schedule_{{season}}.parquet"
)

GAME_COLUMNS: Final = (
    "game_id",
    "season",
    "game_datetime",
    "game_date",
    "home_id",
    "home_team",
    "away_id",
    "away_team",
    "home_score",
    "away_score",
    "neutral_site",
)


@dataclass(frozen=True)
class DataBuildReport:
    """Auditable row counts for the sequential eligibility filters."""

    raw_rows: int
    completed_rows: int
    forfeit_rows: int
    final_rows: int
    missing_core_rows: int
    inactive_team_rows: int
    non_division_one_rows: int
    duplicate_game_rows: int
    eligible_rows: int
    first_season: int
    last_season: int


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required_source_columns() -> set[str]:
    return {
        "game_id",
        "season",
        "game_date_time",
        "game_date",
        "home_id",
        "home_display_name",
        "away_id",
        "away_display_name",
        "home_score",
        "away_score",
        "neutral_site",
        "status_type_name",
        "status_type_completed",
        "home_is_active",
        "away_is_active",
    }


def _read_source(path: Path) -> pl.DataFrame:
    schema = pl.read_parquet_schema(path)
    missing = _required_source_columns() - set(schema)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path.name} is missing required columns: {missing_text}")

    expressions: list[pl.Expr] = [
        pl.col("game_id").cast(pl.String),
        pl.col("season").cast(pl.Int32),
        pl.col("game_date_time").dt.convert_time_zone("UTC").alias("game_datetime"),
        pl.col("game_date").cast(pl.Date),
        pl.col("home_id").cast(pl.String),
        pl.col("home_display_name").cast(pl.String).alias("home_team"),
        pl.col("away_id").cast(pl.String),
        pl.col("away_display_name").cast(pl.String).alias("away_team"),
        pl.col("home_score").cast(pl.Int16, strict=False),
        pl.col("away_score").cast(pl.Int16, strict=False),
        pl.col("neutral_site").fill_null(False).cast(pl.Boolean),
        pl.col("status_type_name").cast(pl.String),
        pl.col("status_type_completed").fill_null(False).cast(pl.Boolean),
        pl.col("home_is_active").fill_null(False).cast(pl.Boolean),
        pl.col("away_is_active").fill_null(False).cast(pl.Boolean),
    ]
    if "away_non_div1_team" in schema:
        expressions.append(
            pl.col("away_non_div1_team")
            .fill_null(False)
            .cast(pl.Boolean)
            .alias("away_non_division_one")
        )
    else:
        expressions.append(pl.lit(False).alias("away_non_division_one"))
    return pl.read_parquet(path).select(expressions)


def normalize_sources(paths: list[Path]) -> tuple[pl.DataFrame, DataBuildReport]:
    """Apply frozen eligibility rules and return the normalized game table."""

    if not paths:
        raise ValueError("at least one source parquet file is required")
    source = pl.concat([_read_source(path) for path in sorted(paths)], how="vertical")
    raw_rows = source.height
    completed_rows = source.filter(pl.col("status_type_completed")).height
    forfeit_rows = source.filter(
        pl.col("status_type_completed") & (pl.col("status_type_name") == "STATUS_FORFEIT")
    ).height

    final = source.filter(pl.col("status_type_name") == "STATUS_FINAL")
    final_rows = final.height
    core_valid = (
        pl.col("game_id").is_not_null()
        & pl.col("game_datetime").is_not_null()
        & pl.col("game_date").is_not_null()
        & pl.col("home_id").is_not_null()
        & pl.col("away_id").is_not_null()
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    missing_core_rows = final.filter(~core_valid).height
    core = final.filter(core_valid)

    active = pl.col("home_is_active") & pl.col("away_is_active")
    inactive_team_rows = core.filter(~active).height
    active_games = core.filter(active)

    division_one = ~pl.col("away_non_division_one")
    non_division_one_rows = active_games.filter(~division_one).height
    eligible = active_games.filter(division_one)

    duplicate_game_rows = eligible.height - eligible.unique("game_id", keep="first").height
    games = (
        eligible.unique("game_id", keep="first")
        .select(GAME_COLUMNS)
        .sort(["game_datetime", "game_id"])
    )
    validate_games(games)
    report = DataBuildReport(
        raw_rows=raw_rows,
        completed_rows=completed_rows,
        forfeit_rows=forfeit_rows,
        final_rows=final_rows,
        missing_core_rows=missing_core_rows,
        inactive_team_rows=inactive_team_rows,
        non_division_one_rows=non_division_one_rows,
        duplicate_game_rows=duplicate_game_rows,
        eligible_rows=games.height,
        first_season=int(games.select(pl.col("season").min()).item()),
        last_season=int(games.select(pl.col("season").max()).item()),
    )
    return games, report


def validate_games(games: pl.DataFrame) -> None:
    """Fail closed on conditions that would make temporal ordering ambiguous."""

    missing = set(GAME_COLUMNS) - set(games.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"normalized games are missing columns: {missing_text}")
    if games.select(GAME_COLUMNS).null_count().to_numpy().sum() != 0:
        raise ValueError("normalized games contain null values")
    if games["game_id"].n_unique() != games.height:
        raise ValueError("normalized games contain duplicate game IDs")
    invalid_scores = games.filter((pl.col("home_score") < 0) | (pl.col("away_score") < 0)).height
    if invalid_scores:
        raise ValueError("normalized games contain negative scores")
    same_team = games.filter(pl.col("home_id") == pl.col("away_id")).height
    if same_team:
        raise ValueError("a team cannot be both home and away in one game")

    appearances = pl.concat(
        [
            games.select("game_id", "game_datetime", pl.col("home_id").alias("team_id")),
            games.select("game_id", "game_datetime", pl.col("away_id").alias("team_id")),
        ]
    )
    simultaneous = (
        appearances.group_by("team_id", "game_datetime")
        .agg(pl.len().alias("games"))
        .filter(pl.col("games") > 1)
        .height
    )
    if simultaneous:
        raise ValueError("a team has multiple games at the same source timestamp")


def load_games(path: Path | str = Path("data/games.parquet")) -> pl.DataFrame:
    """Load the checked-in normalized dataset and revalidate its invariants."""

    games = pl.read_parquet(path).sort(["game_datetime", "game_id"])
    validate_games(games)
    return games


def build_data_artifact(
    raw_dir: Path,
    output_path: Path,
    manifest_path: Path,
) -> DataBuildReport:
    """Build the normalized data artifact and its source manifest."""

    paths = [raw_dir / f"mbb_schedule_{season}.parquet" for season in SOURCE_SEASONS]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source files: {', '.join(missing)}")

    games, report = normalize_sources(paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    games.write_parquet(output_path, compression="zstd", statistics=True)

    sources = []
    for path in paths:
        season = int(path.stem.rsplit("_", maxsplit=1)[1])
        sources.append(
            {
                "asset": path.name,
                "bytes": path.stat().st_size,
                "season": season,
                "sha256": sha256_file(path),
                "url": SOURCE_URL_TEMPLATE.format(season=season),
            }
        )
    manifest = {
        "build_report": asdict(report),
        "dataset": {
            "asset": output_path.name,
            "bytes": output_path.stat().st_size,
            "columns": list(GAME_COLUMNS),
            "sha256": sha256_file(output_path),
        },
        "license": "CC-BY-4.0",
        "snapshot_date": SOURCE_SNAPSHOT_DATE,
        "source": "SportsDataverse/hoopR processed ESPN men's college basketball schedules",
        "source_release_tag": SOURCE_TAG,
        "sources": sources,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def fetch_sources(raw_dir: Path, manifest_path: Path | None = None) -> None:
    """Download the pinned release assets and optionally verify known hashes."""

    expected: dict[str, str] = {}
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {item["asset"]: item["sha256"] for item in manifest["sources"]}

    raw_dir.mkdir(parents=True, exist_ok=True)
    for season in SOURCE_SEASONS:
        name = f"mbb_schedule_{season}.parquet"
        destination = raw_dir / name
        if (
            destination.is_file()
            and name in expected
            and sha256_file(destination) == expected[name]
        ):
            continue
        request = Request(
            SOURCE_URL_TEMPLATE.format(season=season),
            headers={"User-Agent": "ncaab-model-validation/0.1"},
        )
        with urlopen(request, timeout=120) as response:
            payload = response.read()
        digest = sha256(payload).hexdigest()
        if name in expected and digest != expected[name]:
            raise ValueError(f"source hash mismatch for {name}")
        destination.write_bytes(payload)
