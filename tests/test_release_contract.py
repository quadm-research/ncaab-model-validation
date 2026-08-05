from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ncaab_model_validation.data import load_games, sha256_file
from ncaab_model_validation.reporting import _round_float_columns

ROOT = Path(__file__).resolve().parents[1]


def test_real_data_artifact_matches_manifest() -> None:
    data_path = ROOT / "data" / "games.parquet"
    manifest = json.loads((ROOT / "data" / "source-manifest.json").read_text(encoding="utf-8"))
    games = load_games(data_path)

    assert games.height == 65_360
    assert games["season"].min() == 2015
    assert games["season"].max() == 2026
    assert manifest["build_report"]["forfeit_rows"] == 16
    assert sha256_file(data_path) == manifest["dataset"]["sha256"]


def test_checked_results_preserve_the_declared_complexity_comparison() -> None:
    pooled = pl.read_csv(ROOT / "results" / "pooled-metrics.csv")
    comparisons = pl.read_csv(ROOT / "results" / "mae-comparisons.csv")
    mae = {row["model"]: row["mae"] for row in pooled.iter_rows(named=True)}

    assert pooled["games"].unique().to_list() == [43_326]
    assert mae["boost"] < mae["ridge"] < mae["elo"] < mae["baseline"]
    boost_ridge = comparisons.filter(
        (pl.col("candidate") == "boost") & (pl.col("reference") == "ridge")
    ).row(0, named=True)
    assert boost_ridge["ci_95_low"] > 0.0
    assert boost_ridge["seasons_better"] == 8
    assert boost_ridge["seasons_total"] == 8


def test_generated_result_hashes_match_reproduction_manifest() -> None:
    manifest = json.loads(
        (ROOT / "results" / "reproduction-manifest.json").read_text(encoding="utf-8")
    )

    for artifact in manifest["artifacts"]:
        path = ROOT / "results" / artifact["asset"]
        assert path.stat().st_size == artifact["bytes"]
        assert sha256_file(path) == artifact["sha256"]


def test_persisted_float_columns_have_a_platform_stable_precision() -> None:
    values = [1.12345678904, -2.98765432106]
    frame = pl.DataFrame({"prediction": values, "game_id": ["a", "b"]})

    rounded = _round_float_columns(frame)

    assert rounded["prediction"].to_list() == [round(value, 10) for value in values]
    assert rounded["game_id"].to_list() == ["a", "b"]
