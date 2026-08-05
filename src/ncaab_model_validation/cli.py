"""Command-line entry points for data provenance and model validation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ncaab_model_validation.data import (
    build_data_artifact,
    fetch_sources,
    load_games,
    sha256_file,
)
from ncaab_model_validation.evaluation import run_evaluation
from ncaab_model_validation.features import build_features


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ncaab-validate",
        description="Reproduce the QuadM Research NCAA basketball model-validation case study.",
    )
    commands = parser.add_subparsers(dest="command")

    analyze = commands.add_parser("analyze", help="rerun the frozen walk-forward analysis")
    analyze.add_argument("--data", type=Path, default=Path("data/games.parquet"))
    analyze.add_argument("--output", type=Path, default=Path("results"))

    build = commands.add_parser("build-data", help="normalize already-downloaded source assets")
    build.add_argument("--raw-dir", type=Path, required=True)
    build.add_argument("--output", type=Path, default=Path("data/games.parquet"))
    build.add_argument("--manifest", type=Path, default=Path("data/source-manifest.json"))

    fetch = commands.add_parser("fetch-data", help="download the pinned public source assets")
    fetch.add_argument("--raw-dir", type=Path, required=True)
    fetch.add_argument("--manifest", type=Path, default=Path("data/source-manifest.json"))

    reproduce = commands.add_parser(
        "reproduce", help="rebuild normalized data from local source assets and analyze it"
    )
    reproduce.add_argument("--raw-dir", type=Path, required=True)
    reproduce.add_argument("--data", type=Path, default=Path("data/games.parquet"))
    reproduce.add_argument("--manifest", type=Path, default=Path("data/source-manifest.json"))
    reproduce.add_argument("--output", type=Path, default=Path("results"))

    verify = commands.add_parser("verify-data", help="validate the data and its published hash")
    verify.add_argument("--data", type=Path, default=Path("data/games.parquet"))
    verify.add_argument("--manifest", type=Path, default=Path("data/source-manifest.json"))
    return parser


def _analyze(data_path: Path, output_dir: Path) -> None:
    from ncaab_model_validation.reporting import format_console_summary, write_results

    games = load_games(data_path)
    features = build_features(games)
    result = run_evaluation(features)
    write_results(result, features, output_dir)
    print(format_console_summary(result))


def _verify_data(data_path: Path, manifest_path: Path) -> None:
    games = load_games(data_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = str(manifest["dataset"]["sha256"])
    actual = sha256_file(data_path)
    if actual != expected:
        raise ValueError("normalized dataset hash does not match the source manifest")
    expected_rows = int(manifest["build_report"]["eligible_rows"])
    if games.height != expected_rows:
        raise ValueError("normalized dataset row count does not match the source manifest")
    print(f"verified {games.height:,} games; sha256={actual}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested reproducibility action."""

    args = _parser().parse_args(argv)
    command = args.command
    if command is None:
        _analyze(Path("data/games.parquet"), Path("results"))
        return 0
    if command == "analyze":
        _analyze(args.data, args.output)
    elif command == "build-data":
        report = build_data_artifact(args.raw_dir, args.output, args.manifest)
        print(
            f"wrote {report.eligible_rows:,} eligible games "
            f"for seasons {report.first_season}-{report.last_season}"
        )
    elif command == "fetch-data":
        fetch_sources(args.raw_dir, args.manifest)
        print(f"downloaded source assets to {args.raw_dir}")
    elif command == "reproduce":
        report = build_data_artifact(args.raw_dir, args.data, args.manifest)
        print(f"rebuilt {report.eligible_rows:,} eligible games")
        _analyze(args.data, args.output)
    elif command == "verify-data":
        _verify_data(args.data, args.manifest)
    else:
        raise AssertionError(f"unhandled command: {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
