"""Write deterministic evaluation artifacts for independent review."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ncaab_model_validation.data import sha256_file
from ncaab_model_validation.evaluation import (
    DEFAULT_BOOTSTRAP_REPETITIONS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_TEST_SEASONS,
    EvaluationResult,
)
from ncaab_model_validation.models import MODEL_SPECIFICATIONS
from ncaab_model_validation.plotting import plot_fold_mae, plot_pooled_mae


def _records(frame: pl.DataFrame) -> list[dict[str, object]]:
    return [dict(row) for row in frame.iter_rows(named=True)]


def write_results(
    result: EvaluationResult,
    features: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Persist tables, audit rows, charts, and a machine-readable summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    result.fold_metrics.write_csv(output_dir / "fold-metrics.csv", float_precision=10)
    result.pooled_metrics.write_csv(output_dir / "pooled-metrics.csv", float_precision=10)
    result.comparisons.write_csv(output_dir / "mae-comparisons.csv", float_precision=10)
    result.predictions.write_parquet(
        output_dir / "walk-forward-predictions.parquet",
        compression="zstd",
        statistics=True,
    )
    features.write_parquet(
        output_dir / "point-in-time-features.parquet",
        compression="zstd",
        statistics=True,
    )
    plot_fold_mae(result.fold_metrics, output_dir / "mae-by-season.svg")
    plot_pooled_mae(result.pooled_metrics, output_dir / "pooled-mae.svg")

    summary = {
        "bootstrap": {
            "cluster": "source game_date within test season",
            "confidence_interval": 0.95,
            "repetitions": DEFAULT_BOOTSTRAP_REPETITIONS,
            "seed": DEFAULT_BOOTSTRAP_SEED,
        },
        "comparisons": _records(result.comparisons),
        "model_specifications": MODEL_SPECIFICATIONS,
        "pooled_metrics": _records(result.pooled_metrics),
        "test_seasons": list(DEFAULT_TEST_SEASONS),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_names = (
        "fold-metrics.csv",
        "mae-by-season.svg",
        "mae-comparisons.csv",
        "point-in-time-features.parquet",
        "pooled-mae.svg",
        "pooled-metrics.csv",
        "summary.json",
        "walk-forward-predictions.parquet",
    )
    manifest = {
        "artifacts": [
            {
                "asset": name,
                "bytes": (output_dir / name).stat().st_size,
                "sha256": sha256_file(output_dir / name),
            }
            for name in artifact_names
        ],
        "generator": "ncaab-model-validation 0.1.0",
    }
    (output_dir / "reproduction-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def format_console_summary(result: EvaluationResult) -> str:
    """Return a compact terminal summary for the reproducibility command."""

    lines = ["Pooled walk-forward metrics (2019-2026)", "model       MAE       RMSE      bias"]
    for row in result.pooled_metrics.sort("mae").iter_rows(named=True):
        lines.append(
            f"{row['model']!s:<9} {float(row['mae']):>8.3f} "
            f"{float(row['rmse']):>10.3f} {float(row['bias']):>9.3f}"
        )
    boost_ridge = result.comparisons.filter(
        (pl.col("candidate") == "boost") & (pl.col("reference") == "ridge")
    ).row(0, named=True)
    lines.extend(
        [
            "",
            "Primary paired comparison (positive favors candidate)",
            (
                "boost vs ridge MAE improvement: "
                f"{float(boost_ridge['mae_improvement']):.3f} "
                f"[95% date-clustered CI {float(boost_ridge['ci_95_low']):.3f}, "
                f"{float(boost_ridge['ci_95_high']):.3f}]"
            ),
        ]
    )
    return "\n".join(lines)
