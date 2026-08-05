# Reviewer guide

A focused review can be completed in this order:

1. Read `docs/research-protocol.md` for the frozen estimand and comparisons.
2. Inspect `data/source-manifest.json` and `docs/data-provenance.md` for row accounting and hashes.
3. Read `build_features` in `src/ncaab_model_validation/features.py`. Confirm that feature emission precedes state updates.
4. Read `run_evaluation` and `clustered_mae_comparison` in `src/ncaab_model_validation/evaluation.py`. Confirm that training seasons are strictly earlier and resampling is paired by date within season.
5. Compare `results/fold-metrics.csv` with `results/mae-comparisons.csv`. Check whether pooled conclusions hide a weak season.
6. Run the local gates and reproduction commands from the README.

Useful challenges include changing a current game's score and confirming its pregame features do not move, injecting a same-team/same-timestamp conflict, replacing a final with a forfeit, and rerunning the clustered comparison with the fixed seed.

The strongest limitation is intentional: this is historical final-score modeling, not a prospective forecast or market-value study. Review comments should distinguish implementation validity, statistical interpretation, and external usefulness.
