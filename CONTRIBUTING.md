# Contributing

Small, reviewable changes are welcome when they strengthen point-in-time validity, source provenance, or statistical interpretation.

## Ground rules

- Do not add private datasets, bookmaker payloads, credentials, private paths, or proprietary research parameters.
- Keep synthetic data confined to tests. The main study must continue to use the pinned public real-data snapshot.
- Emit features before updating state with the current result.
- Treat a changed feature window, model hyperparameter, population rule, or test period as a new experiment rather than silently replacing the published specification.
- Preserve the simple-model references and paired uncertainty calculation.
- Add an adversarial test for every new temporal or data-quality failure mode.

## Local checks

```bash
uv lock --check
uv sync --all-groups --frozen
uv run ruff format --check .
uv run ruff check .
uv run mypy src/ncaab_model_validation
uv run pytest
uv run python -m compileall -q src tests
uv run ncaab-validate verify-data
uv run ncaab-validate analyze --output build/reproduction
```

Pull requests should state the research decision being changed, the evidence that motivates it, and which untouched data will evaluate it.
