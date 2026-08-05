# Disclosure boundary

## Included

- Public SportsDataverse/hoopR schedule and final-score fields needed for the study.
- Exact source asset hashes and transformation counts.
- Generic pregame team-state features and frozen model specifications.
- Game-level point-in-time features and held-out predictions.
- Fold metrics, paired clustered uncertainty, and deterministic figures.
- Synthetic unit-test fixtures used only to challenge timing and inference behavior.

## Intentionally excluded

- Bookmaker quotes, exchange messages, order books, or raw vendor payloads.
- Credentials, private paths, infrastructure identifiers, and operational configuration.
- Proprietary market selection, pricing, sizing, execution, or risk parameters.
- Private research datasets, private results, and private candidate taxonomies.
- Player-level availability or minutes features whose eligible universe is not known before tipoff.
- Claims about profitability, tradable edge, capacity, or live performance.

## Why this boundary matters

The repository is meant to let a reviewer inspect statistical decisions without revealing private research assets or implying evidence the study does not contain. Final-score forecast accuracy is not the same estimand as market value. A model can reduce margin error and still have no economic use after price, timing, limits, and costs.

The public result therefore stops at model validation. Any downstream market research remains a separate, private, point-in-time study.
