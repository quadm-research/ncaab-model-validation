# Research protocol

## Question

Can a small set of pregame team-state features improve held-out NCAA men's basketball margin forecasts, and does a frozen nonlinear model add practically meaningful accuracy beyond a regularized linear model?

This protocol is designed to expose research judgment rather than maximize a leaderboard score. It privileges temporal validity, explicit comparisons, and reviewable failure conditions.

## Population and outcome

The source snapshot contains SportsDataverse/hoopR schedule assets for seasons 2015 through 2026. Eligibility is applied in this order:

1. retain `STATUS_FINAL` games, excluding scheduled, canceled, and administrative-forfeit records;
2. require a source timestamp, source game date, team IDs, and both scores;
3. require both teams to be marked active by the source;
4. remove opponents explicitly marked non-Division-I where that field is available;
5. deduplicate by source game ID and fail if a team has two eligible games at the same timestamp.

The outcome is final home score minus final away score. The normalized artifact contains 65,360 games. The filtering audit is pinned in `data/source-manifest.json`.

## Point-in-time feature clock

Games are sorted by full source timestamp in UTC, with game ID as a deterministic tie-breaker for unrelated simultaneous games. Calendar date alone is not used for ordering because late tournament games can share an Eastern date while occurring many hours apart.

For each game, the code:

1. advances each team to the current season and applies fixed offseason shrinkage;
2. reads both teams' state and emits the feature row;
3. reads the current final score only as the target;
4. updates both teams after the row has been emitted.

The frozen features are:

- pregame Elo difference;
- exponentially weighted margin difference;
- trailing-five margin difference;
- home and away exponentially weighted points for and against;
- rest difference from full timestamps, capped at 14 days;
- games played in the current season for each team;
- neutral-site indicator.

Constants are fixed at Elo K = 24, non-neutral home advantage = 55 rating points, exponential weight = 0.18, and offseason retention = 0.60. Default state is a 1500 Elo rating and 70 points for/against.

## Models

Four specifications are fit without within-study tuning:

| Name | Specification |
| --- | --- |
| Venue mean | Training-only mean margin, split by neutral-site status |
| Elo-only ridge | Standardized Elo difference and neutral flag, Ridge alpha = 10 |
| Full ridge | Standardized frozen feature set, Ridge alpha = 10 |
| Gradient boost | Histogram gradient boosting with learning rate 0.05, 200 iterations, 15 leaves, minimum leaf size 100, L2 = 10, seed 42 |

The venue mean tests whether any team information helps. Elo-only ridge tests a compact strength summary. Full ridge tests incremental value from interpretable state. Gradient boosting is the complexity check against full ridge.

## Walk-forward design

Each season from 2019 through 2026 is tested once. A fold trains only on seasons strictly earlier than the held-out season. Models are not refit during the held-out season, although features legitimately incorporate results observed earlier in that season.

The primary loss is mean absolute error in points. Root mean squared error is secondary, and signed bias is reported to reveal directional drift. Pooled metrics weight games equally; per-season tables make stability visible.

## Paired uncertainty

Model comparisons use the game-level difference in absolute error. A positive value means the candidate model improved on its reference. The bootstrap resamples entire source game-date clusters within each test season, preserving the daily dependence shared by games and the original season mix. It uses 5,000 draws and seed 20260805.

The intervals are descriptive uncertainty estimates for this historical sequence. They are not a guarantee of future error and are not converted into trading or economic claims.

## Decision discipline

The feature windows, shrinkage, model hyperparameters, test seasons, metric, and bootstrap design were frozen before the corrected final run. The same specifications are retained even if one season is unfavorable. Any future tuning should begin a separately labeled experiment with a new untouched evaluation period.

The nonlinear model earns inclusion only by beating full ridge consistently enough to justify its reduced interpretability. In this snapshot it improves pooled MAE by about 0.101 points and is better in all eight folds. That is statistically stable under the declared resampling scheme but practically modest, so the repository does not frame it as a breakthrough.
