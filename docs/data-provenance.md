# Data provenance

## Upstream source

The study uses processed ESPN men's college basketball schedule datasets published by SportsDataverse for hoopR:

- data repository: <https://github.com/sportsdataverse/sportsdataverse-data>
- release tag: <https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_mens_college_basketball_schedules>
- hoopR loader documentation: <https://hoopr.sportsdataverse.org/reference/load_mbb_schedule.html>

The twelve source assets cover seasons 2015–2026. `data/source-manifest.json` pins every direct asset URL, byte count, and SHA-256 digest as observed on 2026-08-05.

## Transformation audit

| Stage | Rows |
| --- | ---: |
| Raw source rows | 70,992 |
| Source-marked completed rows | 70,928 |
| Administrative forfeits among completed rows | 16 |
| Final game rows | 70,912 |
| Missing required fields among finals | 0 |
| Inactive-team rows removed | 5,552 |
| Explicit non-Division-I rows removed after active-team filter | 0 |
| Duplicate game rows removed | 0 |
| Eligible normalized games | 65,360 |

The non-Division-I field is not present in every historical asset. The build treats it as false when absent and separately requires both teams to be active. The population should therefore be described exactly as active-team final games with known non-Division-I opponents excluded, not as a formally complete census of Division-I competition.

## Time handling and forfeits

The source's full timezone-aware timestamp is converted to UTC for state ordering. The source game date is retained separately and used as the dependence cluster in inference.

Several holiday-tournament teams appear twice on one Eastern calendar date because one game tipped shortly after midnight and another much later that day. Those are real sequential games and are ordered by timestamp. Administrative 2–0 forfeits are labeled `STATUS_FORFEIT` upstream and are excluded from both targets and state updates. After filtering, no team has two eligible games at the same source timestamp.

## Artifact and licensing

`data/games.parquet` contains only the fields needed to reproduce the study. It is 793,299 bytes with SHA-256 `5cd668615548218e16427d344d61d8d69ae601fec68414003c69f1b423411a69`.

The derived data is distributed under CC BY 4.0 with attribution and modification notices in `DATA_LICENSE.md`. Repository code remains under MIT.

This project is not affiliated with or endorsed by SportsDataverse, hoopR, ESPN, the NCAA, or any team.
