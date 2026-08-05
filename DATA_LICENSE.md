# Data license and attribution

The file `data/games.parquet` is a transformed subset of the SportsDataverse/hoopR processed ESPN men's college basketball schedule datasets.

- Creator and project: SportsDataverse contributors, hoopR
- Source: <https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_mens_college_basketball_schedules>
- Upstream license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Snapshot date: 2026-08-05
- Changes made: selected seasons 2015–2026; retained final games with valid scores and active teams; removed known non-Division-I opponents, administrative forfeits, unused source fields, and duplicates; renamed fields; converted game timestamps to UTC; sorted records chronologically; encoded the result as compressed Parquet.

The derived data file is distributed under CC BY 4.0. Attribution does not imply endorsement by SportsDataverse, hoopR, ESPN, or any upstream contributor.

The MIT license in `LICENSE` applies to the repository's original code and documentation. It does not replace the data license or grant rights in third-party marks.
