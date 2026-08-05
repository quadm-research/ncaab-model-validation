# Normalized data artifact

`games.parquet` is the checked-in real-data input for the public study. It contains 65,360 chronologically ordered final games from source seasons 2015 through 2026.

The artifact is intentionally narrow: source game ID, season, full UTC timestamp, source game date, team IDs and names, final scores, and neutral-site status. It excludes administrative forfeits and rows that fail the frozen eligibility rules.

`source-manifest.json` records the twelve upstream release assets, their URLs and SHA-256 hashes, all sequential filter counts, and the normalized artifact hash.

The data is a modified subset of SportsDataverse/hoopR processed schedule data and is distributed under CC BY 4.0. See the repository-level `DATA_LICENSE.md` for complete attribution and modification notices.
