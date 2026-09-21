# 00 — iGEM Teams Dataset

## Setup

The notebook reads two source files hosted on Google Drive (absolute paths
under `My Drive/SynBio/igem_teams/`):

- `team_project_descriptions_manual_entries_v2.tsv` — project titles and
  abstracts for each iGEM team (ISO-8859-1 encoded).
- `team_meta_full.tsv` — metadata (year, country, institution, section,
  track, status, etc.).

Make sure the Google-Drive file paths in the notebook point to valid
locations before running.

## Description

The notebook **`read_igem_data.ipynb`** prepares the iGEM competition
dataset for downstream analysis. All its outputs go to today's run folder,
`assets/<date>/00/` (see *Where results live* in the root README).

1. **Load raw data** — reads the two TSV files containing team metadata
   and project descriptions.
2. **Filter** — keeps only teams whose `Status` is *accepted* and whose
   `Section` is not *high-school*.
3. **Merge** — left-joins the project descriptions onto the metadata on
   `TeamID`, and prints how many teams have a non-empty abstract.
4. **Clean** — drops rows that have no project abstract.
5. **Rename columns** — maps the original column names to short codes
   used consistently across the rest of the pipeline: `UT` (team ID),
   `TI` (title), `AB` (abstract), `PY` (year), `AU` (team name), `WC`
   (topics), `DI` (wiki URL), `ID` (track), `Countries`, `Institutions`;
   empty `CR` and `C1` columns are added. The merge also keeps `Year_y`,
   which is the year column used by the downstream steps (`04`, `05`).
6. **Summary statistics** — prints counts such as total teams, year
   range, unique countries, institutions, tracks, and topics.
7. **Visualisations** — bar charts of teams per year, the top-20
   countries, institutions, tracks, and topics, and teams per section;
   line charts of the top-6 countries over time (raw counts and share of
   each year's teams, with Japan always included).
8. **Stats report** — writes `stats_report.md`, with its figures in
   `figures/` (`yearly_trend.png`, `country_top20.png`,
   `country_trend_raw.png`, `country_trend_norm.png`, `tracks_top20.png`).
9. **Export** — saves the cleaned dataset as `igem.txt` (TSV).

## Other files

- `team_awards.tsv` — iGEM award records per team (`Year`, `TeamID`, `Team`,
  `AwardTitle`, `AwardType`, `AwardSubType`, `AwardDecision`, `AwardGroup`,
  `AwardIcon`). It is **not** produced by the notebook; it is an external
  input read by `05-reporting/09-awards.ipynb` (`TeamID` matches `UT`).
