# 00 — iGEM Teams Dataset

## Setup

The notebook reads two source files hosted on Google Drive:

- `team_project_descriptions.tsv` — project abstracts for each iGEM team.
- `team_meta_full.tsv` — metadata (year, country, institution, track, etc.).

Make sure the Google-Drive file paths in the notebook point to valid
locations before running.

## Description

The notebook **`read_igem_data.ipynb`** prepares the iGEM competition
dataset for downstream analysis.

1. **Load raw data** — reads the two TSV files containing team metadata
   and project descriptions.
2. **Filter** — keeps only teams whose participation status is
   *accepted*.
3. **Merge** — joins metadata and project descriptions on the team
   identifier.
4. **Clean** — drops rows that have no project abstract.
5. **Rename columns** — maps the original column names to short codes
   used consistently across the rest of the pipeline (e.g. `TI` for
   title, `AB` for abstract, `PY` for publication year).
6. **Summary statistics** — prints counts such as total teams, year
   range, unique countries, institutions, and tracks.
7. **Visualisations** — produces bar charts showing the number of teams
   per year, the top-20 countries, institutions, tracks, and topics.
8. **Export** — saves the cleaned dataset to `assets/igem.txt` (TSV).
