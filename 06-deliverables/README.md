# 06 — Deliverables

Shareable outputs generated from the pipeline results.

## Contents

| File | Description |
|:-----|:------------|
| `deliverables.ipynb` | Notebook that builds the two Excel files from upstream TSV/TXT files. |
| `setup_run.py` | Prepares today's run folder: copies the upstream files the notebook reads into it when missing. |
| `20260602/` | Snapshot of the 2 June 2026 deliverables: `slides.md` (Marp slide deck, 8 slides), its rendered `slides.pdf`, and the Excel workbooks as of that date. |

### Outputs (in `assets/<date>/06/`)

| File | Description |
|:-----|:------------|
| `synbio_papers.xlsx` | Papers workbook with 3 sheets: **Cluster Summary**, **Paper Assignments**, **Literature Preceded**. |
| `igem_teams.xlsx` | Teams workbook with 3 sheets: **Cluster Summary**, **Team Assignments**, **iGEM Preceded**. |

### Workbook sheets

- **Cluster Summary** — `cluster_summary_papers.tsv` / `cluster_summary_igem.tsv`
  with the high / mid / low hierarchy IDs and names prepended and the topic
  description appended.
- **Paper / Team Assignments** — one row per non-outlier document with its
  hierarchy IDs and names, plus metadata (papers: `ID`, `doi`, `title`,
  `cited_by_count`, `publication_year`, `source_name`; teams: `UT`, `Title`,
  `Institutions`, `Year`).
- **Literature Preceded / iGEM Preceded** — `literature_preceded.tsv` /
  `igem_preceded.tsv` from `05-reporting/06-density_comparison.ipynb`. Topics
  are split by the mean-year gap (`delta_mean_years > 0` / `< 0`), while the
  precedence charts in `05-reporting` rank by the first quartile, so the two
  can differ for individual topics.

## How to use

1. Run the upstream steps (today or in earlier runs): `00`–`04`, and in
   `05-reporting` at least `01-cluster_summary_IGEM`,
   `02-cluster_summary_papers`, `03-compute_coords`, and
   `06-density_comparison`.
2. Run `deliverables.ipynb` from this folder. Its first cell copies any
   missing inputs into today's run folder (from the newest earlier run) and the
   workbooks are written to `assets/<date>/06/` (see *Where results live* in the root README).

## Slides

The Marp deck lives in `20260602/slides.md`; preview it with the
[Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)
extension. Its image links (`../assets/reports/*.png`) point to the old flat
`assets/reports/` folder, which no longer exists: figures now live in the run
folders (`assets/<date>/05/`). Three of the four images are not produced by the current
pipeline: `umap_heatmap.png` is not written by any committed code (the closest
current figure is `umap_density_ratio.png`), and `umap_papers.png` /
`umap_overlay.png` came from the removed `reporting.ipynb` (now
`umap_papers_{level}_{style}.png` and `umap_overlay_{level}.png`).
