# 05 — Reporting

## Setup

All input files are outputs of the previous pipeline steps (datasets,
embeddings, topic assignments, topic names, and the hierarchy tables from
`04-topic_hierarchy`), stored in the run folders under `assets/`.
`09-awards.ipynb` additionally reads `00-IGEM_teams_dataset/team_awards.tsv`. No
external API access is needed.

Every notebook and script starts with `RUN = setup()` from this folder's
`setup_run.py`. It copies the upstream files it needs into today's run folder when
they are missing there (from the newest earlier run), and all outputs are written
to `assets/<date>/05/`. Files one 05 notebook reads from another (coordinates,
precedence table, LQ table) are copied the same way, so any single notebook can be
re-run on its own (see *Where results live* in the root README).

## Notebooks and run order

Notebooks are numbered in run order (there is no `05-`). The two cluster
summaries are independent of the joint-UMAP chain.

| Notebook | Needs | Main outputs |
|---|---|---|
| `01-cluster_summary_IGEM.ipynb` | `igem.txt`, teams topics + names | `cluster_summary_igem.tsv` |
| `02-cluster_summary_papers.ipynb` | `synbio_openalex.txt`, papers topics + names | `cluster_summary_papers.tsv`, `impact_vs_price_index.png` |
| `03-compute_coords.ipynb` | embeddings, doc topics | `joint_umap_{papers,teams}_xy.tsv` |
| `04-joint_umap.ipynb` | `03`, hierarchy tables | `umap_*` topic maps |
| `06-density_comparison.ipynb` | `03`, hierarchy tables | density maps, `overlap_precedence_full.tsv`, `igem_preceded.tsv`, `literature_preceded.tsv` |
| `07-precedence_charts.ipynb` | `06` | `overlap_precedence_*.png` |
| `08-appendix.ipynb` | `03`, `06` | supplementary / robustness tables and figures |
| `09-awards.ipynb` | `03`, `06`, `08`, `team_awards.tsv` | `awards_*` tables and figure |

`06-deliverables` needs `01`, `02`, and `06`.

## Cluster summaries

### `01-cluster_summary_IGEM.ipynb`

Cluster-level summaries for the **iGEM teams** topic model (year-based, no
citation metrics), excluding outliers (topic −1). Per topic: team and record
counts, average / median / std / min / max year, Price Index (share of teams in
the last five years), average year rank (mean percentile rank of the team year
across all teams), and the number of distinct countries (`n_countries`). Also
shows a recency-vs-year-rank scatter (not saved). **Output:**
`assets/<date>/05/cluster_summary_igem.tsv`.

### `02-cluster_summary_papers.ipynb`

Cluster-level summaries for the **papers** topic model, excluding outliers.
Citations are normalised within each publication year (z-score, z-score of
log(1 + citations), and percentile rank); per topic the notebook reports paper
count, average citations, the three normalised impact averages, average year,
Price Index, and average year rank. It also draws impact-vs-recency and
impact-vs-Price-Index scatters (only the second is saved). **Outputs:**
`assets/<date>/05/cluster_summary_papers.tsv`,
`assets/<date>/05/impact_vs_price_index.png`.

## Joint-UMAP analysis

Notebooks `03`–`07` and `09` share **one** UMAP projection and are backed by
the `aux/` package.

**Unified coordinates.** The projection stacks both corpora and is fit **once**
by `03-compute_coords.ipynb`, which persists the 2D coordinates. Every other
notebook loads those coordinates and draws on the **same** axis limits with an
equal aspect ratio — so a teams-only map shows teams in their real sub-region of
the shared space instead of being stretched to fill the frame, and the papers,
teams, overlay, and density maps are all geometrically comparable.

**Coloring × labeling.** Topic maps can be coloured by any hierarchy level —
`micro` (low-level clusters), `meso` (mid groups), `macro` (high groups) — and
labelled in two styles: `side` (gutter labels with elbow connectors) or
`overlay` (labels placed on top of the clusters). `04-joint_umap.ipynb` emits
every combination.

### `03-compute_coords.ipynb`

Projects papers + teams together with a single 2D UMAP fit (cosine metric,
`n_neighbors=15`, `min_dist=0.1`, seed 42). The long radial tail of outlying
points is then pulled back towards the edge of the cloud
(`squash_radial_outliers`) so the canvas is used efficiently. Run first; re-run
only when embeddings or topic models change. **Outputs:**
`assets/<date>/05/joint_umap_papers_xy.tsv`,
`assets/<date>/05/joint_umap_teams_xy.tsv`.

### `04-joint_umap.ipynb`

Topic scatter maps on the shared canvas: papers (point size by citations) and
teams, each at micro / meso / macro coloring × side / overlay labels, plus the
papers-muted + teams-coloured overlay per level. **Outputs:**
`assets/<date>/05/umap_{papers,teams}_{level}_{style}.png`,
`assets/<date>/05/umap_overlay_{level}.png`.

### `06-density_comparison.ipynb`

Kernel-density estimate of each corpus over the shared space (bandwidth 0.15 on
a 300 × 300 grid, each density normalised to sum to 1), rendered as
`log2(teams_density / papers_density)` (red = teams-dense, blue = papers-dense).
Classifies topics into coverage zones and computes the **temporal precedence** of
overlap-zone topics (see below). The first map labels the most teams-dense and
papers-dense topics; the second labels the overlap topics with the most extreme
`delta_q1_years`. **Outputs:**
`assets/<date>/05/umap_density_ratio.png`, `umap_density_ratio_extremes.png`,
`overlap_precedence_full.tsv`, `igem_preceded.tsv`, `literature_preceded.tsv`
(the last two are consumed by `06-deliverables`).

### `07-precedence_charts.ipynb`

Four views of the overlap-topic precedence (loaded from
`overlap_precedence_full.tsv`): split violins of year distributions (A), a
median-year dumbbell with IQR bands (B), a diverging precedence-gap plot centred
at zero (C), and a pixel/grid heatmap that "pixelates" the split violins — two
year-cell strips per topic (Papers blue, Teams orange), each coloured by its
share of the topic's peak year, with a per-row totals panel (D). Each view is
produced for **two precedence criteria** — the first quartile (`delta_q1_years`)
and the stricter earliest 5% (`delta_p5_years`, the leading edge of each corpus).
**Outputs:**
`assets/<date>/05/overlap_precedence_{violin_horizontal,dumbbell_horizontal,diverging_gap,pixel_grid}.png`
(first quartile) and the matching `…_p5.png` files (earliest 5%).

## Supplementary analyses

### `08-appendix.ipynb`

Runs five standalone scripts (in this folder) that back the supplementary
figures and robustness checks. They re-use the persisted joint coordinates, so
no UMAP / BERTopic re-run is needed, but `03-compute_coords` and
`06-density_comparison` must have run first (today or in an earlier run). The
scripts do not import `aux/`; they re-implement the logic they need. Run the
cells top to bottom: §3 and §4 read the table written by §2.

| § | Script | What it answers | Outputs (`assets/<date>/05/`) |
|---|---|---|---|
| 1 | `density_eras.py` | Is the teams-vs-papers density split stable across eras (2004–2014 vs 2015–2025)? | `density_ratio_eras.png` |
| 2 | `geography_lq.py` | Which countries are over-represented in iGEM relative to the literature (Location Quotient)? | `geography_location_quotient.tsv`, `geography_lq.png` |
| 3 | `geography_topic_matrix.py` | Do outsider-tilted countries concentrate in the iGEM-led overlap topics? | `geography_topic_matrix.tsv`, `geography_topic_matrix.png`, `geography_lead_orientation.png` |
| 4 | `geography_outsiderness.py` | Do countries stop being outsiders over time (literature-density "outsiderness" score per team)? | `geography_outsiderness_teams.tsv`, `geography_outsiderness_trajectory.tsv`, `geography_outsiderness_trajectory.png`, `geography_outsiderness_field_check.png` |
| 5 | `precedence_robustness.py` | Does the precedence result survive other year statistics and neighbourhood parameters? | `precedence_robustness.tsv`, `precedence_robustness_sensitivity.tsv`, `precedence_robustness.png` |

§3 and §5 also read `overlap_precedence_full.tsv` from `06-density_comparison`.

### `09-awards.ipynb`

Tests whether iGEM **village awards** (the per-track "Best … Project" prizes, e.g.
Best Therapeutics Project) line up
with the precedence clusters and with team outsiderness. Keeps
`AwardDecision = winner` × `AwardType = village` rows from `team_awards.tsv`,
restricted to the study population (so high-school winners are dropped), and
joins each winner to its map position, home cluster, and contemporaneous
outsiderness (`geography_outsiderness_teams.tsv` from `08-appendix` §4). It then:

- counts winners around each overlap-precedence cluster using the same
  neighbourhood rule as the precedence table (Table A);
- compares each home macro cluster's share of winners with its share of teams
  (Table B; meso clusters are shown but not saved);
- compares winners' outsiderness with all teams and nominees, and across award
  tracks grouped as clinical / engineering / environmental·applied
  (Mann–Whitney, Kruskal–Wallis).

**Outputs:** `awards_village_winners.tsv`, `awards_by_overlap_cluster.tsv`,
`awards_by_home_cluster.tsv`, `awards_outsiderness_by_track.tsv`,
`awards_outsiderness.png`.

## Shared module — `aux/`

Imported via `sys.path.insert(0, str(Path.cwd()))` then `from aux.<module> import …`.
Used by notebooks `03`, `04`, `06`, `07`, and `09`; `01`, `02`, and the `08`
scripts don't use it. Its I/O helpers (`load_plot_data`, `save_coords`,
`save_precedence`, …) take the `RUN` handle as their first argument.

| Module | Responsibility |
|---|---|
| `paths.py` | Palette, seed, persisted-coordinate filenames |
| `coords.py` | Compute / persist / load the joint projection; build hierarchy-aware plot frames; shared limits; per-level colours and label anchors |
| `labels.py` | Label placement: `add_side_labels`, `add_overlay_labels`, `add_density_labels` |
| `scatter.py` | `plot_topic_map` (level × label style) and `plot_overlay` |
| `density.py` | KDE density-ratio grid, heatmap rendering, zone classification, label selection |
| `precedence.py` | `compute_precedence` / `save_precedence`, `gather_year_pools`, `gather_year_grid`, and the violin / dumbbell / diverging / pixel-grid charts |

File locations come from the `RUN` handle, not from `paths.py`.

## Temporal precedence explanation

**What is compared to what**

INPUT LAYERS
- Papers (individual rows): paper id, publication_year, x, y in joint UMAP
- Teams (individual rows): team id (`UT`), Year_y, x, y in joint UMAP
- Topic labels: each paper belongs to one low-level paper topic; each team to one team topic

TOPIC SUMMARY LAYER
- Build paper-topic and team-topic centroids (median x, y of their points; outliers excluded)
- Sample the density-ratio map at each centroid
- Classify each topic centroid as teams-dominant (log2 ratio > 1), papers-only
  (< −1), overlap (in between), or unmapped (outside the grid)

PRECEDENCE LAYER
For each overlap paper topic:
1. Compute the centroid of that paper topic
2. Set the radius to 1.5 × the median distance of the topic's papers from the centroid
3. Draw a circle around that centroid
4. Query a KD-tree built from all team points
5. Collect all team points inside the circle; skip the topic if fewer than 3 have a year
6. Pull their team years (Year_y) and the paper topic's publication years
7. Summarise each year distribution with seven statistics — min, p5, q1, mean,
   median, q3, max (`papers_<stat>_year`, `teams_<stat>_year`)
8. Compute `delta_<stat>_years = teams_<stat>_year − papers_<stat>_year` for each statistic

WHICH STATISTIC IS USED WHERE
- `overlap_precedence_full.tsv` keeps all seven deltas, sorted by `delta_mean_years`.
- `igem_preceded.tsv` / `literature_preceded.tsv` split topics by the **mean**:
  `delta_mean_years < 0` vs `> 0`.
- The precedence charts and the extremes density map use the **first quartile**
  (`delta_q1_years`); the charts also have an earliest-5% version (`delta_p5_years`).

OUTPUT INTERPRETATION
- delta < 0: nearby teams are earlier → teams preceded the literature
- delta > 0: papers are earlier → the literature preceded teams

The neighbourhood query captures **individual** team points and their years (not
team-topic centroids), so the anchor is one paper-topic centroid while the
evidence is many nearby teams, possibly spanning several team topics. This
measures local practice-vs-literature timing in semantic space without forcing a
one-to-one paper-topic ↔ team-topic pairing.
