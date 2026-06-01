# 05 — Reporting

## Setup

All input files are the outputs of the previous pipeline steps (datasets,
embeddings, topic assignments, topic names, and — for the joint-UMAP notebooks —
the hierarchy tables from `04-topic_hierarchy`), all stored under `assets/`. No
external API access is needed.

## Cluster summaries

### `cluster_summary_papers.ipynb`

Cluster-level summaries and figures for the **papers** topic model: per-topic
citation-normalised impact (z-score, log z-score, percentile rank), average
year, Price Index, and recency, plus impact-vs-recency / impact-vs-Price-Index
scatters. **Output:** `assets/reports/cluster_summary_papers.tsv`.

### `cluster_summary_IGEM.ipynb`

Cluster-level summaries for the **iGEM teams** topic model (year-based, no
citation metrics): team counts, average year, Price Index, recency, and
top-country distribution per topic. **Output:**
`assets/reports/cluster_summary_igem.tsv`.

## Joint-UMAP analysis (modular)

These four notebooks replace the former monolithic `reporting_enhanced.ipynb`.
They split the joint-embedding analysis into focused units that all share **one**
UMAP projection, and they are backed by the `aux/` package.

**Unified coordinates.** The projection stacks both corpora and is fit **once**
by `compute_coords.ipynb`, which persists the 2D coordinates. Every other
notebook loads those coordinates and draws on the **same** axis limits with an
equal aspect ratio — so a teams-only map shows teams in their real sub-region of
the shared space instead of being stretched to fill the frame, and the papers,
teams, overlay, and density maps are all geometrically comparable.

**Coloring × labeling.** Topic maps can be coloured by any hierarchy level —
`micro` (low-level clusters), `meso` (mid groups), `macro` (high groups) — and
labelled in two styles: `side` (gutter labels with elbow connectors) or
`overlay` (labels placed on top of the clusters). `joint_umap.ipynb` emits every
combination.

Run order: **`compute_coords` → `joint_umap` / `density_comparison` →
`precedence_charts`** (the last needs the precedence table written by
`density_comparison`).

### `compute_coords.ipynb`

Projects papers + teams together with a single UMAP fit and persists the
coordinates. Run first; re-run only when embeddings or topic models change.
**Outputs:** `assets/reports/joint_umap_papers_xy.tsv`,
`assets/reports/joint_umap_teams_xy.tsv`.

### `joint_umap.ipynb`

Topic scatter maps on the shared canvas: papers and teams, each at micro / meso /
macro coloring × side / overlay labels, plus the papers-muted + teams-coloured
overlay per level. **Outputs:** `assets/reports/umap_{papers,teams}_{level}_{style}.png`,
`assets/reports/umap_overlay_{level}.png`.

### `density_comparison.ipynb`

Kernel-density estimate of each corpus over the shared space, rendered as
`log2(teams_density / papers_density)` (red = teams-dense, blue = papers-dense).
Classifies topics into coverage zones and computes the **temporal precedence** of
overlap-zone topics. **Outputs:**
`assets/reports/umap_density_ratio.png`, `umap_density_ratio_extremes.png`,
`overlap_precedence_full.tsv`, `igem_preceded.tsv`, `literature_preceded.tsv`
(the last two are consumed by `06-deliverables`).

### `precedence_charts.ipynb`

Three views of the overlap-topic precedence (loaded from
`overlap_precedence_full.tsv`), sorted by `delta_q1_years`: split violins of year
distributions (A), a median-year dumbbell with IQR bands (B), and a diverging
precedence-gap plot centred at zero (C). **Outputs:**
`assets/reports/overlap_precedence_{violin_horizontal,dumbbell_horizontal,diverging_gap}.png`.

### Shared module — `aux/`

Imported via `sys.path.insert(0, str(Path.cwd()))` then `from aux.<module> import …`.

| Module | Responsibility |
|---|---|
| `paths.py` | Paths, palette, seed, persisted-coordinate filenames |
| `coords.py` | Compute / persist / load the joint projection; build hierarchy-aware plot frames; shared limits; per-level colours and label anchors |
| `labels.py` | Label placement: `add_side_labels`, `add_overlay_labels`, `add_density_labels` |
| `scatter.py` | `plot_topic_map` (level × label style) and `plot_overlay` |
| `density.py` | KDE density-ratio grid, heatmap rendering, zone classification, label selection |
| `precedence.py` | `compute_precedence` / `save_precedence`, `gather_year_pools`, and the violin / dumbbell / diverging charts |

Paths in `paths.py` resolve from the module's own location, so the notebooks
work regardless of the kernel's working directory.

## Temporal precedence explanation

**What is compared to what**

INPUT LAYERS
- Papers (individual rows): paper_id, publication_year, x, y in joint UMAP
- Teams (individual rows): team_id, Year_y, x, y in joint UMAP
- Topic labels: each paper belongs to one paper topic; each team to one team topic

TOPIC SUMMARY LAYER
- Build paper-topic and team-topic centroids from their points
- Sample the density-ratio map at each centroid
- Classify each topic centroid as teams-dominant, papers-only, or overlap

PRECEDENCE LAYER
For each overlap paper topic:
1. Compute the centroid of that paper topic
2. Compute a radius from the paper-topic spread
3. Draw a circle around that centroid
4. Query a KD-tree built from all team points
5. Collect all team points inside the circle
6. Pull their team years (Year_y)
7. Compare avg_year_papers (of the paper topic) with avg_year_teams (nearby teams)
8. Compute delta_years = avg_year_teams − avg_year_papers

OUTPUT INTERPRETATION
- delta_years < 0: nearby teams are earlier → teams preceded the literature
- delta_years > 0: papers are earlier → the literature preceded teams

The neighbourhood query captures **individual** team points and their years (not
team-topic centroids), so the anchor is one paper-topic centroid while the
evidence is many nearby teams, possibly spanning several team topics. This
measures local practice-vs-literature timing in semantic space without forcing a
one-to-one paper-topic ↔ team-topic pairing.

## `reporting.ipynb` (legacy)

The original single-notebook joint visualisation (joint UMAP scatters, density
ratio, and precedence in one place). Superseded by the modular notebooks above,
which share one persisted projection and add hierarchy-level colouring; kept for
reference.
