# 04 — Reporting

## Setup

All input files are the outputs of the previous pipeline steps
(datasets, embeddings, topic assignments, and topic names stored under
`assets/`).  No external API access is needed.

## Notebooks

### `cluster_summary_papers.ipynb`

Cluster-level summaries and figures for the **papers** topic model.

1. **Load and merge** — reads the papers dataset, document-level topic
   assignments, and topic names; joins them into a single table.
2. **Yearly normalisation** — computes three citation-normalised metrics
   within each publication year: z-score of raw citations, z-score of
   log-transformed citations, and percentile rank.
3. **Aggregate by topic** — for each topic, computes average citations,
   the three normalised impact scores, the average publication year, the
   Price Index (share of papers from the last five years), and a
   volume-weighted recency rank.
4. **Impact vs Recency scatter** — one dot per topic; position encodes
   average year and impact, dot size encodes topic volume.
5. **Impact vs Price Index scatter** — same idea but using the Price
   Index on the horizontal axis.

**Output:** `assets/reports/cluster_summary.tsv`

### `cluster_summary_IGEM.ipynb`

Cluster-level summaries for the **iGEM teams** topic model (year-based
normalisation; no citation metrics).

1. **Load and merge** — reads the iGEM dataset, team-level topic
   assignments, and topic names.
2. **Aggregate by topic** — computes team counts, average year, Price
   Index, year-rank recency, and top-country distribution per topic.
3. **Recency scatter** — one dot per topic showing volume vs recency.

**Output:** `assets/reports/cluster_summary_igem.tsv`

### `reporting.ipynb`

Joint visualisation and quantitative comparison of both topic models in
a shared embedding space.

1. **Joint UMAP projection** — concatenates the 384-dim embedding
   vectors of both corpora and projects them together into 2D with UMAP,
   so proximity between a paper and a team project is directly
   interpretable.
2. **Papers scatter** — each paper is a dot, coloured by topic, sized by
   citations.  Topic labels are placed at cluster centroids.
3. **Teams scatter** — same layout for iGEM teams.
4. **Overlay plot** — papers appear as a muted grey background and teams
   in full colour, making it easy to see where the two corpora overlap.
5. **Kernel density estimation** — estimates a smooth density surface
   for each corpus over a 300×300 grid covering the shared UMAP space.
6. **Log₂ density ratio** — computes the ratio of team density to paper
   density at every grid cell.  Positive values mean iGEM teams are more
   concentrated; negative values mean the literature dominates.
7. **Density ratio heatmap** — renders the ratio surface with a
   diverging colour map (blue = literature-dominated, red =
   teams-dominated), a centred colour bar, and topic labels.
8. **Zone classification** — classifies each topic centroid as
   *papers-only*, *teams-dominant*, or *overlap* based on the local
   density ratio.
9. **Temporal precedence** — for each overlap-zone topic, uses
   spatial-neighbour queries to compare average publication years between
   corpora, revealing which explored a given area first.

**Outputs:**
- `assets/reports/igem_preceded.tsv`
- `assets/reports/literature_preceded.tsv`
