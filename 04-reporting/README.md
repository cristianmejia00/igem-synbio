# 04 — Reporting

## Setup

All input files are the outputs of the previous pipeline steps
(datasets, embeddings, topic assignments, and topic names stored under
`assets/`).  No external API access is needed.

## Description

The notebook **`reporting.ipynb`** brings together every artefact
produced by the pipeline and generates publication-ready figures and
analytical tables.

### Citation normalisation

1. **Load and merge** — reads the papers dataset, document-level topic
   assignments, and topic names; joins them into a single table.
2. **Yearly normalisation** — because older papers accumulate more
   citations, three normalised metrics are computed within each
   publication year: a z-score of raw citations, a z-score of
   log-transformed citations, and a percentile rank.

### Cluster-level summary

3. **Aggregate by topic** — for each topic, computes average citations,
   the three normalised impact scores, the average publication year, the
   Price Index (share of papers from the last five years), and a
   volume-weighted recency rank.
4. **Impact vs Recency scatter** — one dot per topic; position encodes
   average year and impact, dot size encodes topic volume.
5. **Impact vs Price Index scatter** — same idea but using the Price
   Index on the horizontal axis.

### Joint UMAP projection

6. **Stack embeddings** — concatenates the 384-dimensional embedding
   vectors of both corpora (papers and iGEM teams) and projects them
   together into two dimensions with UMAP.  This ensures both datasets
   live in the same coordinate space.
7. **Papers scatter** — each paper is a dot, coloured by topic, sized by
   citations.  Topic labels are placed at cluster centroids.
8. **Teams scatter** — same layout for iGEM teams.
9. **Overlay plot** — papers appear as a muted grey background and teams
   in full colour, making it easy to see where the two corpora overlap.

### Density ratio analysis

10. **Kernel density estimation** — estimates a smooth density surface
    for each corpus over a 300×300 grid that covers the shared UMAP
    space.
11. **Log₂ density ratio** — computes the ratio of team density to paper
    density at every grid cell.  Positive values mean iGEM teams are
    more concentrated; negative values mean the literature dominates.
    Regions with negligible density from both corpora are masked out.
12. **Density ratio heatmap** — renders the ratio surface with a
    diverging colour map (blue = literature-dominated, red =
    teams-dominated), a centred colour bar, and topic labels for both
    corpora.

### Quantitative overlap

13. **Zone classification** — for each topic centroid, looks up the
    local density ratio and classifies it as *papers-only*,
    *teams-dominant*, or *overlap* (where the log₂ ratio is within ±1).
14. **Temporal precedence** — for each overlap-zone topic, uses
    spatial-neighbour queries to find the closest topics from the other
    corpus and compares their average publication years.  This reveals
    whether the literature or iGEM teams explored a given area first.
15. **Export tables** — saves two TSV files to `assets/reports/`:
    - `igem_preceded.tsv` — overlap areas where iGEM activity came
      before the academic literature.
    - `literature_preceded.tsv` — overlap areas where the literature
      came first.
