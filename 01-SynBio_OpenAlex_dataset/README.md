# 01 — Synthetic Biology OpenAlex Dataset

## Setup

No API key is needed — the OpenAlex API is free and open.
A contact e-mail is included in the query URL so that OpenAlex can reach
out if usage is unusual; update it in `build_query_url()` in
`get_synbio_data.ipynb` if needed. The notebook also downloads an ISO-3166
country-code table from GitHub, so it needs internet access.

## Description

This folder contains two notebooks and a configuration file that
together download, merge, and describe the academic-literature dataset
used throughout the project.

### `query.yaml`

A YAML file that defines the search strategy: year range (1950–2025) and
seven query parts.  The keyword expressions are adapted from **Appendix A**
of:

> Baaden, P., Bröring, S., Rennings, M., & Shapira, P. (2026).
> Researcher positions and the emergence of interdisciplinary scientific
> fields – The case of synthetic biology. *Research Policy*, 55(3),
> 105395. <https://doi.org/10.1016/j.respol.2025.105395>

Each part is a boolean keyword expression targeting synthetic-biology
terminology (e.g. "synthetic biology", "synthetic genome", "artificial
cell", "biobrick").  The query is split into parts because of OpenAlex
API query-length limits.  Parts 1–6 include NOT clauses that exclude
false positives (photosynthesis, "generation"/"general", cell phones,
fuel cells, and similar terms); part 7 (artificial nucleic acids and
BioBrick terms) has no NOT clause.

### `get_synbio_data.ipynb`

1. **Read configuration** — loads `query.yaml` and prints the year range
   and the name and opening text of each query part.
2. **Build API URLs** — constructs OpenAlex URLs that search titles and
   abstracts, filtered to articles, non-retracted works, and the year
   range, sorted by citation count.
3. **Download** — for each part, prints the number of matching articles,
   then pages through the full result set using cursor-based pagination
   (200 records per page) with a pause between requests.
4. **Transform** — flattens nested JSON fields into a tabular format: the
   abstract is rebuilt from the inverted index; authors, institutions,
   and countries are joined into semicolon-separated strings; the journal is
   split into `source_id` / `source_name`; concepts with a score ≥ 0.3
   are kept. Country codes are mapped to full names using the ISO-3166
   standard. Duplicate IDs within a part are dropped.
5. **Save** — writes one TSV file per query part to
   `assets/openalex_data/synbio_openalex_PART_<n>.txt`.

### `merge_and_describe.ipynb`

1. **Merge** — reads all `synbio_openalex_PART_*.txt` files and
   concatenates them into a single DataFrame.
2. **Deduplicate** — removes articles that appear in more than one query
   part (by OpenAlex `id`).
3. **Filter** — keeps only English articles (`language == "en"`), drops
   articles without an abstract, and **drops articles published before
   2004** so the corpus covers the same period as iGEM.
4. **Handle missing values** — fills blanks with empty strings (text
   columns) or zeros (numeric columns).
5. **Export** — saves the merged dataset to `assets/synbio_openalex.txt`.
6. **Summary statistics** — prints dataset dimensions, coverage of key
   fields, and citation statistics.
7. **Visualisations** — bar charts of articles per year and the top-20
   countries, institutions, and concepts; line charts of the top-6
   countries over time (raw counts and share of each year's papers),
   using the same country colours as the iGEM charts.
8. **Stats report** — writes `stats_report.md` in this folder, with its
   figures in `figures/` (`yearly_trend.png`, `country_top20.png`,
   `country_trend_raw.png`, `country_trend_norm.png`,
   `concepts_top20.png`). `figures/figures_20260520/` is an earlier
   snapshot.

> **Known mismatch.** The part files currently in `assets/openalex_data/`
> also contain a `referenced_works` column, written by an earlier version
> of `get_synbio_data.ipynb`. The current version no longer requests that
> field, but the field-coverage cell in `merge_and_describe.ipynb` still
> reads it, so that cell will fail on freshly downloaded data.
