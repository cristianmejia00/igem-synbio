# 01 — Synthetic Biology OpenAlex Dataset

## Setup

**OpenAlex API key.** OpenAlex meters its API. Requests without a key share a
small free daily budget ($0.10, about 1,000 requests) with everyone on the same
network IP address, so on a busy network they can fail with
`429 Too Many Requests` until the budget resets at midnight UTC. Get a free key
(<https://help.openalex.org/api/authentication/>) and save it, on its own, in
`openalex.key` in this folder. The file is git-ignored (`*.key`), and the key is
sent as a request header, so it never appears in URLs or notebook output. A full
download takes about 190 requests.

A contact e-mail is included in the query URL so that OpenAlex can reach
out if usage is unusual; update it in `build_query_url()` in
`get_synbio_data.ipynb` if needed. The notebook also downloads an ISO-3166
country-code table from GitHub, so it needs internet access.

## Description

This folder contains two notebooks and a configuration file that
together download, merge, and describe the academic-literature dataset
used throughout the project. All outputs go to today's run folder,
`assets/<date>/01/` (see *Where results live* in the root README).

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
   (200 records per page) with a pause between requests. Brief rate limits
   and server errors are retried; if the daily budget is used up, the notebook
   stops with OpenAlex's explanation and the time until the budget resets.
4. **Transform** — flattens nested JSON fields into a tabular format: the
   abstract is rebuilt from the inverted index; authors, institutions,
   and countries are joined into semicolon-separated strings; the journal is
   split into `source_id` / `source_name`; concepts with a score ≥ 0.3
   are kept. Country codes are mapped to full names using the ISO-3166
   standard. Duplicate IDs within a part are dropped.
5. **Save** — writes one TSV file per query part,
   `synbio_openalex_PART_<n>.txt`.

### `merge_and_describe.ipynb`

1. **Merge** — reads all `synbio_openalex_PART_*.txt` files and
   concatenates them into a single DataFrame. If today's run has no PART files
   (no download today), all of them are copied in from the single newest
   earlier run that has them. The merge stops if any part listed in
   `query.yaml` is missing (an interrupted download).
2. **Deduplicate** — removes articles that appear in more than one query
   part (by OpenAlex `id`).
3. **Filter** — keeps only English articles (`language == "en"`), drops
   articles without an abstract, and **drops articles published before
   2004** so the corpus covers the same period as iGEM.
4. **Handle missing values** — fills blanks with empty strings (text
   columns) or zeros (numeric columns).
5. **Export** — saves the merged dataset as `synbio_openalex.txt`.
6. **Summary statistics** — prints dataset dimensions, coverage of key
   fields, and citation statistics.
7. **Visualisations** — bar charts of articles per year and the top-20
   countries, institutions, and concepts; line charts of the top-6
   countries over time (raw counts and share of each year's papers),
   using the same country colours as the iGEM charts.
8. **Stats report** — writes `stats_report.md`, with its figures in
   `figures/` (`yearly_trend.png`, `country_top20.png`,
   `country_trend_raw.png`, `country_trend_norm.png`,
   `concepts_top20.png`).

> **Known mismatch.** The part files currently in `assets/2026-03-22/01/`
> also contain a `referenced_works` column, written by an earlier version
> of `get_synbio_data.ipynb`. The current version no longer requests that
> field, but the field-coverage cell in `merge_and_describe.ipynb` still
> reads it, so that cell will fail on freshly downloaded data.
