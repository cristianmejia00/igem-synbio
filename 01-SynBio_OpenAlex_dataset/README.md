# 01 — Synthetic Biology OpenAlex Dataset

## Setup

No API key is needed — the OpenAlex API is free and open.  
A contact e-mail is included in the query URL so that OpenAlex can reach
out if usage is unusual; update it in `get_synbio_data.ipynb` if needed.

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
API query-length limits.  Every part also includes NOT clauses that
exclude false positives related to photosynthesis, fuel cells, and
similar terms.

### `get_synbio_data.ipynb`

1. **Read configuration** — loads `query.yaml` and prints each query
   part with its expected result count.
2. **Build API URLs** — constructs OpenAlex search URLs with filters
   for article type, non-retracted status, and the year range.
3. **Download** — pages through the full result set of each query part
   using cursor-based pagination with built-in rate limiting.
4. **Transform** — flattens nested JSON fields (abstract inverted index,
   author/institution lists, journal metadata, concept tags) into a
   tabular format.  Country codes are mapped to full names using the
   ISO-3166 standard.
5. **Save** — writes one TSV file per query part into
   `assets/openalex_data/`.

### `merge_and_describe.ipynb`

1. **Merge** — reads all per-part TSV files and concatenates them into a
   single DataFrame.
2. **Deduplicate** — removes articles that appear in more than one query
   part.
3. **Handle missing values** — fills blanks with empty strings or zeros
   as appropriate.
4. **Export** — saves the merged dataset to `assets/synbio_openalex.txt`.
5. **Summary statistics** — prints dataset dimensions, coverage of key
   fields, and citation statistics.
6. **Visualisations** — bar charts of articles per year, top-20
   countries, institutions, and concepts.
