# 05 — Topic Hierarchy

Builds a three-level topic hierarchy (low → mid → high) for each corpus
by cutting BERTopic's agglomerative merge tree at two levels, then
assigns AI-generated names to the mid and high groups.

## Setup

Install the project-level dependencies (`requirements.txt`).  The
naming notebook also requires an OpenAI API key stored at
`03-topic_names/openai.key`.

## Shared module

### `hierarchy_utils.py`

Reusable functions imported by both hierarchy notebooks:

| Function | Purpose |
|---|---|
| `build_hierarchy_maps()` | Build the merge tree and extract cluster maps for every k |
| `get_topic_embeddings()` | Extract and align topic embeddings for non-outlier topics |
| `select_best_k()` | Score candidate k values via silhouette and select the best |
| `auto_mid_k_range()` | Derive the search range for the mid-level k from low-topic count |
| `build_topic_hierarchy_df()` | Create the (low, mid, high) mapping table |
| `build_doc_map()` | Merge document-level topic assignments with the hierarchy |
| `build_name_map()` | Merge topic names with the hierarchy |
| `build_summary()` | Compute per-group counts and year statistics for mid and high levels |

## Notebooks

### `get_papers_topic_hierarchy.ipynb`

Hierarchy for the **papers** topic model (~240 low-level topics).

- **MID_K** auto-selected by silhouette score over [HIGH_K_MAX + 1, n // 3]
- **HIGH_K** auto-selected from [4, 11] by silhouette score

**Outputs:**
- `assets/reports/papers_topic_hierarchy_map.tsv` — document-level mapping (ID, low, mid, high)
- `assets/reports/papers_topic_name_hierarchy.tsv` — topic names mapped to hierarchy (global_name, low, mid, high)
- `assets/reports/papers_topic_hierarchy_summary.tsv` — mid/high group summary stats

### `get_teams_topic_hierarchy.ipynb`

Hierarchy for the **iGEM teams** topic model (161 low-level topics).

- **MID_K** auto-selected by silhouette score over [HIGH_K_MAX + 1, n // 3]
- **HIGH_K** auto-selected from [4, 11] by silhouette score

**Outputs:**
- `assets/reports/teams_topic_hierarchy_map.tsv` — document-level mapping (UT, low, mid, high)
- `assets/reports/teams_topic_name_hierarchy.tsv` — topic names mapped to hierarchy (global_name, low, mid, high)
- `assets/reports/teams_topic_hierarchy_summary.tsv` — mid/high group summary stats

### `name_hierarchy_levels.ipynb`

Assigns globally unique, publication-ready names to mid and high groups
for **both** corpora using OpenAI function calling (gpt-4.1-nano).

For each group the LLM receives the low-level sub-topics (name +
description) that belong to it and returns a single name.  Prompts are
defined in `prompts_hierarchy.yaml`.

Run this **after** both hierarchy notebooks.

**Updates:**
- `assets/reports/papers_topic_name_hierarchy.tsv` — adds `mid_name`, `high_name` columns
- `assets/reports/teams_topic_name_hierarchy.tsv` — adds `mid_name`, `high_name` columns
