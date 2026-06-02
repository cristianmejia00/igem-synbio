# 04 — Topic Hierarchy

Builds a three-level topic hierarchy (low → mid → high) for each corpus by
cutting BERTopic's agglomerative merge tree at two levels, then assigns
AI-generated names to the mid and high groups.

## Setup

Install the project-level dependencies (`requirements.txt`). The naming
notebooks also require an OpenAI API key stored at `03-topic_names/openai.key`
(read from there directly).

## Layout

The pipeline is split **per corpus** so the teams and papers hierarchies can be
re-run independently — updating one no longer recomputes (or re-issues the LLM
naming calls for) the other.

```text
04-topic_hierarchy/
├── prompts_hierarchy.yaml   theme definition + hierarchy-naming prompt (shared)
├── aux/                     shared helper functions imported by every notebook
├── 01-teams/                iGEM teams hierarchy
└── 02-papers/               SynBio OpenAlex papers hierarchy
```

Each of `01-teams/` and `02-papers/` contains the same two notebooks, differing
only in a small CONFIG block at the top (`PREFIX`, `ID_COL`, `YEAR_COL`,
`RAW_FILE`):

| Notebook | Purpose |
|---|---|
| `get_topic_hierarchy.ipynb` | Build the low → mid → high hierarchy from the BERTopic merge tree; auto-select the mid/high levels by silhouette; write the three report tables → `assets/reports/` |
| `name_hierarchy_levels.ipynb` | Name the mid and high groups with an LLM (OpenAI function calling) and add `mid_name` / `high_name` to the hierarchy table |

Run `get_topic_hierarchy.ipynb` before `name_hierarchy_levels.ipynb` within a
corpus folder.

## Shared resources

### `prompts_hierarchy.yaml`

The theme definition ("Synthetic Biology") and the system prompt used to name
hierarchy groups. Edit it to adjust style or framing without touching code.

### `aux/`

Reusable mechanics are factored out of the notebooks so each notebook stays a
thin, dataset-specific orchestration layer. Imported via
`sys.path.insert(0, str(Path.cwd().parent))` then `from aux.<module> import …`.

| Module | Key functions |
|---|---|
| `paths.py` | `PROJECT_ROOT`, `STEP_DIR`, `ASSETS_DIR`, `MODELS_DIR`, `EMBEDDINGS_DIR`, `REPORTS_DIR`, `PROMPTS_PATH`, `OPENAI_KEY_PATH`, `OPENAI_MODEL`, `SEED`, `HIGH_K_MIN/MAX`, `set_seed()` |
| `hierarchy.py` | `load_hierarchy_inputs()`, `select_hierarchy_levels()`, `write_hierarchy_reports()`, and the building blocks `build_hierarchy_maps()`, `get_topic_embeddings()`, `select_best_k()`, `auto_mid_k_range()`, `build_topic_hierarchy_df()`, `build_doc_map()`, `build_name_map()`, `build_summary()` |
| `naming.py` | `load_prompts()`, `make_client()`, `build_system_prompt()`, `build_rename_tool()`, `build_user_message()`, `name_hierarchy_level()`, `load_naming_inputs()`, `save_named_hierarchy()` |

Paths in `paths.py` resolve from the module's own location, so the notebooks
work regardless of the kernel's working directory, and `prompts_hierarchy.yaml`
/ `03-topic_names/openai.key` are always found.

## Outputs (per corpus, in `assets/reports/`)

- `<prefix>_topic_hierarchy_map.tsv` — document-level mapping (`<id>, low, mid, high`)
- `<prefix>_topic_name_hierarchy.tsv` — low-level names mapped to `low, mid, high`,
  plus `mid_name` / `high_name` after the naming notebook runs
- `<prefix>_topic_hierarchy_summary.tsv` — mid/high group counts and year stats

(`<prefix>` is `teams` or `papers`.) These feed `05-reporting` and
`06-deliverables`.

## How levels are chosen

- **high** — k searched over `[HIGH_K_MIN, HIGH_K_MAX]` (default `[4, 11]`),
  best silhouette wins.
- **mid** — k searched over `[HIGH_K_MAX + 1, n_low // 3]` (kept separate from
  the high level and from the low level), best silhouette wins.

Outlier documents (low = −1) are carried through with mid = high = −1.
