# igem-synbio

Comparative topic-model analysis of the **iGEM competition** and the
**synthetic-biology academic literature**.

The pipeline downloads and cleans both datasets, computes
sentence-transformer embeddings, discovers topics with BERTopic, names
them with GPT, and produces publication-ready figures that visualise
where iGEM teams and the scientific literature converge or diverge — and
which community got there first.

## Repository structure

| Folder | Purpose |
|--------|---------|
| `00-IGEM_teams_dataset/` | Load, clean, and export the iGEM teams dataset |
| `01-SynBio_OpenAlex_dataset/` | Download and merge synthetic-biology articles from OpenAlex |
| `02-topic_model/` | Compute embeddings, fit BERTopic models, and evaluate hyperparameters |
| `03-topic_names/` | Generate human-readable topic names with an LLM |
| `04-topic_hierarchy/` | Build a low → mid → high topic hierarchy and name the groups |
| `05-reporting/` | Produce figures, tables, and overlap/precedence analysis |
| `06-deliverables/` | Assemble the final spreadsheets and slide deck |
| `assets/` | All data the pipeline produces, in dated run folders (git-ignored) |
| `run_paths.py` | Shared helper for the run folders, and a status command |
| `archive/` | Retired material, not part of the pipeline (git-ignored) |

Each folder has its own **README** with a detailed description of the
notebooks it contains.  The folders are numbered in execution order.

## Where results live

Everything the code produces goes to `assets/<YYYY-MM-DD>/<stage>/`: one folder
per run date, with one sub-folder per code folder (`00` … `06`).

```text
assets/2026-09-21/
  00/  igem.txt, stats_report.md, figures/
  01/  synbio_openalex_PART_*.txt, synbio_openalex.txt, stats_report.md, figures/
  02/  embeddings, corpora, topic models, topic info, document topics
  03/  topic names
  04/  topic hierarchy tables
  05/  reporting tables and figures
  06/  Excel deliverables
```

- **Run date** = today, fixed when a notebook starts. Override with
  `IGEM_RUN_DATE=YYYY-MM-DD`. Running again on the same day overwrites that day's
  files.
- **Setup scripts.** Folders `02`–`06` each have a `setup_run.py`, called from the
  first cell of every notebook (or run directly: `python setup_run.py`). It copies
  the upstream files that stage reads into today's folder when they are missing,
  taking them from the newest earlier run and printing
  `↳ copied 02/papers_doc_topics.txt from 2026-06-11`. So each date folder holds
  everything its results were built from, and you can re-run a single step without
  re-running the whole pipeline. Copies keep their original modification time, and
  on macOS they take no extra disk space. `00` and `01` are independent starting
  points and need no setup.
- **Consistency check.** A notebook stops if a file it reads was built on data
  that has since changed (for example, topic names from before the topic model was
  re-run), and names the step to re-run. Re-runs that produce identical files don't
  trigger it. Set `IGEM_ALLOW_STALE=1` to continue anyway with a warning.
- **Status.** `python run_paths.py [YYYY-MM-DD]` lists the runs and, for a given run
  (default: the newest), which files were produced that day and which were carried
  over from earlier runs, plus any consistency problems.

## Environment

Latest execution environment used by the authors (cristianmejia00):

| Component | Version |
|-----------|---------|
| OS | macOS |
| Editor | Visual Studio Code |
| Python | 3.13.9 |
| Package manager | pip (conda base environment) |

## Initial setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/cristianmejia00/igem-synbio.git
   cd igem-synbio
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   This installs every library needed across all pipeline steps.
   See `requirements.txt` for the full pinned list.

4. **OpenAI API key** (only for step 03)

   Create a plain-text file at `03-topic_names/openai.key` containing
   your API key.  The file is git-ignored.

5. **Run the notebooks in order**

   Open each folder sequentially (`00` → `01` → `02` → `03` → `04` →
   `05` → `06`) and execute the notebooks inside.  Each folder's README
   explains its specific requirements and execution details.  To re-run
   only part of the pipeline, start from the step you changed; the setup
   scripts pick up everything else from the newest earlier run.

## Reproducibility

All stochastic steps use a fixed random seed (`SEED = 42`).  Combined
with the pinned dependency versions in `requirements.txt`, this should
reproduce identical results on the same platform.
