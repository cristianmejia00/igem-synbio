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
| `04-reporting/` | Produce figures, tables, and overlap/precedence analysis |
| `assets/` | Intermediate and output data files (git-ignored) |

Each folder has its own **README** with a detailed description of the
notebooks it contains.  The folders are numbered in execution order.

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

   Open each folder sequentially (`00` → `01` → `02` → `03` → `04`)
   and execute the notebooks inside.  Each folder's README explains its
   specific requirements and execution details.

## Reproducibility

All stochastic steps use a fixed random seed (`SEED = 42`).  Combined
with the pinned dependency versions in `requirements.txt`, this should
reproduce identical results on the same platform.
