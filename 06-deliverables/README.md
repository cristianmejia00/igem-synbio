# 06 — Deliverables

Shareable outputs generated from the pipeline results (notebooks 00–05).

## Contents

| File | Description |
|:-----|:------------|
| `slides.md` | Marp slide deck (8 slides). Edit directly; preview with the [Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode) extension. |
| `synbio_papers.xlsx` | Papers workbook with 3 sheets: **Cluster Summary**, **Paper Assignments**, **Literature Preceded**. |
| `igem_teams.xlsx` | Teams workbook with 3 sheets: **Cluster Summary**, **Team Assignments**, **iGEM Preceded**. |
| `deliverables.ipynb` | Notebook that builds the two Excel files from upstream TSV/TXT assets. |

## How to use

1. Run all upstream notebooks (00–05) so that `assets/reports/` is populated.
2. Run `deliverables.ipynb` to generate the Excel workbooks.
3. Edit `slides.md` as needed — images reference `../assets/reports/*.png`.
