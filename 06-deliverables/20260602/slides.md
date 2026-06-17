---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 22px;
  }
  h1 { font-size: 1.6em; }
  h2 { font-size: 1.3em; }
  table { font-size: 0.7em; }
  img { max-height: 55vh; }
  img[alt~="half"] { max-height: 45vh; }
  .columns { display: flex; gap: 1.5em; }
  .columns > div { flex: 1; min-width: 0; }
  .small-table table { font-size: 0.55em; }
  .small-table td, .small-table th { padding: 2px 6px; }
---

# Comparative Topic-Model Analysis of iGEM and the Synthetic Biology Literature

**Authors:** _[Author names]_
Generated: 2026-04-08 21:53

---

## Background & Objective

**Synthetic biology** spans a growing body of **academic research** and
a worldwide community of student innovators competing in the **iGEM**
(International Genetically Engineered Machine) competition.

Despite shared goals, these two communities evolve on different timelines
and emphasize different topics.

### Objective

Map both corpora onto a **shared topic landscape** to identify:
- Where the two communities **converge** or **diverge**
- Which community addresses a topic **first** (temporal precedence)

---

## Data & Methods

<div style="display:flex; gap:2em;">
<div style="flex:1;">

### Data sources
| Corpus | Records |
|:-------|--------:|
| OpenAlex papers | 24,202 |
| iGEM team projects | 4,548 |

</div>
<div style="flex:1;">

### Pipeline
```
 1. Collect    → OpenAlex API + iGEM registry
       ↓
 2. Embed      → Sentence-Transformers
       ↓
 3. Cluster    → BERTopic (UMAP + HDBSCAN)
       ↓
 4. Analyze    → Hierarchy · Overlap · Precedence
```

</div>
</div>

---

## Results — Papers Topic Overview

<div class="columns">
<div>

**24,202** papers clustered into:
- **4** high-level groups
- **44** mid-level groups
- **263** low-level topics

Each dot is a topic. X = **Price Index**
(recent-paper fraction), Y = **citation
impact** (log-normalized).

</div>
<div>

![half Impact vs Price Index](../assets/reports/impact_vs_price_index.png)

</div>
</div>

---

## Results — Papers Topic Map

2-D UMAP projection of **24,202** papers, colored by topic.

![Papers UMAP](../assets/reports/umap_papers.png)

---

## Results — iGEM Teams & Overlap

<div class="columns">
<div>

**4,548** iGEM projects clustered into:
- **10** high-level groups
- **25** mid-level groups
- **161** low-level topics

Papers in grey; team projects in color.
Joint UMAP space enables direct
spatial comparison.

</div>
<div>

![half Overlay](../assets/reports/umap_overlay.png)

</div>
</div>

---

## Results — Density Heatmap

**Red** = iGEM-dominated, **blue** = literature-dominated, **white** = balanced.

![Heatmap](../assets/reports/umap_heatmap.png)

---

<!-- _class: small-table -->
## Results — Temporal Precedence

<div class="columns">
<div>

**Literature precedes iGEM**

| # | Topic | Δ yr |
|--:|:------|-----:|
| 1 | Synbio in Infectious Disease | +15.5 |
| 2 | Cytokine Gene Regulation | +13.5 |
| 3 | Fungal Enzyme & Pathogenicity | +10.1 |
| 4 | Viral Synbio & Vaccines | +8.7 |
| 5 | Vascular Gene & Receptor Eng. | +8.7 |
| 6 | High-Throughput Genetic Eng. | +7.1 |
| 7 | Synbio-Driven Biomaterials | +6.1 |
| 8 | Vaccine & Immunotherapy | +5.7 |
| 9 | Pathogen Detection via Synbio | +5.6 |
| 10 | Plant Gene Regulation | +5.4 |

</div>
<div>

**iGEM precedes Literature**

| # | Topic | Δ yr |
|--:|:------|-----:|
| 1 | AI in Synthetic Biology | -6.0 |
| 2 | Microbial Chassis Dev. | -4.5 |
| 3 | CRISPR in Synbio | -4.5 |
| 4 | AI-Driven Imaging & Data | -4.4 |
| 5 | Microalgal Synbio | -4.1 |
| 6 | CRISPR Control Technologies | -3.6 |
| 7 | Microbial Community Eng. | -3.2 |
| 8 | Plant Resilience Eng. | -3.2 |
| 9 | CRISPR-Based Bacterial Reg. | -3.0 |
| 10 | Synth. Promoter Eng. in Yeast | -3.0 |

</div>
</div>

_Δ yr = avg year (teams) − avg year (papers). Top 10 Δ yr each._
