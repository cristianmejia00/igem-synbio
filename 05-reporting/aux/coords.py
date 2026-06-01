"""Joint UMAP projection: compute once, persist, and load for every plot.

This module is the single source of truth for coordinates (e1). The projection
stacks the papers and teams embeddings and fits **one** 2D UMAP, so a paper and
a team project that land near each other are directly comparable. The resulting
``(x, y)`` are saved to ``assets/reports`` and reloaded by all downstream
notebooks — nothing recomputes or re-stretches them.

The plotting frames carry the three hierarchy levels side by side so a single
frame can be coloured by micro / meso / macro (e2):

    micro -> ("topic",  "low_name")    # low-level BERTopic clusters
    meso  -> ("mid",    "mid_name")    # mid-level hierarchy groups
    macro -> ("high",   "high_name")   # high-level hierarchy groups
"""
from itertools import cycle

import numpy as np
import pandas as pd
from umap import UMAP

from .paths import (
    ASSETS_DIR, EMBEDDINGS_DIR, MODELS_DIR, OUTLIER_COLOR, PALETTE,
    PAPERS_COORDS_FILE, REPORTS_DIR, SEED, TEAMS_COORDS_FILE,
)

# level name -> (id column, label column) in the plot frames
LEVELS = {
    "micro": ("topic", "low_name"),
    "meso": ("mid", "mid_name"),
    "macro": ("high", "high_name"),
}


# ── 1. Compute / persist / load the joint projection ──────────────────────────
def compute_joint_umap(papers_emb, teams_emb, seed=SEED, n_neighbors=15, min_dist=0.1):
    """Fit one 2D UMAP on the stacked corpora; return (papers_xy, teams_xy)."""
    all_emb = np.vstack([papers_emb, teams_emb])
    umap_2d = UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        metric="cosine", random_state=seed,
    )
    all_xy = umap_2d.fit_transform(all_emb)
    return all_xy[:len(papers_emb)], all_xy[len(papers_emb):]


def save_coords(papers_ids, papers_xy, teams_ids, teams_xy, reports_dir=REPORTS_DIR):
    """Persist per-document coordinates keyed by their id columns."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": papers_ids, "x": papers_xy[:, 0], "y": papers_xy[:, 1]}).to_csv(
        reports_dir / PAPERS_COORDS_FILE, sep="\t", index=False)
    pd.DataFrame({"UT": teams_ids, "x": teams_xy[:, 0], "y": teams_xy[:, 1]}).to_csv(
        reports_dir / TEAMS_COORDS_FILE, sep="\t", index=False)


def load_coords(reports_dir=REPORTS_DIR):
    """Load the persisted papers / teams coordinate tables."""
    papers = pd.read_csv(reports_dir / PAPERS_COORDS_FILE, sep="\t")
    teams = pd.read_csv(reports_dir / TEAMS_COORDS_FILE, sep="\t")
    return papers, teams


def shared_limits(df_papers, df_teams, pad=1.5):
    """Axis limits spanning **both** corpora — the same canvas for every plot.

    Because the limits come from the joint cloud, a teams-only plot occupies its
    real sub-region instead of being stretched to fill the frame.
    """
    xs = np.concatenate([df_papers["x"].to_numpy(), df_teams["x"].to_numpy()])
    ys = np.concatenate([df_papers["y"].to_numpy(), df_teams["y"].to_numpy()])
    return (xs.min() - pad, xs.max() + pad), (ys.min() - pad, ys.max() + pad)


# ── 2. Build plotting frames (coords + hierarchy levels + metadata) ───────────
def _build_corpus_frame(coords, doc_topics, id_col, hierarchy, raw, year_col):
    """Merge coords + topic ids + hierarchy levels + raw year/citation columns."""
    df = doc_topics.merge(coords, on=id_col, how="inner")           # + x, y
    h = hierarchy.rename(columns={"global_name": "low_name", "low": "topic"})
    df = df.merge(h, on="topic", how="left")                        # + mid/high(+names)
    df = df.merge(raw, on=id_col, how="left")                       # + year/citations
    return df.rename(columns={year_col: "year"})


def load_plot_data(reports_dir=REPORTS_DIR, models_dir=MODELS_DIR, assets_dir=ASSETS_DIR):
    """Load everything the plots need, returning unified, hierarchy-aware frames.

    Returns ``(df_papers, df_teams, xlim, ylim)``. Each frame has columns:
    ``x, y, topic, low_name, mid, high, mid_name, high_name, year`` (papers also
    carry ``cited_by_count``); ``id`` for papers and ``UT`` for teams.
    """
    papers_coords, teams_coords = load_coords(reports_dir)

    papers_dt = pd.read_csv(models_dir / "papers_doc_topics.txt", sep="\t")
    teams_dt = pd.read_csv(models_dir / "teams_doc_topics.txt", sep="\t")

    papers_h = pd.read_csv(reports_dir / "papers_topic_name_hierarchy.tsv", sep="\t")
    teams_h = pd.read_csv(reports_dir / "teams_topic_name_hierarchy.tsv", sep="\t")

    papers_raw = pd.read_csv(
        assets_dir / "synbio_openalex.txt", sep="\t",
        usecols=["id", "publication_year", "cited_by_count"],
    )
    papers_raw["publication_year"] = pd.to_numeric(papers_raw["publication_year"], errors="coerce")
    papers_raw["cited_by_count"] = (
        pd.to_numeric(papers_raw["cited_by_count"], errors="coerce").fillna(0).astype(int)
    )
    teams_raw = pd.read_csv(assets_dir / "igem.txt", sep="\t", usecols=["UT", "Year_y"])
    teams_raw["Year_y"] = pd.to_numeric(teams_raw["Year_y"], errors="coerce")

    df_papers = _build_corpus_frame(papers_coords, papers_dt, "id", papers_h, papers_raw, "publication_year")
    df_teams = _build_corpus_frame(teams_coords, teams_dt, "UT", teams_h, teams_raw, "Year_y")

    xlim, ylim = shared_limits(df_papers, df_teams)
    return df_papers, df_teams, xlim, ylim


# ── 3. Per-level colouring and label anchors ──────────────────────────────────
def _outlier_mask(df, level_id):
    """True where a row has no assigned group at this level (topic -1 or NaN)."""
    return ~(df[level_id].notna() & (df[level_id] != -1))


def assign_level_colors(df, level_id, palette=PALETTE, outlier_color=OUTLIER_COLOR):
    """Map each row to a colour by its group id at ``level_id`` (outliers grey)."""
    groups = sorted(
        g for g in df.loc[~_outlier_mask(df, level_id), level_id].unique()
    )
    cc = cycle(palette)
    cmap = {g: next(cc) for g in groups}
    colors = df[level_id].map(cmap)
    return colors.where(~_outlier_mask(df, level_id), outlier_color), cmap


def level_centroids(df, level_id, level_name, min_docs=1):
    """One label anchor per group at a level: median x/y, size, and name."""
    sub = df[~_outlier_mask(df, level_id)]
    g = sub.groupby(level_id).agg(
        cx=("x", "median"),
        cy=("y", "median"),
        n_docs=("x", "size"),
        label=(level_name, "first"),
    ).reset_index()
    return g[g["n_docs"] >= min_docs].reset_index(drop=True)
