"""Topic scatter maps on the unified joint-UMAP canvas.

Every map uses the **shared** ``xlim``/``ylim`` from the joint projection plus an
equal aspect ratio, so papers, teams, and the overlay share one geometry and a
teams-only map is never stretched to fill the frame (e1). Colouring is selected
by hierarchy level and labels by placement style (e2).
"""
import numpy as np
import matplotlib.pyplot as plt

from .coords import LEVELS, OUTLIER_COLOR, assign_level_colors, level_centroids, _outlier_mask
from .labels import add_side_labels, add_overlay_labels


def _point_sizes(df, size_col, base=4.0, scale=30.0):
    """Log-scaled marker sizes from ``size_col`` (constant when absent)."""
    if size_col and size_col in df.columns:
        vals = df[size_col].fillna(0).clip(lower=0)
        mx = vals.max()
        if mx > 0:
            return (base + scale * (np.log1p(vals) / np.log1p(mx))).to_numpy()
    return np.full(len(df), base)


def _apply_canvas(ax, xlim, ylim, title):
    """Pin the shared limits + equal aspect and strip the axes."""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.axis("off")


def plot_topic_map(df, level, label_style, xlim, ylim, title, out_path=None,
                   size_col=None, min_docs=20, figsize=(12, 12), dpi=150):
    """One topic map: colour by ``level`` (micro/meso/macro), label by ``label_style``.

    ``label_style`` is ``"side"`` (gutter labels with connectors) or ``"overlay"``
    (labels on top of clusters). Outliers are drawn underneath in grey.
    """
    level_id, level_name = LEVELS[level]
    colors, _ = assign_level_colors(df, level_id)
    omask = _outlier_mask(df, level_id).to_numpy()
    sizes = _point_sizes(df, size_col)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(df.loc[omask, "x"], df.loc[omask, "y"], c=OUTLIER_COLOR, s=sizes[omask],
               alpha=0.15, edgecolors="none", zorder=1)
    ax.scatter(df.loc[~omask, "x"], df.loc[~omask, "y"], c=colors[~omask], s=sizes[~omask],
               alpha=0.5, edgecolors="none", zorder=2)

    _apply_canvas(ax, xlim, ylim, title)

    centroids = level_centroids(df, level_id, level_name, min_docs=min_docs)
    if label_style == "side":
        add_side_labels(ax, centroids, xlim, ylim, fontsize=5.5, max_labels_per_side=36)
        fig.subplots_adjust(left=0.02, right=0.98)
    elif label_style == "overlay":
        add_overlay_labels(ax, centroids, xlim, ylim, size_col="n_docs")
    else:
        raise ValueError(f"label_style must be 'side' or 'overlay', got {label_style!r}")

    if out_path is not None:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig, ax


def plot_overlay(df_papers, df_teams, xlim, ylim, level="micro",
                 title="Overlay — Papers (muted) + Teams (colored)",
                 out_path=None, min_docs=10, figsize=(12, 12), dpi=150):
    """Papers as a muted grey backdrop with teams drawn on top and side-labelled."""
    level_id, level_name = LEVELS[level]
    colors, _ = assign_level_colors(df_teams, level_id)
    omask = _outlier_mask(df_teams, level_id).to_numpy()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(df_papers["x"], df_papers["y"], c=OUTLIER_COLOR, s=1.5, alpha=0.8, edgecolors="none", zorder=1)
    ax.scatter(df_teams.loc[omask, "x"], df_teams.loc[omask, "y"], c="#a0a0a0", s=6, alpha=0.25,
               edgecolors="none", zorder=2)
    ax.scatter(df_teams.loc[~omask, "x"], df_teams.loc[~omask, "y"], c=colors[~omask], s=12,
               alpha=0.7, edgecolors="white", linewidth=0.3, zorder=3)

    centroids = level_centroids(df_teams, level_id, level_name, min_docs=min_docs)
    add_side_labels(ax, centroids, xlim, ylim, fontsize=6, min_gap_frac=0.026, max_labels_per_side=30)

    _apply_canvas(ax, xlim, ylim, title)
    fig.subplots_adjust(left=0.18, right=0.82)
    if out_path is not None:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig, ax
