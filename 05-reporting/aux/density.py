"""Kernel-density comparison of the two corpora over the joint UMAP space.

Estimates a density for each corpus on a shared grid, plots
``log2(teams_density / papers_density)`` with a density-faded alpha channel, and
classifies topics into coverage zones (papers-only / overlap / teams-dominant).
All of this reuses the persisted joint coordinates, so the density map lives on
the same canvas as the scatter maps.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


def compute_density_ratio(papers_xy, teams_xy, xlim, ylim, grid_res=300, bw=0.15):
    """Return the log2 density-ratio grid, density-fade alpha, and an interpolator.

    The result dict carries everything the heatmap and the precedence code need:
    ``ratio_grid``, ``alpha_grid``, ``vmax``, ``grid_res`` and ``ratio_interp``
    (a ``(y, x) -> log2_ratio`` interpolator used to sample topic centroids).
    """
    kde_papers = gaussian_kde(papers_xy.T, bw_method=bw)
    kde_teams = gaussian_kde(teams_xy.T, bw_method=bw)

    x_min, x_max = xlim
    y_min, y_max = ylim
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res))
    grid_pts = np.vstack([xx.ravel(), yy.ravel()])

    p = kde_papers(grid_pts)
    q = kde_teams(grid_pts)
    p /= p.sum()
    q /= q.sum()

    eps = 1e-12
    ratio_grid = np.log2((q + eps) / (p + eps)).reshape(xx.shape)
    vmax = np.percentile(np.abs(ratio_grid), 98)
    ratio_grid = np.clip(ratio_grid, -vmax, vmax)

    combined = p.reshape(xx.shape) + q.reshape(xx.shape)
    alpha_grid = np.clip(combined / np.percentile(combined, 95), 0, 1)
    alpha_grid = gaussian_filter(alpha_grid, sigma=3)

    x_vals = np.linspace(x_min, x_max, grid_res)
    y_vals = np.linspace(y_min, y_max, grid_res)
    ratio_interp = RegularGridInterpolator(
        (y_vals, x_vals), np.ma.filled(ratio_grid, fill_value=np.nan),
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    return {
        "ratio_grid": ratio_grid, "alpha_grid": alpha_grid, "vmax": vmax,
        "grid_res": grid_res, "ratio_interp": ratio_interp,
    }


def draw_density_heatmap(ax, dens, xlim, ylim, add_colorbar=True):
    """Render the density-ratio RGBA image + colourbar; return ``(norm, cmap)``."""
    vmax = dens["vmax"]
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    cmap = plt.cm.RdBu_r
    rgba = cmap(norm(dens["ratio_grid"]))
    rgba[..., 3] = dens["alpha_grid"]

    ax.imshow(rgba, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
              origin="lower", aspect="auto", interpolation="bilinear")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03)
        cbar.set_label("log2(teams density / papers density)", fontsize=10)
    return norm, cmap


def classify_zone(r):
    """|log2_ratio| > 1 means one corpus has >2x the other's density."""
    if pd.isna(r):
        return "unmapped"
    if r > 1:
        return "teams-dominant"
    if r < -1:
        return "papers-only"
    return "overlap"


def topic_zones(df, level_id, level_name, ratio_interp):
    """Per-topic centroid table with sampled density ratio and coverage zone."""
    sub = df[df[level_id].notna() & (df[level_id] != -1)]
    ctr = sub.groupby(level_id).agg(
        cx=("x", "median"), cy=("y", "median"), n_docs=("x", "size"),
        avg_year=("year", "mean"), label=(level_name, "first"),
    ).reset_index()
    ctr["log2_ratio"] = ratio_interp(np.column_stack([ctr["cy"].to_numpy(), ctr["cx"].to_numpy()]))
    ctr["zone"] = ctr["log2_ratio"].apply(classify_zone)
    return ctr


def curated_density_labels(papers_ctr, teams_ctr, n=10, max_chars=25):
    """Top teams-dense (red) + top papers-dense (blue) + a couple of balanced labels."""
    show_teams = teams_ctr.sort_values("log2_ratio", ascending=False).head(n)
    show_papers = papers_ctr.sort_values("log2_ratio", ascending=True).head(n)
    overlap = papers_ctr.loc[papers_ctr["log2_ratio"].abs() < 0.5].sort_values("n_docs", ascending=False).head(2)

    rows = []
    for _, r in show_teams.iterrows():
        rows.append({"cx": r["cx"], "cy": r["cy"], "label": str(r["label"])[:max_chars],
                     "fontsize": 7.0, "color": "#6c1a1a", "fontstyle": "normal",
                     "fontweight": "bold", "box_alpha": 0.78})
    for _, r in show_papers.iterrows():
        rows.append({"cx": r["cx"], "cy": r["cy"], "label": str(r["label"])[:max_chars],
                     "fontsize": 6.6, "color": "#1a1a6c", "fontstyle": "italic",
                     "fontweight": "normal", "box_alpha": 0.74})
    for _, r in overlap.iterrows():
        rows.append({"cx": r["cx"], "cy": r["cy"], "label": str(r["label"])[:max_chars],
                     "fontsize": 6.4, "color": "black", "fontstyle": "normal",
                     "fontweight": "normal", "box_alpha": 0.72})
    return pd.DataFrame(rows)


def dfprec_extreme_labels(df_prec, papers_ctr, n_extreme=10, max_chars=28):
    """Labels for the most teams-first / papers-first overlap topics (by delta_q1).

    Label text comes from ``df_prec`` (low-level ``global_name``); the anchor
    positions are taken from the paper-topic centroids (``topic, cx, cy``).
    """
    if df_prec.empty:
        return pd.DataFrame()
    low = df_prec.sort_values("delta_q1_years", ascending=True).head(n_extreme).copy()
    high = df_prec.sort_values("delta_q1_years", ascending=False).head(n_extreme).copy()
    low["extreme_group"] = "teams_preceded"
    high["extreme_group"] = "papers_preceded"
    label_df = pd.concat([low, high], ignore_index=True).drop_duplicates(subset=["topic"], keep="first")

    label_df = label_df.merge(papers_ctr[["topic", "cx", "cy"]], on="topic", how="left")

    rows = []
    for _, r in label_df.iterrows():
        if pd.isna(r["cx"]) or pd.isna(r["cy"]):
            continue
        color = "#7f0000" if r["extreme_group"] == "teams_preceded" else "#08306b"
        rows.append({"cx": r["cx"], "cy": r["cy"], "label": str(r["global_name"])[:max_chars],
                     "fontsize": 7.0, "color": color, "box_alpha": 0.80})
    return pd.DataFrame(rows)
