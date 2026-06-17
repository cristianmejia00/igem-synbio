"""Temporal precedence between iGEM teams and the literature, and its charts.

For each *overlap-zone* paper topic we query the teams that fall within its
neighbourhood in the joint UMAP space and compare year distributions. Negative
deltas (teams minus papers) mean iGEM teams got there first.

The saved tables keep the original schema (``topic`` + ``global_name`` + the
``*_year`` / ``delta_*_years`` columns) because ``06-deliverables`` consumes
``igem_preceded.tsv`` and ``literature_preceded.tsv``.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

from .density import classify_zone
from .paths import REPORTS_DIR

_STATS = ("min", "p5", "q1", "mean", "median", "q3", "max")

# Human-readable label for each precedence statistic, used in chart titles/axes.
_STAT_LABELS = {
    "min": "earliest record",
    "p5": "earliest 5% (P5)",
    "q1": "first quartile (Q1)",
    "mean": "mean",
    "median": "median",
    "q3": "third quartile (Q3)",
    "max": "latest record",
}


def summarize_years(values, prefix):
    """min/p5/q1/mean/median/q3/max of a year series, prefixed in the column names.

    ``p5`` is the 5th percentile — the leading edge of a corpus, used for the
    stricter "earliest 5%" precedence comparison alongside the first quartile.
    """
    s = pd.Series(values).dropna().astype(float)
    if s.empty:
        return {f"{prefix}_{k}_year": np.nan for k in _STATS}
    return {
        f"{prefix}_min_year": round(float(s.min()), 2),
        f"{prefix}_p5_year": round(float(s.quantile(0.05)), 2),
        f"{prefix}_q1_year": round(float(s.quantile(0.25)), 2),
        f"{prefix}_mean_year": round(float(s.mean()), 2),
        f"{prefix}_median_year": round(float(s.median()), 2),
        f"{prefix}_q3_year": round(float(s.quantile(0.75)), 2),
        f"{prefix}_max_year": round(float(s.max()), 2),
    }


def _topic_centroids(df):
    """Low-level topic centroids with size, mean year, and low-level name."""
    sub = df[df["topic"] >= 0].copy()
    return sub.groupby("topic").agg(
        cx=("x", "median"), cy=("y", "median"), n_docs=("x", "size"),
        avg_year=("year", "mean"), global_name=("low_name", "first"),
    ).reset_index()


def compute_precedence(df_papers, df_teams, ratio_interp, min_nearby=3, radius_mult=1.5):
    """Build the overlap-precedence table.

    Returns ``(df_prec, papers_ctr, teams_ctr)`` where the centroid frames carry
    ``log2_ratio`` and a coverage ``zone``, and ``df_prec`` is sorted by
    ``delta_mean_years`` (most teams-first first).
    """
    papers_ctr = _topic_centroids(df_papers)
    papers_ctr["log2_ratio"] = ratio_interp(np.column_stack([papers_ctr["cy"], papers_ctr["cx"]]))
    papers_ctr["zone"] = papers_ctr["log2_ratio"].apply(classify_zone)

    teams_ctr = _topic_centroids(df_teams)
    teams_ctr["log2_ratio"] = ratio_interp(np.column_stack([teams_ctr["cy"], teams_ctr["cx"]]))
    teams_ctr["zone"] = teams_ctr["log2_ratio"].apply(classify_zone)

    teams_tree = cKDTree(df_teams[["x", "y"]].to_numpy())
    _df_p = df_papers[df_papers["topic"] >= 0]

    precedence = []
    for _, row in papers_ctr[papers_ctr["zone"] == "overlap"].iterrows():
        topic_pts = _df_p.loc[_df_p["topic"] == row["topic"], ["x", "y"]].to_numpy()
        dists = np.sqrt((topic_pts[:, 0] - row["cx"]) ** 2 + (topic_pts[:, 1] - row["cy"]) ** 2)
        radius = np.median(dists) * radius_mult

        nearby_idx = teams_tree.query_ball_point([row["cx"], row["cy"]], r=radius)
        paper_years = _df_p.loc[_df_p["topic"] == row["topic"], "year"].dropna()
        nearby_years = df_teams.iloc[nearby_idx]["year"].dropna()
        if len(nearby_years) < min_nearby or len(paper_years) == 0:
            continue

        rec = {
            "topic": row["topic"],
            "global_name": row["global_name"],
            "n_papers": int(row["n_docs"]),
            "n_nearby_teams": int(len(nearby_idx)),
        }
        rec.update(summarize_years(paper_years, prefix="papers"))
        rec.update(summarize_years(nearby_years, prefix="teams"))
        for stat in _STATS:
            rec[f"delta_{stat}_years"] = round(rec[f"teams_{stat}_year"] - rec[f"papers_{stat}_year"], 2)
        precedence.append(rec)

    df_prec = pd.DataFrame(precedence)
    if not df_prec.empty:
        df_prec = df_prec.sort_values("delta_mean_years").reset_index(drop=True)
    return df_prec, papers_ctr, teams_ctr


def save_precedence(df_prec, reports_dir=REPORTS_DIR):
    """Write the full table plus the iGEM-first / literature-first splits."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    df_prec.to_csv(reports_dir / "overlap_precedence_full.tsv", sep="\t", index=False)
    igem = df_prec[df_prec["delta_mean_years"] < 0].copy()
    lit = df_prec[df_prec["delta_mean_years"] > 0].copy()
    igem.to_csv(reports_dir / "igem_preceded.tsv", sep="\t", index=False)
    lit.to_csv(reports_dir / "literature_preceded.tsv", sep="\t", index=False)
    return igem, lit


def gather_year_pools(df_prec, df_papers, df_teams, min_nearby=3, radius_mult=1.5, max_chars=60,
                      stat="q1"):
    """Long-form (topic_label, year, corpus) records for the split-violin chart.

    Reuses the same neighbourhood rule as ``compute_precedence`` so the violins
    match the saved precedence stats. Rows are ordered by ``delta_{stat}_years``
    (``stat="q1"`` for the first quartile, ``"p5"`` for the earliest-5% view).
    Returns ``(plot_df, ordered_labels)``.
    """
    order = df_prec.sort_values(f"delta_{stat}_years", ascending=True).copy()
    order["topic_label"] = order["global_name"].astype(str).str.slice(0, max_chars)

    _df_p = df_papers[df_papers["topic"] >= 0]
    tree = cKDTree(df_teams[["x", "y"]].to_numpy())

    records = []
    for _, row in order.iterrows():
        psub = _df_p[_df_p["topic"] == row["topic"]]
        pyears = psub["year"].dropna()
        if psub.empty or pyears.empty:
            continue
        cx, cy = psub["x"].median(), psub["y"].median()
        dists = np.sqrt((psub["x"].to_numpy() - cx) ** 2 + (psub["y"].to_numpy() - cy) ** 2)
        radius = np.median(dists) * radius_mult
        idx = tree.query_ball_point([cx, cy], r=radius)
        tyears = df_teams.iloc[idx]["year"].dropna()
        if len(tyears) < min_nearby:
            continue
        records += [{"topic_label": row["topic_label"], "year": float(y), "corpus": "Papers"} for y in pyears.to_numpy()]
        records += [{"topic_label": row["topic_label"], "year": float(y), "corpus": "Teams"} for y in tyears.to_numpy()]

    plot_df = pd.DataFrame(records)
    ordered = order["topic_label"].tolist()
    if not plot_df.empty:
        plot_df = plot_df[plot_df["topic_label"].isin(ordered)].copy()
        plot_df["topic_label"] = pd.Categorical(plot_df["topic_label"], categories=ordered, ordered=True)
    return plot_df, ordered


# ── Charts ────────────────────────────────────────────────────────────────────
def plot_violin(plot_df, ordered_labels, out_path=None, dpi=180, stat="q1"):
    """Option A — horizontal split violins of year distributions per overlap topic.

    ``stat`` only labels the chart with the precedence criterion the rows are
    sorted by; pass ``gather_year_pools(..., stat=stat)`` to match the ordering.
    """
    import seaborn as sns
    fig_h = max(18, 0.34 * len(ordered_labels))
    fig, ax = plt.subplots(figsize=(14, fig_h), dpi=dpi)
    sns.violinplot(
        data=plot_df, y="topic_label", x="year", hue="corpus", split=True, orient="h",
        inner="quart", cut=0, bw_method=0.25, density_norm="width", linewidth=0.5,
        palette={"Papers": "#1f77b4", "Teams": "#ff7f0e"}, ax=ax,
    )
    ax.set_title(f"Overlap Topics: Year Distributions by Corpus (sorted by delta_{stat}_years)",
                 fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Topic label")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="Corpus", loc="lower right")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
    return fig, ax


def plot_dumbbell(df_prec, out_path=None, dpi=180, stat="q1"):
    """Option B — papers vs teams median years with IQR bands, per overlap topic.

    The markers/bands always summarise the full distribution (median + Q1-Q3);
    ``stat`` selects the precedence criterion the rows are ordered by
    (``"q1"`` first quartile, ``"p5"`` earliest 5%).
    """
    db = df_prec.sort_values(f"delta_{stat}_years", ascending=True).copy()
    db["topic_label"] = db["global_name"].astype(str).str.slice(0, 60)
    y = np.arange(len(db))

    fig_h = max(14, 0.30 * len(db))
    fig, ax = plt.subplots(figsize=(14, fig_h), dpi=dpi)
    ax.hlines(y=y, xmin=db["papers_median_year"], xmax=db["teams_median_year"],
              color="0.65", linewidth=1.0, alpha=0.8, zorder=1)
    ax.hlines(y - 0.12, db["papers_q1_year"], db["papers_q3_year"], color="#1f77b4", linewidth=2.2, alpha=0.8, zorder=2)
    ax.hlines(y + 0.12, db["teams_q1_year"], db["teams_q3_year"], color="#ff7f0e", linewidth=2.2, alpha=0.8, zorder=2)
    ax.scatter(db["papers_median_year"], y, s=22, color="#1f77b4", zorder=3)
    ax.scatter(db["teams_median_year"], y, s=22, color="#ff7f0e", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(db["topic_label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Year")
    ax.set_ylabel("Topic label")
    ax.set_title(f"Overlap Topics: Papers vs Teams Year Dumbbell (sorted by delta_{stat}_years)",
                 fontsize=14)
    ax.grid(axis="x", alpha=0.25)
    legend_items = [
        Line2D([0], [0], color="#1f77b4", lw=2.2, label="Papers IQR (Q1-Q3)"),
        Line2D([0], [0], color="#ff7f0e", lw=2.2, label="Teams IQR (Q1-Q3)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=6, label="Papers median"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", markersize=6, label="Teams median"),
    ]
    ax.legend(handles=legend_items, loc="lower right", title="Statistics")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
    return fig, ax


def plot_diverging(df_prec, out_path=None, dpi=180, stat="q1"):
    """Option C — diverging precedence gap centred at zero.

    The highlighted dot/line is ``delta_{stat}_years`` (``"q1"`` first quartile,
    ``"p5"`` the stricter earliest-5% leading edge); the grey band always spans
    the delta IQR (delta_q1 to delta_q3) as a spread/robustness reference.
    """
    delta_col = f"delta_{stat}_years"
    stat_label = _STAT_LABELS.get(stat, stat)
    dg = df_prec.sort_values(delta_col, ascending=True).copy()
    dg["topic_label"] = dg["global_name"].astype(str).str.slice(0, 60)

    dg["support"] = np.minimum(dg["n_papers"], dg["n_nearby_teams"])
    sup = dg["support"].fillna(0)
    dg["marker_size"] = 20 + 120 * np.sqrt(sup / sup.max()) if sup.max() > 0 else 30

    def _gap_color(v):
        if v < -0.25:
            return "#b2182b"
        if v > 0.25:
            return "#2166ac"
        return "#7f7f7f"

    dg["gap_color"] = dg[delta_col].apply(_gap_color)
    y = np.arange(len(dg))

    fig_h = max(14, 0.30 * len(dg))
    fig, ax = plt.subplots(figsize=(15, fig_h), dpi=dpi)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.9, zorder=1)
    ax.grid(axis="x", alpha=0.20, zorder=0)
    ax.hlines(y=y, xmin=dg["delta_q1_years"], xmax=dg["delta_q3_years"], color="0.75",
              linewidth=1.6, alpha=0.9, zorder=2)
    for yi, val, col in zip(y, dg[delta_col], dg["gap_color"]):
        ax.plot([0, val], [yi, yi], color=col, linewidth=1.8, alpha=0.85, zorder=3)
    ax.scatter(dg[delta_col], y, s=dg["marker_size"], c=dg["gap_color"],
               edgecolor="white", linewidth=0.5, alpha=0.95, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(dg["topic_label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Precedence gap in years (teams - papers) using {stat_label}; "
                  "negative means teams preceded")
    ax.set_ylabel("Topic label")
    ax.set_title(f"Diverging precedence gap by topic (sorted by {delta_col})", fontsize=14)

    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    ax.text(xmin + 0.02 * span, -1.8, "Teams preceded", color="#b2182b", fontsize=10, ha="left", va="bottom")
    ax.text(xmax - 0.02 * span, -1.8, "Papers preceded", color="#2166ac", fontsize=10, ha="right", va="bottom")
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#b2182b", markersize=7, label=f"Teams preceded ({delta_col} < 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac", markersize=7, label=f"Papers preceded ({delta_col} > 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f", markersize=7, label="Near balance"),
        Line2D([0], [0], color="0.75", lw=1.6, label="Delta IQR (Q1 to Q3)"),
        Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="Zero gap"),
    ]
    ax.legend(handles=legend_items, loc="lower right", title="Encoding")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
    return fig, ax
