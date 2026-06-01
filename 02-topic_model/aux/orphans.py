"""Profiling helpers for outlier documents (topic = -1).

Used by the combined ``03-combined/orphans.ipynb`` notebook to understand *why*
the HDBSCAN outlier rate is high and whether any noise documents could
reasonably be reassigned.  Functions either return tables (left to the notebook
to ``display``) or draw a Matplotlib figure directly.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .paths import ASSETS_DIR, EMBEDDINGS_DIR, MODELS_DIR, REPORTS_DIR


def load_outlier_inputs(raw_file, embeddings_file, doc_topics_file, id_col):
    """Load metadata + embeddings + topic assignments for one dataset.

    Returns ``(df, embeddings, doc_topics)`` where ``df`` is the original
    metadata merged with the per-document ``topic`` label.
    """
    meta = pd.read_csv(ASSETS_DIR / raw_file, sep="\t")
    doc_topics = pd.read_csv(MODELS_DIR / doc_topics_file, sep="\t")
    embeddings = np.load(EMBEDDINGS_DIR / embeddings_file)
    df = meta.merge(doc_topics, on=id_col, how="inner")
    return df, embeddings, doc_topics


def outlier_summary(df, name, year_col):
    """Print basic outlier statistics for one dataset."""
    total = len(df)
    outliers = (df["topic"] == -1).sum()
    assigned = total - outliers
    print(f'── {name} {"─" * (50 - len(name))}')
    print(f"  Total docs   : {total:>8,}")
    print(f"  Assigned     : {assigned:>8,}  ({100 * assigned / total:.1f}%)")
    print(f"  Outliers (−1): {outliers:>8,}  ({100 * outliers / total:.1f}%)")
    print(f"  Year range   : {df[year_col].min():.0f} – {df[year_col].max():.0f}")
    print()


def plot_outlier_rate_by_year(df, year_col, title):
    """Bar chart of total vs. outlier counts per year, with outlier-rate line."""
    grouped = df.groupby(year_col)["topic"].agg(
        total="count",
        outliers=lambda s: (s == -1).sum(),
    )
    grouped["rate"] = grouped["outliers"] / grouped["total"]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(grouped.index, grouped["total"], color="steelblue", alpha=0.5, label="Total")
    ax1.bar(grouped.index, grouped["outliers"], color="tomato", alpha=0.7, label="Outliers")
    ax1.set_ylabel("Document count")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(grouped.index, grouped["rate"], "k-o", ms=3, label="Outlier rate")
    ax2.set_ylabel("Outlier rate")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")

    ax1.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_text_length(df, text_col, title):
    """Compare word-count distributions of outlier vs. assigned documents."""
    df = df.copy()
    df["_text_len"] = df[text_col].fillna("").str.split().str.len()
    outlier = df.loc[df["topic"] == -1, "_text_len"]
    assigned = df.loc[df["topic"] != -1, "_text_len"]

    if outlier.empty:
        print(f"  {title}: no outliers — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, max(outlier.quantile(0.99), assigned.quantile(0.99)) + 20, 10)
    ax.hist(assigned, bins=bins, alpha=0.5, density=True, label="Assigned")
    ax.hist(outlier, bins=bins, alpha=0.5, density=True, label="Outlier")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Short-text stats
    for threshold in [10, 20, 50]:
        n_short_out = (outlier <= threshold).sum()
        n_short_asg = (assigned <= threshold).sum()
        print(f"  ≤{threshold} words  →  outliers: {n_short_out:,} ({100*n_short_out/len(outlier):.1f}%)"
              f"  |  assigned: {n_short_asg:,} ({100*n_short_asg/len(assigned):.1f}%)")


def language_outlier_table(df, top_n=15):
    """Outlier rate by language (papers only). Returns a table to display."""
    if "language" not in df.columns:
        return None
    lang = df.groupby(["language", df["topic"].apply(
        lambda t: "outlier" if t == -1 else "assigned"
    )]).size().unstack(fill_value=0)
    lang["total"] = lang.sum(axis=1)
    lang = lang.sort_values("total", ascending=False).head(top_n)
    lang["outlier_rate"] = lang["outlier"] / lang["total"]
    return lang


def nearest_centroid_analysis(embeddings, topics_arr, name):
    """Cosine distance from each outlier to the nearest cluster centroid.

    Plots the outlier vs. assigned distance distributions, prints quantiles, and
    returns ``(dists_outliers, dists_assigned)``.
    """
    topics_arr = np.array(topics_arr)
    assigned_mask = topics_arr != -1
    outlier_mask = topics_arr == -1

    if outlier_mask.sum() == 0:
        print(f"  {name}: no outliers — skipping centroid analysis.")
        return np.array([]), np.array([])

    # Compute cluster centroids (mean embedding per topic)
    unique_topics = np.unique(topics_arr[assigned_mask])
    centroids = np.vstack([
        embeddings[topics_arr == t].mean(axis=0) for t in unique_topics
    ])

    # Cosine distance from every outlier to every centroid
    dists_outliers = cdist(embeddings[outlier_mask], centroids, metric="cosine").min(axis=1)
    dists_assigned = cdist(embeddings[assigned_mask], centroids, metric="cosine").min(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 60)
    ax.hist(dists_assigned, bins=bins, alpha=0.5, density=True, label="Assigned")
    ax.hist(dists_outliers, bins=bins, alpha=0.5, density=True, label="Outlier")
    ax.set_xlabel("Cosine distance to nearest centroid")
    ax.set_ylabel("Density")
    ax.set_title(f"{name} — distance to nearest cluster centroid")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Quantiles
    print(f"  {name} outlier distance quantiles:")
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        print(f"    {q:.0%}: {np.quantile(dists_outliers, q):.4f}")
    print(f"  {name} assigned distance quantiles:")
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        print(f"    {q:.0%}: {np.quantile(dists_assigned, q):.4f}")

    # How many outliers are closer than the median assigned distance?
    median_asg = np.median(dists_assigned)
    recoverable = (dists_outliers <= median_asg).sum()
    print(f"\n  Outliers closer than median assigned distance ({median_asg:.4f}): "
          f"{recoverable:,} / {outlier_mask.sum():,} ({100*recoverable/outlier_mask.sum():.1f}%)")

    return dists_outliers, dists_assigned


def citation_profile(df, cited_col="cited_by_count"):
    """Citation distribution of outlier vs. assigned papers.

    Plots the (clipped) distributions and returns a describe() table.
    """
    df = df.copy()
    df["_is_outlier"] = df["topic"] == -1

    cite_stats = df.groupby("_is_outlier")[cited_col].describe()
    cite_stats.index = ["Assigned", "Outlier"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, grp in df.groupby("_is_outlier"):
        vals = grp[cited_col].clip(upper=grp[cited_col].quantile(0.99))
        ax.hist(vals, bins=60, alpha=0.5, density=True, label="Outlier" if label else "Assigned")
    ax.set_xlabel("Cited-by count (clipped at 99th pctl)")
    ax.set_ylabel("Density")
    ax.set_title("Papers — citation distribution")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return cite_stats


def concept_overlap(df, n=20):
    """Compare the top OpenAlex concepts of outlier vs. assigned papers."""
    def top_concepts(mask):
        concepts = df.loc[mask, "concepts"].dropna().str.split(";").explode().str.strip()
        return concepts.value_counts().head(n)

    out_concepts = top_concepts(df["topic"] == -1)
    asg_concepts = top_concepts(df["topic"] != -1)

    compare = pd.DataFrame({
        "rank_outliers": range(1, len(out_concepts) + 1),
        "concept": out_concepts.index,
        "count_outliers": out_concepts.values,
    })
    compare_asg = pd.DataFrame({
        "concept": asg_concepts.index,
        "rank_assigned": range(1, len(asg_concepts) + 1),
        "count_assigned": asg_concepts.values,
    })
    return compare.merge(compare_asg, on="concept", how="left")


def sample_outlier_docs(df, title_col, abstract_col, year_col, name, k=10, seed=42):
    """Print a random sample of outlier titles + abstract snippets."""
    print(f"── {k} random outlier {name} {'─' * max(0, 40 - len(name))}")
    sample = df.loc[df["topic"] == -1]
    if sample.empty:
        print("  No outliers.")
        return
    sample = sample.sample(min(k, len(sample)), random_state=seed)
    for _, r in sample.iterrows():
        abstract = str(r.get(abstract_col, ""))[:200]
        print(f"  [{r[year_col]:.0f}]  {r[title_col]}")
        print(f"         {abstract}...")
        print()


def sample_outlier_table(df, cols, k=20, seed=42, truncate_col=None, truncate_len=300):
    """Return a random sample of outlier rows as a table (or None if empty)."""
    outliers = df.loc[df["topic"] == -1, cols]
    if outliers.empty:
        return None
    sample = outliers.sample(min(k, len(outliers)), random_state=seed).reset_index(drop=True)
    if truncate_col is not None:
        sample[truncate_col] = sample[truncate_col].fillna("").str[:truncate_len]
    return sample


def save_orphans(df, cols, out_file, reports_dir=REPORTS_DIR):
    """Write the outlier documents to a TSV under ``reports_dir``."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    orphans = df.loc[df["topic"] == -1, cols].copy()
    orphans.to_csv(reports_dir / out_file, sep="\t", index=False)
    return orphans


def build_summary(df, emb, topic_col_vals, name, text_col):
    """Compute the one-row outlier summary used in the final summary table."""
    topics_arr = np.array(topic_col_vals)
    is_out = topics_arr == -1
    n_total = len(df)
    n_out = is_out.sum()
    n_asg = n_total - n_out

    if n_out == 0:
        return {
            "dataset": name, "total": n_total, "assigned": n_asg,
            "outliers": 0, "outlier_rate": "0.0%",
            "outliers_≤20_words": "0", "near_centroid": "0", "far_from_clusters": "0",
        }

    # Text length
    wc = df[text_col].fillna("").str.split().str.len()
    short_out = (wc[is_out] <= 20).sum()

    # Distance to nearest centroid
    unique_topics = np.unique(topics_arr[~is_out])
    centroids = np.vstack([emb[topics_arr == t].mean(axis=0) for t in unique_topics])
    d_out = cdist(emb[is_out], centroids, metric="cosine").min(axis=1)
    d_asg = cdist(emb[~is_out], centroids, metric="cosine").min(axis=1)
    median_asg = np.median(d_asg)
    near_centroid = (d_out <= median_asg).sum()

    return {
        "dataset": name,
        "total": n_total,
        "assigned": n_asg,
        "outliers": n_out,
        "outlier_rate": f"{100*n_out/n_total:.1f}%",
        "outliers_≤20_words": f"{short_out:,} ({100*short_out/n_out:.1f}%)",
        "near_centroid": f"{near_centroid:,} ({100*near_centroid/n_out:.1f}%)",
        "far_from_clusters": f"{n_out - near_centroid:,} ({100*(n_out - near_centroid)/n_out:.1f}%)",
    }
