"""
Shared utilities for building a three-level topic hierarchy (low → mid → high)
from a BERTopic model's merge tree.

Used by:
  - get_papers_topic_hierarchy.ipynb
  - get_teams_topic_hierarchy.ipynb
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.metrics import silhouette_score


# ── Hierarchy construction ────────────────────────────────────────────────────

def build_hierarchy_maps(
    model: BERTopic,
    corpus_texts: list[str],
    max_k: int | None = None,
    use_ctfidf: bool = True,
) -> tuple[pd.DataFrame, dict[int, dict[int, int]], list[int]]:
    """Build the BERTopic merge tree and extract cluster maps for every k.

    Parameters
    ----------
    model : BERTopic
        A fitted BERTopic model (must have .topics_ populated).
    corpus_texts : list[str]
        Training-aligned documents (same order as model.topics_).
    max_k : int or None
        Largest k to track.  Defaults to the number of non-outlier topics.
    use_ctfidf : bool
        If True (default), use the model's pre-computed c-TF-IDF matrix for
        merging — faster and avoids an internal BERTopic indexing bug that
        can occur with ``use_ctfidf=False`` on some models.

    Returns
    -------
    hier_df : pd.DataFrame
        Raw hierarchical-topics table from BERTopic.
    maps_by_k : dict[int, dict[int, int]]
        ``{k: {low_topic: group_id}}`` for every tracked k.
    low_topics : list[int]
        Sorted list of non-outlier low-level topic IDs.
    """
    print("  Building BERTopic hierarchical merge tree …")
    if len(corpus_texts) != len(model.topics_):
        raise ValueError(
            f"corpus length ({len(corpus_texts)}) ≠ model.topics_ length "
            f"({len(model.topics_)}). hierarchical_topics requires "
            "training-aligned documents."
        )

    hier_df = model.hierarchical_topics(corpus_texts, use_ctfidf=use_ctfidf)

    # Auto-detect column names across BERTopic versions
    col_parent = "Parent_ID" if "Parent_ID" in hier_df.columns else "Parent Topic"
    col_left = "Child_Left_ID" if "Child_Left_ID" in hier_df.columns else "Child Left Topic"
    col_right = "Child_Right_ID" if "Child_Right_ID" in hier_df.columns else "Child Right Topic"
    col_distance = "Distance"
    if col_distance not in hier_df.columns:
        raise ValueError("Could not find 'Distance' column in hierarchical topics dataframe")

    # Non-outlier topics from the model
    low_topics = sorted(int(t) for t in set(model.topics_) if t >= 0)
    if max_k is None:
        max_k = len(low_topics)

    print(f"  Non-outlier low topics: {len(low_topics)}")
    print(f"  Tracking cluster maps for k = 1 … {max_k}")

    merges = (
        hier_df[[col_parent, col_left, col_right, col_distance]]
        .copy()
        .sort_values(col_distance)
    )

    node_members: dict[int, set[int]] = {t: {t} for t in low_topics}
    active_nodes: set[int] = set(low_topics)
    maps_by_k: dict[int, dict[int, int]] = {}
    tracked_ks = set(range(1, max_k + 1))

    for _, row in merges.iterrows():
        left = int(row[col_left])
        right = int(row[col_right])
        parent = int(row[col_parent])

        if left not in node_members or right not in node_members:
            continue
        if left not in active_nodes or right not in active_nodes:
            continue

        node_members[parent] = node_members[left] | node_members[right]
        active_nodes.discard(left)
        active_nodes.discard(right)
        active_nodes.add(parent)

        k_now = len(active_nodes)
        if k_now in tracked_ks and k_now not in maps_by_k:
            clusters = sorted(
                [sorted(node_members[n]) for n in active_nodes],
                key=lambda c: (min(c), len(c)),
            )
            topic_to_group = {}
            for gid, members in enumerate(clusters):
                for topic in members:
                    topic_to_group[int(topic)] = int(gid)
            maps_by_k[k_now] = topic_to_group

    print(f"  Captured {len(maps_by_k)} distinct k-level snapshots")
    return hier_df, maps_by_k, low_topics


# ── High-level k selection via silhouette scoring ─────────────────────────────

def select_best_high_k(
    model: BERTopic,
    low_topics: list[int],
    maps_by_k: dict[int, dict[int, int]],
    k_min: int,
    k_max: int,
) -> tuple[int, float, list[tuple[int, float]]]:
    """Score candidate high-level k values and return the best.

    Parameters
    ----------
    model : BERTopic
        Must have .topic_embeddings_ populated.
    low_topics : list[int]
        Non-outlier topic IDs.
    maps_by_k : dict
        Output of :func:`build_hierarchy_maps`.
    k_min, k_max : int
        Range of k values to evaluate (inclusive).

    Returns
    -------
    best_k : int
    best_score : float
    all_scores : list[(k, silhouette)]
    """
    print(f"  Scoring high-level k candidates in [{k_min}, {k_max}] …")

    if model.topic_embeddings_ is None:
        raise ValueError("Model has no topic_embeddings_; cannot score hierarchy.")

    # Map topic IDs to their embedding rows.  BERTopic stores embeddings in
    # the order returned by get_topics() (which may or may not include -1).
    # We handle both cases by pairing the sorted IDs with sequential rows.
    model_topics_sorted = sorted(int(t) for t in model.get_topics().keys())
    n_emb = model.topic_embeddings_.shape[0]

    # If the embeddings include an extra row for the outlier topic, or if the
    # topic list includes -1 but embeddings don't, align by the shorter list.
    if len(model_topics_sorted) == n_emb:
        topic_to_emb = {
            tid: model.topic_embeddings_[idx]
            for idx, tid in enumerate(model_topics_sorted)
        }
    elif len(model_topics_sorted) - 1 == n_emb and model_topics_sorted[0] == -1:
        # Embeddings don't include outlier topic — skip -1
        topic_to_emb = {
            tid: model.topic_embeddings_[idx]
            for idx, tid in enumerate(model_topics_sorted[1:])
        }
    elif len(model_topics_sorted) + 1 == n_emb:
        # Embeddings include an extra leading row for the outlier
        topic_to_emb = {
            tid: model.topic_embeddings_[idx + 1]
            for idx, tid in enumerate(model_topics_sorted)
            if tid >= 0
        }
    else:
        raise ValueError(
            f"Cannot align topic IDs ({len(model_topics_sorted)}) with "
            f"topic embeddings ({n_emb} rows)"
        )

    embed_topics = [t for t in low_topics if t in topic_to_emb]
    X = np.vstack([topic_to_emb[t] for t in embed_topics])

    scores: list[tuple[int, float]] = []
    for k in range(k_min, k_max + 1):
        if k not in maps_by_k:
            continue
        labels = [maps_by_k[k][t] for t in embed_topics]
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(X, labels, metric="cosine")
        scores.append((k, s))
        print(f"    k={k:>3d}  silhouette={s:.4f}")

    if not scores:
        raise ValueError(
            f"No valid high-level k found in [{k_min}, {k_max}]"
        )

    best_k, best_score = max(scores, key=lambda x: x[1])
    print(f"  ✓ Selected high-level k = {best_k} (silhouette = {best_score:.4f})")
    return best_k, best_score, scores


# ── Build hierarchy DataFrame ─────────────────────────────────────────────────

def build_topic_hierarchy_df(
    low_topics: list[int],
    low_to_mid: dict[int, int],
    low_to_high: dict[int, int],
) -> pd.DataFrame:
    """Create the (low, mid, high) mapping table for non-outlier topics."""
    df = pd.DataFrame({"low": low_topics})
    df["mid"] = df["low"].map(low_to_mid).astype(int)
    df["high"] = df["low"].map(low_to_high).astype(int)
    print(f"  Hierarchy table: {len(df)} low → {df['mid'].nunique()} mid → {df['high'].nunique()} high")
    return df


# ── Report builders ───────────────────────────────────────────────────────────

def build_doc_map(
    doc_topics: pd.DataFrame,
    hierarchy: pd.DataFrame,
    id_col: str = "ID",
) -> pd.DataFrame:
    """Merge document-level topic assignments with the hierarchy.

    Outlier documents (low = -1) get mid = high = -1.
    """
    out = doc_topics.merge(hierarchy, on="low", how="left")
    out["mid"] = out["mid"].fillna(-1).astype(int)
    out["high"] = out["high"].fillna(-1).astype(int)
    result = out[[id_col, "low", "mid", "high"]].copy()
    n_outlier = (result["low"] == -1).sum()
    print(f"  Document map: {len(result):,} rows ({n_outlier:,} outliers)")
    return result


def build_name_map(
    topic_names: pd.DataFrame,
    hierarchy: pd.DataFrame,
) -> pd.DataFrame:
    """Merge topic names with the hierarchy."""
    out = topic_names.merge(hierarchy, on="low", how="left")
    out["mid"] = out["mid"].fillna(-1).astype(int)
    out["high"] = out["high"].fillna(-1).astype(int)
    result = out[["global_name", "low", "mid", "high"]].sort_values("low")
    print(f"  Name map: {len(result)} topics")
    return result


def build_summary(
    doc_map: pd.DataFrame,
    metadata: pd.DataFrame,
    id_col: str = "ID",
    year_col: str = "publication_year",
) -> pd.DataFrame:
    """Compute per-group counts and year statistics for mid and high levels."""
    merged = doc_map.merge(metadata[[id_col, year_col]], on=id_col, how="left")

    parts = []
    for level in ("mid", "high"):
        agg = (
            merged[merged[level] >= 0]
            .groupby(level, as_index=False)
            .agg(
                total_count=(id_col, "count"),
                avg_publication_year=(year_col, "mean"),
                median_publication_year=(year_col, "median"),
            )
            .rename(columns={level: "group_id"})
        )
        agg.insert(0, "level", level)
        parts.append(agg)

    result = pd.concat(parts, ignore_index=True).sort_values(
        ["level", "group_id"]
    ).reset_index(drop=True)
    print(f"  Summary: {len(result)} rows (mid + high)")
    return result
