"""Build a three-level topic hierarchy (low → mid → high) from a BERTopic
merge tree, and write the per-corpus report tables.

The low-level building blocks (``build_hierarchy_maps`` … ``build_summary``) are
unchanged from the original ``hierarchy_utils.py``; the ``load_hierarchy_inputs``
/ ``select_hierarchy_levels`` / ``write_hierarchy_reports`` wrappers consolidate
the boilerplate so the teams and papers notebooks differ only by a CONFIG block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.metrics import silhouette_score

from .paths import (
    ASSETS_DIR, EMBEDDINGS_DIR, HIGH_K_MAX, HIGH_K_MIN, MODELS_DIR, REPORTS_DIR,
)


# ── Hierarchy construction ────────────────────────────────────────────────────
def build_hierarchy_maps(
    model: BERTopic,
    corpus_texts: list[str],
    max_k: int | None = None,
    use_ctfidf: bool = True,
) -> tuple[pd.DataFrame, dict[int, dict[int, int]], list[int]]:
    """Build the BERTopic merge tree and extract cluster maps for every k.

    Returns ``(hier_df, maps_by_k, low_topics)`` where ``maps_by_k`` is
    ``{k: {low_topic: group_id}}`` for every tracked k and ``low_topics`` are the
    sorted non-outlier low-level topic IDs.
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


# ── Topic embedding extraction ────────────────────────────────────────────────
def get_topic_embeddings(model: BERTopic, low_topics: list[int]) -> tuple[list[int], np.ndarray]:
    """Extract and align topic embeddings for non-outlier topics.

    BERTopic may or may not include an outlier row (-1) in ``topic_embeddings_``;
    this handles all known layouts. Returns ``(embed_topics, X)``.
    """
    if model.topic_embeddings_ is None:
        raise ValueError("Model has no topic_embeddings_; cannot score hierarchy.")

    model_topics_sorted = sorted(int(t) for t in model.get_topics().keys())
    n_emb = model.topic_embeddings_.shape[0]

    if len(model_topics_sorted) == n_emb:
        topic_to_emb = {tid: model.topic_embeddings_[idx] for idx, tid in enumerate(model_topics_sorted)}
    elif len(model_topics_sorted) - 1 == n_emb and model_topics_sorted[0] == -1:
        topic_to_emb = {tid: model.topic_embeddings_[idx] for idx, tid in enumerate(model_topics_sorted[1:])}
    elif len(model_topics_sorted) + 1 == n_emb:
        topic_to_emb = {
            tid: model.topic_embeddings_[idx + 1]
            for idx, tid in enumerate(model_topics_sorted) if tid >= 0
        }
    else:
        raise ValueError(
            f"Cannot align topic IDs ({len(model_topics_sorted)}) with "
            f"topic embeddings ({n_emb} rows)"
        )

    embed_topics = [t for t in low_topics if t in topic_to_emb]
    X = np.vstack([topic_to_emb[t] for t in embed_topics])
    return embed_topics, X


# ── k selection via silhouette scoring ────────────────────────────────────────
def select_best_k(
    embed_topics: list[int],
    X: np.ndarray,
    maps_by_k: dict[int, dict[int, int]],
    k_min: int,
    k_max: int,
    label: str = "k",
) -> tuple[int, float, list[tuple[int, float]]]:
    """Score candidate k by silhouette over [k_min, k_max]; return the best."""
    print(f"  Scoring {label}-level k candidates in [{k_min}, {k_max}] …")

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
        raise ValueError(f"No valid {label}-level k found in [{k_min}, {k_max}]")

    best_k, best_score = max(scores, key=lambda x: x[1])
    print(f"  ✓ Selected {label}-level k = {best_k} (silhouette = {best_score:.4f})")
    return best_k, best_score, scores


def auto_mid_k_range(n_low_topics: int, high_k_max: int) -> tuple[int, int]:
    """Derive the mid-level k range: [high_k_max + 1, n_low_topics // 3]."""
    k_min = high_k_max + 1
    k_max = n_low_topics // 3
    if k_max < k_min:
        raise ValueError(
            f"Too few low-level topics ({n_low_topics}) to derive a mid range: "
            f"k_min={k_min}, k_max={k_max}"
        )
    return k_min, k_max


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
def build_doc_map(doc_topics: pd.DataFrame, hierarchy: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
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


def build_name_map(topic_names: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
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

    result = pd.concat(parts, ignore_index=True).sort_values(["level", "group_id"]).reset_index(drop=True)
    print(f"  Summary: {len(result)} rows (mid + high)")
    return result


# ── High-level orchestration (consolidates the per-corpus boilerplate) ────────
def load_hierarchy_inputs(prefix, id_col, year_col, raw_filename, rename_id_from=None,
                          models_dir=MODELS_DIR, embeddings_dir=EMBEDDINGS_DIR, assets_dir=ASSETS_DIR):
    """Load the model + datasets for one corpus with normalised key columns.

    Returns ``(model, doc_topics, topic_names, corpus, raw)`` where ``doc_topics``
    and ``topic_names`` use ``low`` as the topic column and ``id_col`` as the
    document id (``rename_id_from`` renames the source id column when needed).
    """
    model = BERTopic.load(str(models_dir / f"{prefix}_topic_model"))
    doc_topics = pd.read_csv(models_dir / f"{prefix}_doc_topics.txt", sep="\t")
    topic_names = pd.read_csv(models_dir / f"{prefix}_topic_names.txt", sep="\t")
    corpus = pd.read_csv(embeddings_dir / f"{prefix}_corpus.txt", sep="\t")
    raw = pd.read_csv(assets_dir / raw_filename, sep="\t")

    doc_rename = {"topic": "low"}
    if rename_id_from:
        doc_rename[rename_id_from] = id_col
        corpus = corpus.rename(columns={rename_id_from: id_col})
        raw = raw.rename(columns={rename_id_from: id_col})
    doc_topics = doc_topics.rename(columns=doc_rename)
    topic_names = topic_names.rename(columns={"topic": "low"})

    doc_topics["low"] = doc_topics["low"].astype(int)
    topic_names["low"] = topic_names["low"].astype(int)
    raw[year_col] = pd.to_numeric(raw[year_col], errors="coerce")

    assert {id_col, "low"}.issubset(doc_topics.columns)
    assert {"low", "global_name"}.issubset(topic_names.columns)
    assert {id_col, "text"}.issubset(corpus.columns)
    assert {id_col, year_col}.issubset(raw.columns)
    return model, doc_topics, topic_names, corpus, raw


def select_hierarchy_levels(model, corpus_texts, high_k_min=HIGH_K_MIN, high_k_max=HIGH_K_MAX):
    """Build the merge tree and auto-select the high and mid levels.

    Returns ``(hierarchy_map, sel)`` where ``hierarchy_map`` is the (low, mid,
    high) table and ``sel`` carries the chosen k values, silhouette scores, and
    the mid search range (for display / validation).
    """
    hier_df, maps_by_k, low_topics = build_hierarchy_maps(model, corpus_texts)
    embed_topics, X = get_topic_embeddings(model, low_topics)

    high_k, high_score, high_scores = select_best_k(
        embed_topics, X, maps_by_k, high_k_min, high_k_max, label="high")
    mid_min, mid_max = auto_mid_k_range(len(low_topics), high_k_max)
    mid_k, mid_score, mid_scores = select_best_k(
        embed_topics, X, maps_by_k, mid_min, mid_max, label="mid")

    hierarchy_map = build_topic_hierarchy_df(low_topics, maps_by_k[mid_k], maps_by_k[high_k])
    sel = {
        "low_topics": low_topics,
        "high_k": high_k, "high_score": high_score, "high_scores": high_scores,
        "mid_k": mid_k, "mid_score": mid_score, "mid_scores": mid_scores,
        "mid_min": mid_min, "mid_max": mid_max,
    }
    return hierarchy_map, sel


def write_hierarchy_reports(doc_topics, topic_names, raw, hierarchy_map, id_col, year_col,
                            prefix, reports_dir=REPORTS_DIR):
    """Build and persist the three hierarchy report tables for one corpus.

    Returns ``(doc_map, name_map, summary)``. Files written to ``reports_dir``:
    ``<prefix>_topic_hierarchy_map.tsv``, ``<prefix>_topic_name_hierarchy.tsv``,
    ``<prefix>_topic_hierarchy_summary.tsv``.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    doc_map = build_doc_map(doc_topics, hierarchy_map, id_col=id_col)
    doc_map.to_csv(reports_dir / f"{prefix}_topic_hierarchy_map.tsv", sep="\t", index=False)

    name_map = build_name_map(topic_names, hierarchy_map)
    name_map.to_csv(reports_dir / f"{prefix}_topic_name_hierarchy.tsv", sep="\t", index=False)

    summary = build_summary(doc_map, raw, id_col=id_col, year_col=year_col)
    summary.to_csv(reports_dir / f"{prefix}_topic_hierarchy_summary.tsv", sep="\t", index=False)
    return doc_map, name_map, summary
