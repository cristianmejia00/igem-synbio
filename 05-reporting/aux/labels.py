"""Label-placement strategies for the topic maps.

Three independent placers, each consuming a centroid frame with ``cx, cy,
n_docs, label`` columns:

- ``add_side_labels``    — labels in the left/right gutters with elbow connectors.
- ``add_overlay_labels`` — labels sitting on top of their clusters, force-relaxed
                           to reduce overlap, drawn with a white outline.
- ``add_density_labels`` — overlay variant for the density heatmap: per-label
                           colour/box style with extra vertical spreading.
"""
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe


# ── Side labels (gutter placement with connectors) ────────────────────────────
def _spread_targets(y_values, y_min, y_max, min_gap):
    """Spread y-positions monotonically so labels keep a minimum separation."""
    if len(y_values) == 0:
        return np.array([], dtype=float)
    ys = np.array(y_values, dtype=float)
    ys = np.clip(ys, y_min, y_max)
    for i in range(1, len(ys)):
        if ys[i - 1] - ys[i] < min_gap:
            ys[i] = ys[i - 1] - min_gap
    if ys[-1] < y_min:
        ys += (y_min - ys[-1])
        for i in range(len(ys) - 2, -1, -1):
            if ys[i] - ys[i + 1] < min_gap:
                ys[i] = ys[i + 1] + min_gap
    if ys[0] > y_max:
        ys -= (ys[0] - y_max)
    return np.clip(ys, y_min, y_max)


def _pick_labels_for_side(sub, y_min, y_max, min_gap, max_labels=None):
    """Select a manageable, vertically spread subset of labels for one side."""
    if sub.empty:
        return sub
    capacity = max(1, int(np.floor((y_max - y_min) / min_gap)) + 1)
    target_n = capacity if max_labels is None else min(capacity, max_labels)
    target_n = min(target_n, len(sub))
    if len(sub) <= target_n:
        return sub.sort_values("cy", ascending=False).reset_index(drop=True)

    work = sub.copy()
    bins = np.linspace(y_min, y_max, target_n + 1)
    work["y_bin"] = np.clip(np.digitize(work["cy"], bins) - 1, 0, target_n - 1)
    primary = (
        work.sort_values(["y_bin", "n_docs"], ascending=[True, False])
        .groupby("y_bin", as_index=False).head(1)
    )
    if len(primary) < target_n:
        used = set(primary.index.tolist())
        rest = work.loc[~work.index.isin(used)].sort_values("n_docs", ascending=False).head(target_n - len(primary))
        chosen = pd.concat([primary, rest], axis=0)
    else:
        chosen = primary
    return chosen.sort_values("cy", ascending=False).head(target_n).reset_index(drop=True)


def add_side_labels(ax, labels_df, xlim, ylim, min_gap_frac=0.024, text_pad_frac=0.008,
                    elbow_frac=0.004, fontsize=5.5, max_labels_per_side=None):
    """Place labels on plot sides with elbow connectors and anti-overlap spacing."""
    if labels_df.empty:
        return
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_span = x_max - x_min
    y_span = y_max - y_min

    x_left = x_min - text_pad_frac * x_span
    x_right = x_max + text_pad_frac * x_span
    x_elbow_left = x_min - elbow_frac * x_span
    x_elbow_right = x_max + elbow_frac * x_span
    min_gap = min_gap_frac * y_span

    center_x = 0.5 * (x_min + x_max)
    labels = labels_df.copy()
    labels["side"] = np.where(labels["cx"] <= center_x, "left", "right")

    for side in ("left", "right"):
        sub = labels[labels["side"] == side].copy()
        if sub.empty:
            continue
        sub = _pick_labels_for_side(sub, y_min=y_min, y_max=y_max, min_gap=min_gap, max_labels=max_labels_per_side)
        sub = sub.sort_values("cy", ascending=False).reset_index(drop=True)
        sub["target_y"] = _spread_targets(sub["cy"].to_numpy(), y_min, y_max, min_gap)

        n_sub = len(sub)
        for i, (_, row) in enumerate(sub.iterrows()):
            if side == "left":
                x_txt = x_left
                x_elb = x_elbow_left - 0.004 * x_span * (i - (n_sub - 1) / 2)
                ha = "right"
            else:
                x_txt = x_right
                x_elb = x_elbow_right + 0.004 * x_span * (i - (n_sub - 1) / 2)
                ha = "left"

            ax.plot([row["cx"], x_elb, x_txt], [row["cy"], row["target_y"], row["target_y"]],
                    color="0.35", linewidth=0.45, alpha=0.65, zorder=3, clip_on=False)
            ax.text(x_txt, row["target_y"], str(row["label"])[:34], fontsize=fontsize,
                    ha=ha, va="center", color="black",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.85, ec="none"),
                    zorder=4, clip_on=False)


# ── Overlay labels (on top of clusters, force-relaxed) ────────────────────────
def map_fontsizes(values, min_size=6.5, max_size=14.5):
    """Map cluster sizes to readable font sizes using a log scale."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.array([], dtype=float)
    v = np.clip(v, 1, None)
    lo, hi = np.log1p(v.min()), np.log1p(v.max())
    if hi <= lo:
        return np.full_like(v, (min_size + max_size) / 2.0, dtype=float)
    s = (np.log1p(v) - lo) / (hi - lo)
    return min_size + s * (max_size - min_size)


def _relax_label_positions(anchors, labels, font_sizes, xlim, ylim, n_iter=220):
    """Force-based layout that nudges labels apart while staying over clusters."""
    anchors = np.asarray(anchors, dtype=float)
    pos = anchors.copy()
    n = len(pos)
    if n == 0:
        return pos
    x_span = max(xlim[1] - xlim[0], 1e-9)
    y_span = max(ylim[1] - ylim[0], 1e-9)
    widths = np.array([
        max(0.018, (0.50 * fs * max(len(str(lbl)), 4) / 72.0) / 14.0) * x_span
        for lbl, fs in zip(labels, font_sizes)
    ])
    heights = np.array([max(0.020, (1.25 * fs / 72.0) / 10.0) * y_span for fs in font_sizes])

    for _ in range(n_iter):
        disp = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                overlap_x = (widths[i] + widths[j]) * 0.5 - abs(dx)
                overlap_y = (heights[i] + heights[j]) * 0.5 - abs(dy)
                if overlap_x > 0 and overlap_y > 0:
                    if abs(dy) < 1e-9:
                        dy = 1e-9 if i > j else -1e-9
                    if abs(dx) < 1e-9:
                        dx = 1e-9 if i < j else -1e-9
                    push_y = 0.20 * overlap_y * np.sign(dy)
                    push_x = 0.06 * overlap_x * np.sign(dx)
                    disp[i, 0] += push_x
                    disp[j, 0] -= push_x
                    disp[i, 1] += push_y
                    disp[j, 1] -= push_y
        disp += 0.04 * (anchors - pos)
        pos += disp
        pos[:, 0] = np.clip(pos[:, 0], xlim[0] + 0.01 * x_span, xlim[1] - 0.01 * x_span)
        pos[:, 1] = np.clip(pos[:, 1], ylim[0] + 0.01 * y_span, ylim[1] - 0.01 * y_span)
    return pos


def add_overlay_labels(ax, labels_df, xlim, ylim, size_col="n_docs",
                       min_size=6.3, max_size=10.2, max_chars=36, n_iter=220):
    """Place labels over their clusters with a white outline (no boxes)."""
    if labels_df.empty:
        return
    labels = labels_df["label"].astype(str).str.slice(0, max_chars).tolist()
    if size_col in labels_df.columns:
        font_sizes = map_fontsizes(labels_df[size_col].to_numpy(), min_size, max_size)
    else:
        font_sizes = np.full(len(labels), (min_size + max_size) / 2.0)
    anchors = labels_df[["cx", "cy"]].to_numpy(dtype=float)
    positions = _relax_label_positions(anchors, labels, font_sizes, xlim, ylim, n_iter=n_iter)
    for (x, y), label, fs in zip(positions, labels, font_sizes):
        txt = ax.text(x, y, label, fontsize=float(fs), color="black",
                      ha="center", va="center", zorder=6)
        txt.set_path_effects([pe.Stroke(linewidth=max(1.9, fs * 0.18), foreground="white"), pe.Normal()])


# ── Density-map labels (overlay variant with vertical spreading) ──────────────
def _relax_density_positions(labels_df, xlim, ylim, n_iter=270, y_min_gap_frac=0.01):
    """Force layout with an explicit vertical-spreading pass for crowded labels."""
    anchors = labels_df[["cx", "cy"]].to_numpy(dtype=float)
    pos = anchors.copy()
    if len(pos) == 0:
        return pos
    x_span = max(xlim[1] - xlim[0], 1e-9)
    y_span = max(ylim[1] - ylim[0], 1e-9)
    y_min_gap = y_min_gap_frac * y_span

    fs = labels_df["fontsize"].to_numpy(dtype=float)
    widths = np.array([
        max(0.026, 0.0058 * len(str(lbl)) + 0.0042 * f) * x_span / 12.0
        for lbl, f in zip(labels_df["label"].tolist(), fs)
    ])
    heights = np.array([max(0.021, 0.0037 * f) * y_span / 12.0 for f in fs])

    for _ in range(n_iter):
        disp = np.zeros_like(pos)
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                ox = 0.5 * (widths[i] + widths[j]) - abs(dx)
                oy = 0.5 * (heights[i] + heights[j]) - abs(dy)
                if ox > 0 and oy > 0:
                    sx = np.sign(dx) if abs(dx) > 1e-10 else (1 if i > j else -1)
                    sy = np.sign(dy) if abs(dy) > 1e-10 else (1 if i > j else -1)
                    disp[i, 0] += 0.06 * ox * sx
                    disp[j, 0] -= 0.06 * ox * sx
                    disp[i, 1] += 0.38 * oy * sy
                    disp[j, 1] -= 0.38 * oy * sy
        disp[:, 0] += 0.02 * (anchors[:, 0] - pos[:, 0])
        disp[:, 1] += 0.07 * (anchors[:, 1] - pos[:, 1])
        pos += disp

        order = np.argsort(pos[:, 1])
        for k in range(1, len(order)):
            min_y = pos[order[k - 1], 1] + y_min_gap
            if pos[order[k], 1] < min_y:
                pos[order[k], 1] = min_y
        y_low = ylim[0] + 0.01 * y_span
        y_high = ylim[1] - 0.01 * y_span
        if len(order) > 0 and pos[order[-1], 1] > y_high:
            pos[:, 1] -= (pos[order[-1], 1] - y_high)
            for k in range(len(order) - 2, -1, -1):
                max_y = pos[order[k + 1], 1] - y_min_gap
                if pos[order[k], 1] > max_y:
                    pos[order[k], 1] = max_y
        pos[:, 0] = np.clip(pos[:, 0], xlim[0] + 0.01 * x_span, xlim[1] - 0.01 * x_span)
        pos[:, 1] = np.clip(pos[:, 1], y_low, y_high)
    return pos


def add_density_labels(ax, labels_df, xlim, ylim, n_iter=270, y_min_gap_frac=0.01):
    """Place styled labels (per-row colour / box alpha) on the density heatmap."""
    if labels_df.empty:
        return
    positions = _relax_density_positions(labels_df, xlim, ylim, n_iter=n_iter, y_min_gap_frac=y_min_gap_frac)
    for (x, y), (_, row) in zip(positions, labels_df.iterrows()):
        ax.text(
            x, y, row["label"],
            fontsize=float(row.get("fontsize", 7.0)),
            fontstyle=row.get("fontstyle", "normal"),
            fontweight=row.get("fontweight", "normal"),
            ha="center", va="center", color=row.get("color", "black"),
            bbox=dict(boxstyle="round,pad=0.11", fc="white", alpha=float(row.get("box_alpha", 0.8)), ec="none"),
        )
