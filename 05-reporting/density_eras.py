"""R2 — temporal stability of the spatial division of labour.

The headline density-ratio map pools all years. A reviewer may ask whether the
red (teams-dense) / blue (papers-dense) division is merely an artefact of the
recent volume bulge. We answer by recomputing the *same* self-normalised
``log2(teams_density / papers_density)`` map separately for two eras of the iGEM
period — 2004-2014 and 2015-2025 — on one shared grid, and quantifying how
similar the two maps are (grid-cell correlation and per-topic zone agreement).

Outputs: assets/reports/density_ratio_eras.png
         (correlation and zone-agreement statistics are printed)
"""
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr
from scipy.ndimage import gaussian_filter

ASSETS = Path(__file__).resolve().parents[1] / "assets"
REPORTS = ASSETS / "reports"
MODELS = ASSETS / "topic_models"
ERAS = [("2004–2014", 2004, 2014), ("2015–2025", 2015, 2025)]
GRID = 300
BW = 0.15


def load_frames():
    pc = pd.read_csv(REPORTS / "joint_umap_papers_xy.tsv", sep="\t")
    tc = pd.read_csv(REPORTS / "joint_umap_teams_xy.tsv", sep="\t")
    pdt = pd.read_csv(MODELS / "papers_doc_topics.txt", sep="\t")
    praw = pd.read_csv(ASSETS / "synbio_openalex.txt", sep="\t", usecols=["id", "publication_year"])
    praw["year"] = pd.to_numeric(praw["publication_year"], errors="coerce")
    traw = pd.read_csv(ASSETS / "igem.txt", sep="\t", usecols=["UT", "Year_y"])
    traw["year"] = pd.to_numeric(traw["Year_y"], errors="coerce")
    dp = pc.merge(pdt, on="id").merge(praw[["id", "year"]], on="id")
    dt = tc.merge(traw[["UT", "year"]], on="UT")
    return dp, dt


def shared_limits(dp, dt, pad=1.5):
    xs = np.concatenate([dp["x"].to_numpy(), dt["x"].to_numpy()])
    ys = np.concatenate([dp["y"].to_numpy(), dt["y"].to_numpy()])
    return (xs.min() - pad, xs.max() + pad), (ys.min() - pad, ys.max() + pad)


def grid_pts(xlim, ylim, n=GRID):
    xv = np.linspace(*xlim, n); yv = np.linspace(*ylim, n)
    xx, yy = np.meshgrid(xv, yv)
    return xv, yv, xx, yy, np.vstack([xx.ravel(), yy.ravel()])


def era_ratio(dp, dt, lo, hi, xx, pts):
    pe = dp[(dp["year"] >= lo) & (dp["year"] <= hi)]
    te = dt[(dt["year"] >= lo) & (dt["year"] <= hi)]
    p = gaussian_kde(pe[["x", "y"]].to_numpy().T, bw_method=BW)(pts)
    q = gaussian_kde(te[["x", "y"]].to_numpy().T, bw_method=BW)(pts)
    p /= p.sum(); q /= q.sum()
    eps = 1e-12
    ratio = np.log2((q + eps) / (p + eps)).reshape(xx.shape)
    combined = (p + q).reshape(xx.shape)
    alpha = gaussian_filter(np.clip(combined / np.percentile(combined, 95), 0, 1), sigma=3)
    return ratio, combined, alpha, len(pe), len(te)


def main():
    dp, dt = load_frames()
    xlim, ylim = shared_limits(dp, dt)
    xv, yv, xx, yy, pts = grid_pts(xlim, ylim)

    res = []
    for label, lo, hi in ERAS:
        ratio, comb, alpha, np_, nt_ = era_ratio(dp, dt, lo, hi, xx, pts)
        res.append(dict(label=label, lo=lo, hi=hi, ratio=ratio, comb=comb,
                        alpha=alpha, n_papers=np_, n_teams=nt_))
        print(f"{label}: {np_:,} papers, {nt_:,} teams")

    a, b = res
    # grid-cell correlation, weighted to cells with real combined density
    w = (a["comb"] + b["comb"]).ravel()
    mask = w > np.percentile(w, 60)            # ignore near-empty background
    r_all, _ = pearsonr(a["ratio"].ravel(), b["ratio"].ravel())
    r_dense, _ = pearsonr(a["ratio"].ravel()[mask], b["ratio"].ravel()[mask])
    print(f"\nGrid-cell correlation of the two era maps: r_all = {r_all:.3f}, "
          f"r_dense = {r_dense:.3f}")

    # per-topic zone agreement (paper-topic centroids sampled in each era)
    from scipy.interpolate import RegularGridInterpolator
    def interp(ratio):
        return RegularGridInterpolator((yv, xv), ratio, method="linear",
                                       bounds_error=False, fill_value=np.nan)
    dpv = dp[dp["topic"] >= 0]
    ctr = dpv.groupby("topic").agg(cx=("x", "median"), cy=("y", "median")).reset_index()
    za = interp(a["ratio"])(np.column_stack([ctr["cy"], ctr["cx"]]))
    zb = interp(b["ratio"])(np.column_stack([ctr["cy"], ctr["cx"]]))
    def zone(v):
        return np.where(np.isnan(v), "na", np.where(v > 1, "teams", np.where(v < -1, "papers", "overlap")))
    za, zb = zone(za), zone(zb)
    ok = (za != "na") & (zb != "na")
    agree = (za[ok] == zb[ok]).mean()
    print(f"Per-topic coverage-zone agreement across eras: {100*agree:.0f}% "
          f"of {ok.sum()} paper topics")

    # figure: two panels, shared colour scale
    vmax = max(np.percentile(np.abs(a["ratio"]), 98), np.percentile(np.abs(b["ratio"]), 98))
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    cmap = plt.cm.RdBu_r
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), dpi=150)
    for ax, e in zip(axes, res):
        rgba = cmap(norm(np.clip(e["ratio"], -vmax, vmax)))
        rgba[..., 3] = e["alpha"]
        ax.imshow(rgba, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin="lower",
                  aspect="equal", interpolation="bilinear")
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.axis("off")
        ax.set_title(f'{e["label"]}  (papers={e["n_papers"]:,}, teams={e["n_teams"]:,})', fontsize=12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.045, pad=0.04)
    cbar.set_label("log2(teams density / papers density)")
    fig.suptitle(f"Spatial division of labour is stable across eras "
                 f"(grid r = {r_dense:.2f}, zone agreement = {100*agree:.0f}%)", fontsize=13)
    fig.savefig(REPORTS / "density_ratio_eras.png", dpi=180, bbox_inches="tight")
    print(f"\nSaved → {REPORTS/'density_ratio_eras.png'}")


if __name__ == "__main__":
    main()
