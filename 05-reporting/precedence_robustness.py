"""R1 — robustness of the temporal-precedence result.

Re-implements the density-ratio + precedence neighbourhood logic of
``aux/density.py`` and ``aux/precedence.py`` from the persisted joint coordinates
(no UMAP/BERTopic re-run needed), then checks two things a reviewer would ask:

  (a) Sign stability across the year statistic used (Q1 / median / mean): do the
      lead/lag directions agree regardless of which quantile we read?
  (b) Split stability across the neighbourhood parameters: does the iGEM-first vs
      papers-first count hold as the radius multiplier and the minimum-support
      threshold vary?

It first reproduces the shipped ``overlap_precedence_full.tsv`` at the default
parameters (radius_mult=1.5, min_nearby=3) as a correctness check.

Outputs: assets/reports/precedence_robustness.tsv
         assets/reports/precedence_robustness_sensitivity.tsv
         assets/reports/precedence_robustness.png
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

ASSETS = Path(__file__).resolve().parents[1] / "assets"
REPORTS = ASSETS / "reports"
MODELS = ASSETS / "topic_models"
_STATS = ("q1", "median", "mean")


# ── data ──────────────────────────────────────────────────────────────────────
def load_frames():
    pc = pd.read_csv(REPORTS / "joint_umap_papers_xy.tsv", sep="\t")
    tc = pd.read_csv(REPORTS / "joint_umap_teams_xy.tsv", sep="\t")
    pdt = pd.read_csv(MODELS / "papers_doc_topics.txt", sep="\t")
    tdt = pd.read_csv(MODELS / "teams_doc_topics.txt", sep="\t")
    praw = pd.read_csv(ASSETS / "synbio_openalex.txt", sep="\t",
                       usecols=["id", "publication_year"])
    praw["year"] = pd.to_numeric(praw["publication_year"], errors="coerce")
    traw = pd.read_csv(ASSETS / "igem.txt", sep="\t", usecols=["UT", "Year_y"])
    traw["year"] = pd.to_numeric(traw["Year_y"], errors="coerce")

    dp = pc.merge(pdt, on="id").merge(praw[["id", "year"]], on="id")
    dt = tc.merge(tdt, on="UT").merge(traw[["UT", "year"]], on="UT")
    return dp, dt


def shared_limits(dp, dt, pad=1.5):
    xs = np.concatenate([dp["x"].to_numpy(), dt["x"].to_numpy()])
    ys = np.concatenate([dp["y"].to_numpy(), dt["y"].to_numpy()])
    return (xs.min() - pad, xs.max() + pad), (ys.min() - pad, ys.max() + pad)


def density_interp(dp, dt, xlim, ylim, grid_res=300, bw=0.15):
    """Faithful copy of aux/density.compute_density_ratio's interpolator."""
    kp = gaussian_kde(dp[["x", "y"]].to_numpy().T, bw_method=bw)
    kt = gaussian_kde(dt[["x", "y"]].to_numpy().T, bw_method=bw)
    xv = np.linspace(xlim[0], xlim[1], grid_res)
    yv = np.linspace(ylim[0], ylim[1], grid_res)
    xx, yy = np.meshgrid(xv, yv)
    pts = np.vstack([xx.ravel(), yy.ravel()])
    p = kp(pts); q = kt(pts)
    p /= p.sum(); q /= q.sum()
    eps = 1e-12
    ratio = np.log2((q + eps) / (p + eps)).reshape(xx.shape)
    vmax = np.percentile(np.abs(ratio), 98)
    ratio = np.clip(ratio, -vmax, vmax)
    return RegularGridInterpolator((yv, xv), ratio, method="linear",
                                   bounds_error=False, fill_value=np.nan)


def _q(s, k):
    s = pd.Series(s).dropna().astype(float)
    return {"q1": s.quantile(0.25), "median": s.median(), "mean": s.mean()}[k]


# ── precedence (parametrised) ─────────────────────────────────────────────────
def precedence(dp, dt, interp, names, radius_mult=1.5, min_nearby=3):
    dpv = dp[dp["topic"] >= 0]
    ctr = dpv.groupby("topic").agg(cx=("x", "median"), cy=("y", "median"),
                                   n=("x", "size")).reset_index()
    ctr["lr"] = interp(np.column_stack([ctr["cy"], ctr["cx"]]))
    ctr = ctr[ctr["lr"].abs() <= 1]                      # overlap zone only
    tree = cKDTree(dt[["x", "y"]].to_numpy())
    out = []
    for _, r in ctr.iterrows():
        pts = dpv.loc[dpv["topic"] == r["topic"], ["x", "y"]].to_numpy()
        rad = np.median(np.sqrt((pts[:, 0] - r["cx"])**2 + (pts[:, 1] - r["cy"])**2)) * radius_mult
        idx = tree.query_ball_point([r["cx"], r["cy"]], rad)
        ty = dt.iloc[idx]["year"].dropna()
        py = dpv.loc[dpv["topic"] == r["topic"], "year"].dropna()
        if len(ty) < min_nearby or len(py) == 0:
            continue
        rec = {"topic": int(r["topic"]), "global_name": names.get(int(r["topic"]), ""),
               "n_papers": int(r["n"]), "n_nearby_teams": len(idx)}
        for k in _STATS:
            rec[f"delta_{k}_years"] = round(_q(ty, k) - _q(py, k), 2)
        out.append(rec)
    return pd.DataFrame(out)


def main():
    dp, dt = load_frames()
    xlim, ylim = shared_limits(dp, dt)
    interp = density_interp(dp, dt, xlim, ylim)

    ref = pd.read_csv(REPORTS / "overlap_precedence_full.tsv", sep="\t")
    names = dict(zip(ref["topic"].astype(int), ref["global_name"]))

    base = precedence(dp, dt, interp, names, 1.5, 3)

    # correctness check vs shipped table
    m = base.merge(ref[["topic", "delta_q1_years"]], on="topic", suffixes=("", "_ref"))
    max_diff = (m["delta_q1_years"] - m["delta_q1_years_ref"]).abs().max()
    print(f"Reproduced {len(base)} overlap topics (shipped: {len(ref)}); "
          f"max |delta_q1 diff| vs shipped = {max_diff:.3f}\n")

    # (a) sign stability across quantiles
    def sgn(v):
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)
    for k in _STATS:
        base[f"sign_{k}"] = base[f"delta_{k}_years"].apply(sgn)
    nz = base[(base[[f"sign_{k}" for k in _STATS]] != 0).all(axis=1)]
    agree = (nz["sign_q1"] == nz["sign_median"]) & (nz["sign_q1"] == nz["sign_mean"])
    base["sign_agree_q1_med_mean"] = (
        (base["sign_q1"] == base["sign_median"]) & (base["sign_q1"] == base["sign_mean"]))
    print("(a) Sign stability across Q1 / median / mean:")
    print(f"    {agree.sum()}/{len(nz)} non-zero topics agree in sign across all three "
          f"({100*agree.mean():.0f}%).")
    for k in _STATS:
        neg = (base[f"delta_{k}_years"] < 0).sum()
        pos = (base[f"delta_{k}_years"] > 0).sum()
        print(f"    by delta_{k:6s}: {neg:2d} iGEM-first, {pos:2d} papers-first")

    base.drop(columns=[f"sign_{k}" for k in _STATS]).to_csv(
        REPORTS / "precedence_robustness.tsv", sep="\t", index=False)

    # (b) split stability across radius_mult x min_nearby
    print("\n(b) Split stability across neighbourhood parameters (direction by delta_q1):")
    print(f'    {"radius":>7s}{"min_n":>6s}{"tested":>7s}{"iGEM1st":>8s}{"pap1st":>7s}{"flips*":>7s}')
    base_sign = dict(zip(base["topic"], np.sign(base["delta_q1_years"])))
    sens = []
    for rm in (1.0, 1.5, 2.0):
        for mn in (3, 5, 10):
            d = precedence(dp, dt, interp, names, rm, mn)
            neg = int((d["delta_q1_years"] < 0).sum())
            pos = int((d["delta_q1_years"] > 0).sum())
            common = d[d["topic"].isin(base_sign)]
            flips = int(sum(np.sign(r.delta_q1_years) != base_sign[r.topic]
                            and base_sign[r.topic] != 0
                            for r in common.itertuples()))
            sens.append({"radius_mult": rm, "min_nearby": mn, "n_tested": len(d),
                         "n_igem_first": neg, "n_papers_first": pos, "n_sign_flips_vs_default": flips})
            print(f'    {rm:7.1f}{mn:6d}{len(d):7d}{neg:8d}{pos:7d}{flips:7d}')
    pd.DataFrame(sens).to_csv(REPORTS / "precedence_robustness_sensitivity.tsv", sep="\t", index=False)
    print("    * sign flips of delta_q1 vs the default (1.5, 3) run, among shared topics.")

    # figure: delta_q1 vs delta_mean, coloured by agreement
    fig, ax = plt.subplots(figsize=(7.5, 7), dpi=150)
    ok = base["sign_agree_q1_med_mean"]
    sup = np.minimum(base["n_papers"], base["n_nearby_teams"]).clip(lower=1)
    size = 20 + 120 * np.sqrt(sup / sup.max())
    ax.axhline(0, color="0.6", lw=1); ax.axvline(0, color="0.6", lw=1)
    ax.scatter(base.loc[ok, "delta_q1_years"], base.loc[ok, "delta_mean_years"],
               s=size[ok.values], c="#2a7a2a", alpha=0.8, edgecolor="white", linewidth=0.4,
               label="Q1/median/mean agree in sign")
    ax.scatter(base.loc[~ok, "delta_q1_years"], base.loc[~ok, "delta_mean_years"],
               s=size[~ok.values], c="#d4731c", alpha=0.85, edgecolor="white", linewidth=0.4,
               label="disagree (near zero)")
    lim = max(base["delta_q1_years"].abs().max(), base["delta_mean_years"].abs().max()) * 1.08
    ax.plot([-lim, lim], [-lim, lim], ls=":", color="0.5", lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Precedence gap, Q1  (teams − papers, years)")
    ax.set_ylabel("Precedence gap, mean  (teams − papers, years)")
    ax.set_title("Precedence is sign-stable across year statistics\n(marker size ∝ support)", fontsize=12)
    ax.text(-lim*0.96, -lim*0.9, "iGEM-first quadrant", color="#b2182b", fontsize=9)
    ax.text(lim*0.96, lim*0.92, "papers-first quadrant", color="#2166ac", fontsize=9, ha="right")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORTS / "precedence_robustness.png", dpi=180, bbox_inches="tight")
    print(f"\nSaved → {REPORTS/'precedence_robustness.tsv'}")
    print(f"Saved → {REPORTS/'precedence_robustness_sensitivity.tsv'}")
    print(f"Saved → {REPORTS/'precedence_robustness.png'}")


if __name__ == "__main__":
    main()
