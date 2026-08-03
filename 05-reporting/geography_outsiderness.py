"""Country outsiderness trajectories: are teams still working where the literature is not?

The `geography_lead_orientation.png` scatter cannot simply be extended over time.
Two mechanical constraints block the obvious approaches:

1. `delta_q1_years = teams_q1_year - papers_q1_year` is *defined by* team timing,
   so an early team can only sit near topics that are iGEM-led. Pooled across all
   countries, mean orientation drifts -2.9 -> +0.1 with country labels destroyed.
2. The Location Quotient is compositional: the paper-share-weighted mean of LQ is
   pinned to 1.0 in every window. A global decline in outsiderness is therefore
   unobservable in LQ space -- only redistribution between countries.

This script measures outsiderness with a spatial field instead, so neither
constraint applies: team years never enter the field's construction and nothing
is forced to sum to a constant, so a country's line is free to fall.

    outsiderness(team) = 1 - P( literature density at a random paper
                               <  literature density at this team's location )

An outsiderness of 0.80 means the team sits in a region of the joint map sparser
than where 80% of the literature sits. Bounded [0, 1], scale-free, and referenced
to the literature's own distribution rather than to arbitrary KDE units.

Two literature fields are computed, and the choice matters:

*   ``contemp`` (PRIMARY) -- the KDE is built from papers published **up to and
    including the team's own year**. It asks the honest question: at the time the
    team worked, had the literature arrived there yet?
*   ``fixed`` (DIAGNOSTIC) -- the KDE pools all papers 2004-2025. This is the
    naive choice and it is **biased**: a team that pioneered a region the
    literature only reached years later is scored against that future literature
    and therefore looks like an insider. The bias inflates early-cohort centrality
    and manufactures a spurious upward trend in outsiderness. The two fields
    correlate at r ~= 0.90 cross-sectionally yet disagree completely on the time
    trend, which is exactly the signature of the artefact. `field_check` plots
    both; the fixed field is reported only to show why it must not be used.

A third score, ``ratio`` = log2(teams density / papers density) at the team's
location, reproduces the repo's established density-ratio construct
(`umap_density_ratio.png`) as a cross-check on the papers-only score.

Country attribution follows `geography_lq.py`: one count per team-country, so a
multi-country team contributes to each of its countries.

Inputs : assets/reports/joint_umap_{papers,teams}_xy.tsv, assets/igem.txt,
         assets/synbio_openalex.txt, assets/reports/geography_location_quotient.tsv
Outputs: assets/reports/geography_outsiderness_teams.tsv
         assets/reports/geography_outsiderness_trajectory.tsv
         assets/reports/geography_outsiderness_trajectory.png
         assets/reports/geography_outsiderness_field_check.png
"""
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde

from geography_lq import ISO3_TO_NAME

ASSETS = Path(__file__).resolve().parents[1] / "assets"
REPORTS = ASSETS / "reports"

BW = 0.15               # matches density_eras.py / the shipped density-ratio map
REF_SAMPLE = 3000       # papers subsampled to form the percentile reference
MIN_REF_PAPERS = 200    # a contemporaneous field thinner than this is not trusted
SEED = 0

HALF = 1                # rolling window half-width -> 3-year windows
MIN_TEAMS_GLOBAL = 50   # minimum teams in a window for the global mean to be drawn
MIN_TEAMS_WINDOW = 8    # minimum teams for a country to get a point in a window. def = 8
MIN_TEAMS_TOTAL = 40    # minimum teams for a country to appear at all. def = 40
MIN_WINDOWS = 8         # minimum window points for a country to be plotted

RED, BLUE = "#b2182b", "#2166ac"   # outsider-tilted / core-tilted, as in geography_lq.png
INK = "#1a1a1a"

# 19 lines cannot be told apart by hue, so the thin lines carry only the spread and
# just four are direct-labelled. They are the four largest team communities, not the
# steepest slopes: the extreme slopes (Brazil, India) belong to the smallest samples
# and would draw the eye to noise. Their uncertainty is visible in the right panel.
HIGHLIGHT = ["CHN", "USA", "CAN", "DEU"]


# ── data ──────────────────────────────────────────────────────────────────────
def load():
    pc = pd.read_csv(REPORTS / "joint_umap_papers_xy.tsv", sep="\t")
    praw = pd.read_csv(ASSETS / "synbio_openalex.txt", sep="\t",
                       usecols=["id", "publication_year"])
    praw["year"] = pd.to_numeric(praw["publication_year"], errors="coerce")
    dp = pc.merge(praw[["id", "year"]], on="id").dropna(subset=["year"])
    dp["year"] = dp["year"].astype(int)

    tc = pd.read_csv(REPORTS / "joint_umap_teams_xy.tsv", sep="\t")
    traw = pd.read_csv(ASSETS / "igem.txt", sep="\t", usecols=["UT", "Countries", "Year_y"])
    traw["year"] = pd.to_numeric(traw["Year_y"], errors="coerce")
    dt = tc.merge(traw[["UT", "Countries", "year"]], on="UT").dropna(subset=["year"])
    dt["year"] = dt["year"].astype(int)
    return dp.reset_index(drop=True), dt.reset_index(drop=True)


def explode_countries(dt):
    """One row per (team, country) — matches geography_lq.py's counting."""
    s = dt["Countries"].astype(str).str.replace(",", ";").str.split(";")
    out = dt.assign(iso3=s).explode("iso3")
    out["iso3"] = out["iso3"].str.strip()
    out = out[out["iso3"].ne("") & out["iso3"].ne("nan")]
    return out.reset_index(drop=True)


# ── scores ────────────────────────────────────────────────────────────────────
def _outsiderness(train_xy, ref_xy, eval_xy):
    """1 - (empirical CDF of the field, evaluated at eval points)."""
    kde = gaussian_kde(train_xy.T, bw_method=BW)
    ref = np.sort(kde(ref_xy.T))
    dens = kde(eval_xy.T)
    return 1.0 - np.searchsorted(ref, dens, side="right") / ref.size


def _ref_sample(xy, rng):
    if len(xy) <= REF_SAMPLE:
        return xy
    return xy[rng.choice(len(xy), REF_SAMPLE, replace=False)]


def score_teams(dp, dt, rng):
    pxy = dp[["x", "y"]].to_numpy()
    txy = dt[["x", "y"]].to_numpy()

    # fixed (all-years) field — DIAGNOSTIC ONLY
    fixed = _outsiderness(pxy, _ref_sample(pxy, rng), txy)

    # contemporaneous field — PRIMARY
    contemp = np.full(len(dt), np.nan)
    for y in sorted(dt["year"].unique()):
        sub = dp.loc[dp["year"] <= y, ["x", "y"]].to_numpy()
        if len(sub) < MIN_REF_PAPERS:
            print(f"  year {y}: only {len(sub)} papers ≤ {y} — contemporaneous score skipped")
            continue
        m = (dt["year"] == y).to_numpy()
        contemp[m] = _outsiderness(sub, _ref_sample(sub, rng), txy[m])

    # repo's established density-ratio construct, at the team's location
    kde_t = gaussian_kde(txy.T, bw_method=BW)
    kde_p = gaussian_kde(pxy.T, bw_method=BW)
    eps = 1e-300
    ratio = np.log2((kde_t(txy.T) + eps) / (kde_p(txy.T) + eps))

    return contemp, fixed, ratio


# ── statistics ────────────────────────────────────────────────────────────────
def ols_slope(x, y):
    """Slope per year, its standard error, and R^2."""
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    sxx = ((x - x.mean()) ** 2).sum()
    dof = n - 2
    se = np.sqrt((resid @ resid) / dof / sxx) if dof > 0 and sxx > 0 else np.nan
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return b, se, r2


def rolling_mean(df, years, col, half=HALF):
    return pd.DataFrame([
        dict(year=y, value=df.loc[df["year"].between(y - half, y + half), col].mean(),
             n=int(df["year"].between(y - half, y + half).sum()))
        for y in years])


def trajectories(ex, years, col="contemp"):
    """Country x rolling-window mean outsiderness."""
    rows = []
    for y in years:
        w = ex[ex["year"].between(y - HALF, y + HALF)]
        if w.empty:
            continue
        g_mean = w[col].mean()
        for iso, g in w.groupby("iso3"):
            if len(g) < MIN_TEAMS_WINDOW:
                continue
            rows.append(dict(
                year=y, iso3=iso, country=ISO3_TO_NAME.get(iso, iso), n=len(g),
                outsiderness=g[col].mean(),
                sem=g[col].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan,
                rel_outsiderness=g[col].mean() - g_mean, global_mean=g_mean))
    return pd.DataFrame(rows)


def country_slopes(ex, isos, col="contemp"):
    """Team-level OLS of outsiderness on year, raw and net of the year's global mean."""
    ex = ex.assign(rel=ex[col] - ex.groupby("year")[col].transform("mean"))
    rows = []
    for iso in isos:
        g = ex[ex["iso3"] == iso].dropna(subset=[col])
        b, se, r2 = ols_slope(g["year"].to_numpy(float), g[col].to_numpy())
        rb, rse, _ = ols_slope(g["year"].to_numpy(float), g["rel"].to_numpy())
        lo, hi = g["year"].min(), g["year"].max()
        rows.append(dict(iso3=iso, country=ISO3_TO_NAME.get(iso, iso), n=len(g),
                         slope=b, se=se, r2=r2, rel_slope=rb, rel_se=rse,
                         early=g.loc[g["year"] <= lo + 4, col].mean(),
                         late=g.loc[g["year"] >= hi - 4, col].mean()))
    return pd.DataFrame(rows).sort_values("slope").reset_index(drop=True)


# ── figures ───────────────────────────────────────────────────────────────────
def plot_trajectories(traj, slopes, lq, gseries, gslope, path):
    side = dict(zip(lq["iso3"], lq["log2_lq_overlap"]))
    col = lambda iso: RED if side.get(iso, 0) > 0 else BLUE

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16.5, 7.4), dpi=140, gridspec_kw=dict(width_ratios=[1.5, 1]))

    # cross-country spread: the band, not the individual lines, carries the message
    spread = (traj.groupby("year")["outsiderness"]
              .agg(lo=lambda s: s.quantile(0.25), hi=lambda s: s.quantile(0.75))
              .reset_index())
    ax1.fill_between(spread["year"], spread["lo"], spread["hi"], color="0.55",
                     alpha=0.16, lw=0, zorder=1)

    for iso, g in traj.groupby("iso3"):
        if iso in HIGHLIGHT:
            continue
        g = g.sort_values("year")
        ax1.plot(g["year"], g["outsiderness"], color=col(iso), lw=1.0, alpha=0.22, zorder=2)

    for iso in HIGHLIGHT:
        g = traj[traj["iso3"] == iso].sort_values("year")
        if g.empty:
            continue
        ax1.plot(g["year"], g["outsiderness"], color=col(iso), lw=1.9, alpha=0.95,
                 solid_capstyle="round", zorder=4)
        last = g.iloc[-1]
        ax1.annotate(ISO3_TO_NAME.get(iso, iso), (last["year"], last["outsiderness"]),
                     xytext=(5, 0), textcoords="offset points", fontsize=8.5,
                     color=col(iso), va="center", fontweight="medium", zorder=6)

    ax1.plot(gseries["year"], gseries["value"], color=INK, lw=2.8, ls="--", zorder=5)

    sig = abs(gslope[0]) / gslope[1]
    handles = [
        mlines.Line2D([], [], color=INK, lw=2.8, ls="--", label="all teams (global mean)"),
        mlines.Line2D([], [], color=RED, lw=1.9, label="outsider-tilted country (LQ > 1)"),
        mlines.Line2D([], [], color=BLUE, lw=1.9, label="core-tilted country (LQ < 1)"),
        mpatches.Patch(color="0.55", alpha=0.16, label="inter-quartile range across countries"),
    ]
    ax1.legend(handles=handles, loc="lower left", fontsize=8, ncol=1, frameon=True,
               facecolor="white", edgecolor="none", framealpha=0.88).set_zorder(7)

    ax1.set_xlabel("Year   (3-year rolling window, centred)")
    ax1.set_ylabel("Outsiderness  =  1 − literature-density percentile\n"
                   "(higher = working where the literature of the day is sparse)")
    # ax1.set_title("Do countries stop being outsiders?\n"
    #               f"Global trend {gslope[0]:+.5f}/yr — {sig:.1f}σ, "
    #               f"{'no detectable change over two decades' if sig < 2 else 'significant'}",
    #               fontsize=12)
    ax1.set_title("Do countries stop being outsiders?",
                  fontsize=12)
    ax1.grid(alpha=0.18)
    ax1.xaxis.set_major_locator(MultipleLocator(2))
    ax1.set_xlim(traj["year"].min() - 0.4, traj["year"].max() + 1.7)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    s = slopes.dropna(subset=["slope", "se"]).sort_values("slope").reset_index(drop=True)
    yv = np.arange(len(s))
    ax2.errorbar(s["slope"], yv, xerr=1.96 * s["se"], fmt="none",
                 ecolor="0.65", elinewidth=1.2, capsize=2.5)
    ax2.scatter(s["slope"], yv, s=34, c=[col(i) for i in s["iso3"]],
                edgecolor="white", linewidth=0.6, zorder=3)
    ax2.axvline(0, color=INK, lw=1.1, ls="--", alpha=0.85)
    ax2.set_yticks(yv)
    ax2.set_yticklabels([f"{r.country}  (n={r.n})" for r in s.itertuples()], fontsize=8)
    ax2.set_xlabel("Change in outsiderness per year   (team-level OLS, 95% CI)")
    ax2.set_title("← becoming an insider          becoming more outsider →", fontsize=10)
    ax2.grid(axis="x", alpha=0.22)
    ax2.set_ylim(-0.8, len(s) - 0.2)
    n_cross = int(((s["slope"] - 1.96 * s["se"] < 0) & (s["slope"] + 1.96 * s["se"] > 0)).sum())
    ax2.text(0.5, -0.115, f"{n_cross} of {len(s)} confidence intervals span zero",
             transform=ax2.transAxes, ha="center", fontsize=8.5, color="0.35")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved → {path}")


def plot_field_check(g_contemp, g_fixed, sl_contemp, sl_fixed, path):
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=140)
    ax.plot(g_contemp["year"], g_contemp["value"], color=INK, lw=2.6,
            marker="o", ms=4, mew=0, label="contemporaneous field (papers ≤ team year)")
    ax.plot(g_fixed["year"], g_fixed["value"], color=RED, lw=2.2, ls="--",
            marker="s", ms=4, mew=0, label="fixed all-years field (biased)")
    for s, y, c in [(sl_contemp, 0.055, INK), (sl_fixed, 0.012, RED)]:
        ax.text(0.985, y, f"slope {s[0]:+.5f}/yr   ({abs(s[0]) / s[1]:.1f}σ)",
                transform=ax.transAxes, ha="right", fontsize=9, color=c)
    ax.set_xlabel("Year   (3-year rolling window, centred)")
    ax.set_ylabel("Mean outsiderness, all teams")
    ax.set_title("Why the literature field must be contemporaneous\n"
                 "Pooling all years scores early pioneers against literature that did not yet exist,\n"
                 "manufacturing a spurious rise in outsiderness", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.56), fontsize=9, frameon=False)
    ax.grid(alpha=0.18)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(SEED)
    dp, dt = load()
    print(f"papers with xy+year: {len(dp):,}   teams with xy+year: {len(dt):,}")

    contemp, fixed, ratio = score_teams(dp, dt, rng)
    dt = dt.assign(contemp=contemp, fixed=fixed, ratio=ratio)
    ok = dt.dropna(subset=["contemp"])

    print(f"\noutsiderness (contemporaneous): mean {ok['contemp'].mean():.3f}  "
          f"sd {ok['contemp'].std():.3f}")
    print(f"corr(contemp, fixed)  = {ok['contemp'].corr(ok['fixed']):+.3f}   "
          f"(Spearman {ok['contemp'].corr(ok['fixed'], method='spearman'):+.3f})")
    print(f"corr(contemp, log2 density-ratio) = {ok['contemp'].corr(ok['ratio']):+.3f}   "
          f"(Spearman {ok['contemp'].corr(ok['ratio'], method='spearman'):+.3f})")

    dt.to_csv(REPORTS / "geography_outsiderness_teams.tsv", sep="\t", index=False)

    # ── global trends, team-level (one row per team, not per team-country)
    gb, gse, gr2 = ols_slope(ok["year"].to_numpy(float), ok["contemp"].to_numpy())
    fb, fse, _ = ols_slope(ok["year"].to_numpy(float), ok["fixed"].to_numpy())
    span = ok["year"].max() - ok["year"].min()
    print(f"\nGLOBAL trend, contemporaneous : {gb:+.5f}/yr  (SE {gse:.5f}, "
          f"{abs(gb)/gse:.1f}σ, R²={gr2:.4f})   total over {span} yr: {gb*span:+.3f}")
    print(f"GLOBAL trend, fixed all-years: {fb:+.5f}/yr  (SE {fse:.5f}, "
          f"{abs(fb)/fse:.1f}σ)   total over {span} yr: {fb*span:+.3f}  ← artefact")

    years = np.arange(ok["year"].min() + HALF, ok["year"].max() - HALF + 1)
    g_contemp = rolling_mean(ok, years, "contemp")
    g_fixed = rolling_mean(ok, years, "fixed")
    g_contemp = g_contemp[g_contemp["n"] >= MIN_TEAMS_GLOBAL].reset_index(drop=True)
    g_fixed = g_fixed[g_fixed["n"] >= MIN_TEAMS_GLOBAL].reset_index(drop=True)

    # ── country panel
    ex = explode_countries(ok)
    traj = trajectories(ex, years)
    enough_windows = traj.groupby("iso3").size()
    totals = ex.groupby("iso3").size()
    keep = sorted(i for i in enough_windows[enough_windows >= MIN_WINDOWS].index
                  if totals.get(i, 0) >= MIN_TEAMS_TOTAL)
    traj = traj[traj["iso3"].isin(keep)].reset_index(drop=True)
    traj = traj[traj["year"] >= g_contemp["year"].min()].reset_index(drop=True)
    print(f"\ncountries plotted: {len(keep)}  ({', '.join(keep)})")

    slopes = country_slopes(ex, keep)
    traj.to_csv(REPORTS / "geography_outsiderness_trajectory.tsv", sep="\t", index=False)

    print("\nPer-country trend in outsiderness (contemporaneous field, team-level OLS):")
    print(f'  {"country":20s}{"n":>6s}{"early":>8s}{"late":>7s}{"slope/yr":>11s}{"σ":>7s}{"rel.slope":>11s}')
    for r in slopes.itertuples():
        s = abs(r.slope) / r.se if r.se and np.isfinite(r.se) else np.nan
        print(f'  {r.country:20s}{r.n:6d}{r.early:8.3f}{r.late:7.3f}'
              f'{r.slope:+11.5f}{s:7.1f}{r.rel_slope:+11.5f}')

    z = slopes["slope"].abs() / slopes["se"]
    print(f"\n  {(slopes['slope'] < 0).sum()}/{len(slopes)} countries trend toward the core; "
          f"{((slopes['slope'] < 0) & (z > 1.96)).sum()} significantly down, "
          f"{((slopes['slope'] > 0) & (z > 1.96)).sum()} significantly up (95%).")
    print("  'rel.slope' nets out the year's global mean: a country's movement relative to\n"
          "   its contemporaries. Unlike the LQ, it is not forced to sum to zero.")

    lq = pd.read_csv(REPORTS / "geography_location_quotient.tsv", sep="\t")
    lq["log2_lq_overlap"] = pd.to_numeric(lq["log2_lq_overlap"], errors="coerce")

    plot_trajectories(traj, slopes, lq, g_contemp, (gb, gse),
                      REPORTS / "geography_outsiderness_trajectory.png")
    plot_field_check(g_contemp, g_fixed, (gb, gse), (fb, fse),
                     REPORTS / "geography_outsiderness_field_check.png")
    print(f"Saved → {REPORTS/'geography_outsiderness_trajectory.tsv'}")
    print(f"Saved → {REPORTS/'geography_outsiderness_teams.tsv'}")


if __name__ == "__main__":
    main()
