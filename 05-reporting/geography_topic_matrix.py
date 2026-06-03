"""Country x overlap-topic participation matrix.

Ties the geography result (Location Quotient) to the precedence result. For each
of the 68 overlap topics — the columns of the diverging precedence chart, ordered
by ``delta_q1`` (iGEM-led on the left, literature-led on the right) — we take the
SAME nearby team projects that defined its precedence and break them down by
country. The rows are the countries of the LQ chart (>=10 teams), ordered by
their overall Location Quotient (most outsider-tilted at the top).

Each cell is a *country x topic* specialization (a Location Quotient at the cell
level):

    LQ(c,t) = (country c's share of topic t's nearby teams)
              ----------------------------------------------
              (country c's share of nearby teams across all overlap topics)

LQ(c,t) > 1 (red) = country c is over-represented in topic t given its own
overall iGEM activity; < 1 (blue) = under-represented. Dot size = number of that
country's teams near the topic (support).

It also tests the hypothesis directly: do outsider-tilted (high-LQ) countries
concentrate in the iGEM-led topics? For each country we compute a participation-
weighted average of the topics' ``delta_q1`` (its "precedence orientation") and
correlate it with the country's overall LQ.

Inputs : joint coords, papers_doc_topics, igem.txt, overlap_precedence_full.tsv,
         geography_location_quotient.tsv
Outputs: assets/reports/geography_topic_matrix.tsv
         assets/reports/geography_topic_matrix.png
         assets/reports/geography_lead_orientation.png
"""
import csv
from collections import Counter
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

ASSETS = Path(__file__).resolve().parents[1] / "assets"
REPORTS = ASSETS / "reports"
MODELS = ASSETS / "topic_models"
RADIUS_MULT, MIN_NEARBY, MIN_TEAMS = 1.5, 3, 10


def load():
    dp = (pd.read_csv(REPORTS / "joint_umap_papers_xy.tsv", sep="\t")
          .merge(pd.read_csv(MODELS / "papers_doc_topics.txt", sep="\t"), on="id"))
    tc = pd.read_csv(REPORTS / "joint_umap_teams_xy.tsv", sep="\t")
    meta = pd.read_csv(ASSETS / "igem.txt", sep="\t", usecols=["UT", "Countries"])
    dt = tc.merge(meta, on="UT")
    dt["country"] = (dt["Countries"].astype(str)
                     .str.replace(",", ";").str.split(";").str[0].str.strip())
    dt = dt[dt["country"].ne("") & dt["country"].ne("nan")].reset_index(drop=True)
    return dp, dt


def nearby_counts(dp, dt, order):
    """Long-form (topic, country, n) over the same neighbourhoods as precedence."""
    dpv = dp[dp["topic"] >= 0]
    tree = cKDTree(dt[["x", "y"]].to_numpy())
    countries = dt["country"].to_numpy()
    rows = []
    for r in order.itertuples():
        pts = dpv.loc[dpv["topic"] == r.topic, ["x", "y"]].to_numpy()
        cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
        rad = np.median(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)) * RADIUS_MULT
        idx = tree.query_ball_point([cx, cy], rad)
        if len(idx) < MIN_NEARBY:
            continue
        for c, n in Counter(countries[i] for i in idx).items():
            rows.append({"topic": int(r.topic), "global_name": r.global_name,
                         "delta_q1": float(r.delta_q1_years), "country": c, "n": n})
    return pd.DataFrame(rows)


def main():
    dp, dt = load()
    order = (pd.read_csv(REPORTS / "overlap_precedence_full.tsv", sep="\t")
             .sort_values("delta_q1_years")[["topic", "global_name", "delta_q1_years"]]
             .reset_index(drop=True))
    lq = pd.read_csv(REPORTS / "geography_location_quotient.tsv", sep="\t")
    lq = lq[lq["log2_lq_overlap"].astype(str).ne("")].copy()
    lq["log2_lq_overlap"] = pd.to_numeric(lq["log2_lq_overlap"], errors="coerce")

    long = nearby_counts(dp, dt, order)

    # pivot to counts matrix (all countries x ordered topics)
    topic_order = order["topic"].tolist()
    M = (long.pivot_table(index="country", columns="topic", values="n",
                          aggfunc="sum", fill_value=0)
         .reindex(columns=topic_order, fill_value=0))
    counts = M.to_numpy(dtype=float)
    Nt = counts.sum(0)                          # per-topic totals
    nc = counts.sum(1)                          # per-country totals
    N = counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        exp = np.outer(nc, Nt) / N              # expected under independence
        lqct = np.where((counts > 0) & (exp > 0), counts / exp, np.nan)
    log2lq = np.log2(lqct)

    # country precedence orientation: participation-weighted mean delta_q1
    dq = order.set_index("topic").loc[topic_order, "delta_q1_years"].to_numpy()
    lead = (counts * dq).sum(1) / np.where(nc > 0, nc, np.nan)
    igem_mask = dq < 0
    frac_igem_lead = counts[:, igem_mask].sum(1) / np.where(nc > 0, nc, np.nan)

    res = pd.DataFrame({"country_iso3": M.index, "n_total_nearby": nc.astype(int),
                        "lead_orientation_delta_q1": np.round(lead, 3),
                        "frac_participation_in_igem_lead": np.round(frac_igem_lead, 3)})
    res = res.merge(lq[["iso3", "country", "igem_n", "log2_lq_overlap", "lq_overlap"]],
                    left_on="country_iso3", right_on="iso3", how="left")

    # rows to display = LQ-chart countries (>=10 teams), ordered by overall LQ desc
    disp = res[res["igem_n"] >= MIN_TEAMS].dropna(subset=["log2_lq_overlap"]).copy()
    disp = disp.sort_values("log2_lq_overlap", ascending=False).reset_index(drop=True)

    # ── hypothesis test ───────────────────────────────────────────────────────
    x = disp["log2_lq_overlap"].to_numpy()
    y = disp["lead_orientation_delta_q1"].to_numpy()
    pr_r, pr_p = pearsonr(x, y)
    sp_r, sp_p = spearmanr(x, y)
    out_grp = disp[disp["log2_lq_overlap"] > 0]
    core_grp = disp[disp["log2_lq_overlap"] < 0]
    print("HYPOTHESIS: do outsider-tilted countries concentrate in iGEM-led topics?")
    print(f"  corr(country log2-LQ, precedence orientation): "
          f"Pearson r={pr_r:.3f} (p={pr_p:.3f}), Spearman rho={sp_r:.3f} (p={sp_p:.3f})")
    print(f"  (negative r => higher-LQ/outsider countries sit in more iGEM-led topics)\n")
    print(f"  Outsider-tilted countries (LQ>1, n={len(out_grp)}): "
          f"mean orientation delta_q1 = {out_grp['lead_orientation_delta_q1'].mean():+.2f}, "
          f"mean frac in iGEM-led topics = {out_grp['frac_participation_in_igem_lead'].mean():.2f}")
    print(f"  Core-tilted countries    (LQ<1, n={len(core_grp)}): "
          f"mean orientation delta_q1 = {core_grp['lead_orientation_delta_q1'].mean():+.2f}, "
          f"mean frac in iGEM-led topics = {core_grp['frac_participation_in_igem_lead'].mean():.2f}\n")
    print("Per-country (ordered as the matrix rows, top = most outsider-tilted):")
    print(f'  {"country":18s}{"LQ":>6s}{"orient":>8s}{"%iGEMlead":>10s}{"nearby":>8s}')
    for r in disp.itertuples():
        print(f'  {r.country:18s}{r.lq_overlap:6.2f}{r.lead_orientation_delta_q1:+8.2f}'
              f'{100*r.frac_participation_in_igem_lead:9.0f}%{r.n_total_nearby:8d}')

    # save long-form matrix with cell LQ
    long_out = long.merge(res[["country_iso3", "country"]], left_on="country",
                          right_on="country_iso3", how="left")
    ct_lq = []
    name_by_iso = dict(zip(M.index, range(len(M.index))))
    topic_idx = {t: j for j, t in enumerate(topic_order)}
    for r in long.itertuples():
        i, j = name_by_iso[r.country], topic_idx[r.topic]
        ct_lq.append(log2lq[i, j])
    long = long.assign(log2_lq_cell=np.round(ct_lq, 3))
    long.to_csv(REPORTS / "geography_topic_matrix.tsv", sep="\t", index=False)

    # ── matrix figure ─────────────────────────────────────────────────────────
    iso_to_name = dict(zip(lq["iso3"], lq["country"]))
    rows_iso = disp["country_iso3"].tolist()
    ri = {iso: k for k, iso in enumerate(rows_iso)}
    boundary = int((dq < 0).sum())              # first literature-led column

    fig, ax = plt.subplots(figsize=(20, 11), dpi=140)
    xs, ys, sz, col = [], [], [], []
    for r in long.itertuples():
        if r.country not in ri:
            continue
        i, j = ri[r.country], topic_idx[r.topic]
        xs.append(j); ys.append(i); sz.append(r.n)
        col.append(np.clip(log2lq[name_by_iso[r.country], j], -2, 2))
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2)
    sizes = 8 + 26 * np.sqrt(np.array(sz))
    sc = ax.scatter(xs, ys, s=sizes, c=col, cmap="RdBu_r", norm=norm,
                    edgecolor="0.3", linewidth=0.3, alpha=0.92)
    ax.axvline(boundary - 0.5, color="black", lw=1.4, ls="--", alpha=0.8)
    ax.text(boundary / 2, -1.4, "◄ iGEM-led topics", ha="center", color="#b2182b", fontsize=11)
    ax.text(boundary + (len(topic_order) - boundary) / 2, -1.4, "literature-led topics ►",
            ha="center", color="#2166ac", fontsize=11)
    ax.set_yticks(range(len(rows_iso)))
    ax.set_yticklabels([f"{iso_to_name.get(s, s)}" for s in rows_iso], fontsize=8)
    ax.set_ylim(len(rows_iso) - 0.5, -2.2)
    ax.set_xticks(range(len(topic_order)))
    ax.set_xticklabels([order.loc[order["topic"] == t, "global_name"].iloc[0][:34]
                        for t in topic_order], rotation=90, fontsize=5.2)
    ax.set_xlim(-1, len(topic_order))
    ax.set_title("Country × overlap-topic participation  "
                 "(rows ordered by Location Quotient; columns by precedence Δq1)\n"
                 "colour = over/under-representation log2 LQ(country,topic); size = nearby teams",
                 fontsize=12)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label("log2 LQ(country, topic)")
    fig.tight_layout()
    fig.savefig(REPORTS / "geography_topic_matrix.png", dpi=160, bbox_inches="tight")
    print(f"\nSaved → {REPORTS/'geography_topic_matrix.png'}")

    # ── hypothesis scatter ────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 6.5), dpi=140)
    ax2.axhline(0, color="0.6", lw=1); ax2.axvline(0, color="0.6", lw=1)
    ax2.scatter(x, y, s=30 + 4 * np.sqrt(disp["n_total_nearby"]),
                c=["#b2182b" if v > 0 else "#2166ac" for v in x], alpha=0.85,
                edgecolor="white", linewidth=0.4)
    for r in disp.itertuples():
        ax2.annotate(r.country, (r.log2_lq_overlap, r.lead_orientation_delta_q1),
                     fontsize=6.5, alpha=0.8, xytext=(3, 2), textcoords="offset points")
    b, a = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 50)
    ax2.plot(xr, a + b * xr, ls=":", color="0.4", lw=1.3)
    ax2.set_xlabel("Country overall Location Quotient  (log2; right = outsider-tilted)")
    ax2.set_ylabel("Precedence orientation  (participation-weighted mean Δq1;\nlower = sits in iGEM-led topics)")
    ax2.set_title(f"Are outsider geographies also on the iGEM-led side?\n"
                  f"Pearson r = {pr_r:.2f} (p={pr_p:.2f}), Spearman ρ = {sp_r:.2f}", fontsize=12)
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(REPORTS / "geography_lead_orientation.png", dpi=160, bbox_inches="tight")
    print(f"Saved → {REPORTS/'geography_lead_orientation.png'}")
    print(f"Saved → {REPORTS/'geography_topic_matrix.tsv'}")


if __name__ == "__main__":
    main()
