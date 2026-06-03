"""Geographic specialization of the outsider community: Location Quotient (LQ).

LQ_c = (country c's share of iGEM teams) / (country c's share of SynBio papers).

LQ > 1  → country c is *over*-represented in the outsider (iGEM) community
          relative to its weight in the formal literature (outsider-tilted).
LQ < 1  → country c is *under*-represented among the outsiders relative to the
          literature (insider/core-tilted).

The primary comparison uses the 2004-2025 co-existence window (iGEM only exists
then), so the national composition of the two communities is contrasted over the
same period; an all-years variant is reported for robustness.

iGEM country codes are ISO-3166 alpha-3; OpenAlex country names are English.
A crosswalk maps the former onto the latter. Counting matches the published
top-20 figures: one count per team-country, and one count per distinct country
listed on a paper (multi-country papers contribute to each).

Inputs : assets/igem.txt, assets/synbio_openalex.txt
Outputs: assets/reports/geography_location_quotient.tsv
         assets/reports/geography_lq.png
"""
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets"
REPORTS = ASSETS / "reports"
OVERLAP = (2004, 2025)          # iGEM co-existence window
MIN_TEAMS = 10                  # display threshold for the figure / headline table

# iGEM ISO-3166 alpha-3  ->  OpenAlex English name (as it appears in the papers).
ISO3_TO_NAME = {
    "CHN": "China", "USA": "USA", "CAN": "Canada", "DEU": "Germany",
    "GBR": "United Kingdom", "JPN": "Japan", "TWN": "Taiwan", "IND": "India",
    "FRA": "France", "HKG": "Hong Kong", "NLD": "Netherlands", "MEX": "Mexico",
    "SWE": "Sweden", "KOR": "South Korea", "GRC": "Greece", "CHE": "Switzerland",
    "ESP": "Spain", "AUS": "Australia", "BRA": "Brazil", "TUR": "Turkey",
    "DNK": "Denmark", "ISR": "Israel", "BEL": "Belgium", "IDN": "Indonesia",
    "NOR": "Norway", "SGP": "Singapore", "ITA": "Italy", "POL": "Poland",
    "FIN": "Finland", "MAC": "Macao", "EGY": "Egypt", "AUT": "Austria",
    "KAZ": "Kazakhstan", "CHL": "Chile", "LTU": "Lithuania",
    "ARE": "United Arab Emirates", "RUS": "Russian Federation", "THA": "Thailand",
    "PER": "Peru", "COL": "Colombia", "NZL": "New Zealand", "EST": "Estonia",
    "GHA": "Ghana", "BGR": "Bulgaria", "HUN": "Hungary", "PAN": "Panama",
    "SVN": "Slovenia", "IRL": "Ireland", "CRI": "Costa Rica", "CZE": "Czechia",
    "LVA": "Latvia", "ZAF": "South Africa", "QAT": "Qatar", "ECU": "Ecuador",
    "UGA": "Uganda", "PRI": "Puerto Rico", "BOL": "Bolivia", "ARG": "Argentina",
    "ROU": "Romania", "PRT": "Portugal", "PAK": "Pakistan", "KEN": "Kenya",
    "NPL": "Nepal", "KWT": "Kuwait", "SAU": "Saudi Arabia",
    # iGEM-only (no SynBio papers under this name): Jersey, Honduras, DR Congo
    "JEY": "Jersey", "HND": "Honduras", "COD": "DR Congo",
}


def _year(row, col):
    try:
        return int(float(row[col]))
    except (TypeError, ValueError):
        return None


def count_igem(window=None):
    """Teams per ISO-3166 alpha-3 code (one per team-country)."""
    c = Counter()
    with open(ASSETS / "igem.txt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if window:
                y = _year(row, "Year_y")
                if y is None or not (window[0] <= y <= window[1]):
                    continue
            val = (row.get("Countries") or "").strip()
            for cc in (x.strip() for x in val.replace(",", ";").split(";")):
                if cc:
                    c[cc] += 1
    return c


def count_papers(window=None):
    """Papers per English country name (one per distinct country on the paper)."""
    c = Counter()
    with open(ASSETS / "synbio_openalex.txt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if window:
                y = _year(row, "publication_year")
                if y is None or not (window[0] <= y <= window[1]):
                    continue
            val = (row.get("countries") or "").strip()
            for name in set(x.strip() for x in val.split(";") if x.strip()):
                c[name] += 1
    return c


def build_table():
    igem = count_igem(window=OVERLAP)            # iGEM is entirely within OVERLAP
    pap_ov = count_papers(window=OVERLAP)
    pap_all = count_papers(window=None)

    ig_tot = sum(igem.values())
    pov_tot = sum(pap_ov.values())
    pall_tot = sum(pap_all.values())

    rows = []
    for iso3, n_team in igem.items():
        name = ISO3_TO_NAME.get(iso3, iso3)
        n_pov, n_pall = pap_ov.get(name, 0), pap_all.get(name, 0)
        ig_share = n_team / ig_tot
        pov_share = n_pov / pov_tot if pov_tot else 0.0
        pall_share = n_pall / pall_tot if pall_tot else 0.0
        lq_ov = ig_share / pov_share if pov_share > 0 else np.inf
        lq_all = ig_share / pall_share if pall_share > 0 else np.inf
        rows.append({
            "iso3": iso3, "country": name,
            "igem_n": n_team, "igem_share": round(ig_share, 5),
            "lit_n_overlap": n_pov, "lit_share_overlap": round(pov_share, 5),
            "lq_overlap": (round(lq_ov, 3) if np.isfinite(lq_ov) else "inf"),
            "log2_lq_overlap": (round(float(np.log2(lq_ov)), 3) if np.isfinite(lq_ov) and lq_ov > 0 else ""),
            "lit_n_all": n_pall,
            "lq_all": (round(lq_all, 3) if np.isfinite(lq_all) else "inf"),
        })

    def sort_key(r):
        v = r["lq_overlap"]
        return (1, 0) if v == "inf" else (0, -float(v))
    rows.sort(key=sort_key)
    return rows, dict(ig_tot=ig_tot, pov_tot=pov_tot, pall_tot=pall_tot)


def save_tsv(rows, path):
    cols = ["iso3", "country", "igem_n", "igem_share", "lit_n_overlap",
            "lit_share_overlap", "lq_overlap", "log2_lq_overlap", "lit_n_all", "lq_all"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def plot_lq(rows, path, min_teams=MIN_TEAMS):
    sub = [r for r in rows
           if r["igem_n"] >= min_teams and r["log2_lq_overlap"] != ""]
    sub.sort(key=lambda r: float(r["log2_lq_overlap"]))
    labels = [f'{r["country"]}  (n={r["igem_n"]})' for r in sub]
    vals = [float(r["log2_lq_overlap"]) for r in sub]
    colors = ["#b2182b" if v > 0 else "#2166ac" for v in vals]

    fig_h = max(7, 0.34 * len(sub))
    fig, ax = plt.subplots(figsize=(11, fig_h), dpi=150)
    y = np.arange(len(sub))
    ax.barh(y, vals, color=colors, alpha=0.9, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", lw=1.1, ls="--", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("log2 Location Quotient   =   log2( iGEM share / literature share ),  2004–2025")
    ax.set_title("Geographic specialization of the outsider community (iGEM) vs the literature",
                 fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    xmax = max(abs(min(vals)), abs(max(vals))) * 1.05
    ax.set_xlim(-xmax, xmax)
    ax.text(xmax * 0.97, len(sub) - 0.4, "outsider-tilted →", color="#b2182b",
            ha="right", va="center", fontsize=9)
    ax.text(-xmax * 0.97, len(sub) - 0.4, "← core/insider-tilted", color="#2166ac",
            ha="left", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved → {path}")


if __name__ == "__main__":
    rows, tot = build_table()
    REPORTS.mkdir(parents=True, exist_ok=True)
    save_tsv(rows, REPORTS / "geography_location_quotient.tsv")
    plot_lq(rows, REPORTS / "geography_lq.png")

    print(f"\nTotals — iGEM team-countries: {tot['ig_tot']:,} | "
          f"paper-countries 2004-25: {tot['pov_tot']:,} | all-years: {tot['pall_tot']:,}\n")
    hdr = f'{"country":22s}{"teams":>6s}{"iGEM%":>7s}{"lit%":>7s}{"LQ":>7s}{"log2":>7s}'
    print("TOP outsider-tilted (LQ>1, teams>=10):")
    print(hdr)
    shown = [r for r in rows if r["igem_n"] >= MIN_TEAMS and r["log2_lq_overlap"] != ""]
    for r in sorted(shown, key=lambda r: -float(r["lq_overlap"]))[:15]:
        print(f'{r["country"]:22s}{r["igem_n"]:6d}{100*r["igem_share"]:6.1f}%'
              f'{100*r["lit_share_overlap"]:6.1f}%{float(r["lq_overlap"]):7.2f}{float(r["log2_lq_overlap"]):7.2f}')
    print("\nMOST core/insider-tilted (LQ<1, teams>=10):")
    print(hdr)
    for r in sorted(shown, key=lambda r: float(r["lq_overlap"]))[:12]:
        print(f'{r["country"]:22s}{r["igem_n"]:6d}{100*r["igem_share"]:6.1f}%'
              f'{100*r["lit_share_overlap"]:6.1f}%{float(r["lq_overlap"]):7.2f}{float(r["log2_lq_overlap"]):7.2f}')
