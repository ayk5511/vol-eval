"""Generate Phase 2 QLIKE-ratio bar chart.

For each of the 14 reproduced papers, plot the ratio:
  (QLIKE of paper-claimed winner) / (QLIKE of our best-by-QLIKE model)

A ratio of 1.0 means the paper's winner is also our best (P10 Shen, P15 Mostafa BTC, P21 Aradi).
A ratio > 1.0 means the paper's winner is WORSE than at least one alternative
in our reproduction. The bar is colored by survival outcome.

Output: paper/figures/phase2_qlike_ratio.pdf
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # paper3-vol-eval/
PER_PAPER = ROOT / "studies" / "per_paper"
OUT = Path(__file__).parent / "phase2_qlike_ratio.pdf"

ORDER = [
    ("P10_shen_2021", "P10 Shen", "survives"),
    ("P15_mostafa_2021", "P15 Mostafa (BTC)", "partial"),
    ("P21_aradi_2020", "P21 Aradi", "fails"),
    ("P09_rubio_2022", "P09 Rubio", "fails"),
    ("P11_ersin_2023", "P11 Ersin*", "fails"),
    ("P12_kim_2021", "P12 Kim*", "fails"),
    ("P14_zahid_2022", "P14 Zahid", "fails"),
    ("P18_lei_2021", "P18 Lei*", "fails"),
    ("P20_lux_2020", "P20 Lux", "fails"),
    ("P24_kumar_2024", "P24 Kumar*", "fails"),
    ("P25_xu_2024", "P25 Xu", "fails"),
    ("P26_roszyk_2024", "P26 Roszyk", "fails"),
    ("P27_tondapu_2024", "P27 Tondapu (GBP/USD)", "fails"),
    ("P28_wei_2025", "P28 Wei", "fails"),
]

COLOR = {
    "survives": "#2ca02c",
    "partial":  "#ff7f0e",
    "fails":    "#d62728",
}


def get_ratio(pid: str) -> float:
    """Read the paper's voleval JSON and return claimed_winner_QLIKE / best_QLIKE."""
    p = PER_PAPER / pid / "voleval_result.json"
    d = json.loads(p.read_text())

    # Multi-dataset papers: pick the dataset that drove our story
    representative = {
        "P15_mostafa_2021": "BTC",
        "P09_rubio_2022": "TGLS",
        "P12_kim_2021": "BTC",
        "P27_tondapu_2024": "GBPUSD",
    }

    if "datasets" in d:
        keys = list(d["datasets"].keys())
        rep = representative.get(pid)
        ds = d["datasets"][rep] if rep and rep in d["datasets"] else d["datasets"][keys[0]]
    elif "qlike_by_model" in d:
        ds = d
    elif "GBPUSD" in d:
        ds = d.get(representative.get(pid, "GBPUSD"), d["GBPUSD"])
    else:
        ds = d

    qlikes = ds.get("qlike_by_model", {})
    claimed = ds.get("paper_claimed_winner")
    best = ds.get("best_by_qlike") or ds.get("best_model_observed")
    if not qlikes or not claimed or not best:
        return float("nan")
    return qlikes[claimed] / qlikes[best]


def main():
    rows = []
    for pid, label, outcome in ORDER:
        r = get_ratio(pid)
        rows.append((label, r, outcome))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [r[0] for r in rows]
    ratios = [r[1] for r in rows]
    colors = [COLOR[r[2]] for r in rows]

    y = list(range(len(rows)))
    ax.barh(y, ratios, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("QLIKE ratio: paper-claimed winner / our best-by-QLIKE  (log scale)\n(1.0 = paper winner is also our best; > 1.0 = paper winner is worse)", fontsize=10)
    ax.set_title("Phase 2 reproduction: per-paper QLIKE ratio", fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=COLOR["survives"], edgecolor="black", label="Survives (n=1)"),
        Patch(facecolor=COLOR["partial"],  edgecolor="black", label="Partial (n=1)"),
        Patch(facecolor=COLOR["fails"],    edgecolor="black", label="Fails (n=12)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9, framealpha=0.95)

    # Annotate each bar with its numeric ratio
    for i, r in enumerate(ratios):
        if r == r:  # not nan
            ax.text(r * 1.04, i, f"{r:.2f}", va="center", fontsize=8)

    ax.set_xlim(0.95, max(ratios) * 1.5)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print("\nRatios:")
    for label, r, outcome in zip(labels, ratios, [x[2] for x in rows]):
        print(f"  {label:30}  ratio={r:.3f}  ({outcome})")


if __name__ == "__main__":
    main()
