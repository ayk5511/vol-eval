"""Generate v2 cross-section figures for Paper 3.

Two new figures:
  Fig. 2: Year-by-year mean strict RDS (2018-2025) showing the disclosure trend.
  Fig. 3: Survival outcome by methodology class (hybrid / pure ML / pure econometric).

Both source from studies/results/phase2_rds_strict.csv,
                  studies/results/phase2_headline_models.csv,
                  studies/results/phase2_summary.json.

Outputs:
  paper/figures/phase2_rds_by_year.pdf
  paper/figures/phase2_outcome_by_methodology.pdf
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "studies" / "results"
FIGS = Path(__file__).parent


# ---- Fig. 2: Year-by-year RDS ----

def fig_year_rds():
    with (RESULTS / "phase2_rds_strict.csv").open() as f:
        rows = list(csv.DictReader(f))
    by_year = defaultdict(list)
    for r in rows:
        by_year[int(r["year"])].append(int(r["strict_rds"]))

    years = sorted(by_year.keys())
    means = [np.mean(by_year[y]) for y in years]
    counts = [len(by_year[y]) for y in years]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(years, means, color="#4c72b0", edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.axhline(0.78, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Khan2026Survey broader-sample mean = 0.78")
    ax.axhline(0.47, color="#c44e52", linestyle=":", linewidth=0.8, alpha=0.6,
               label="Our 30-paper sample mean = 0.47")

    for i, (y, m, n) in enumerate(zip(years, means, counts)):
        ax.text(y, m + 0.02, f"n={n}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Year of publication", fontsize=10)
    ax.set_ylabel("Mean strict RDS (this year)", fontsize=10)
    ax.set_title("Phase 2 Fig. 2: Year-by-year mean strict RDS, 2018-2025", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(years)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGS / "phase2_rds_by_year.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")
    print("  Year   n   mean")
    for y, n, m in zip(years, counts, means):
        print(f"  {y}  {n:>3}  {m:>5.2f}")


# ---- Fig. 3: Outcome by methodology class ----

def fig_outcome_by_methodology():
    # Load classifications
    with (RESULTS / "phase2_headline_models.csv").open() as f:
        headline = {r["paper_id"]: r for r in csv.DictReader(f)}

    with (RESULTS / "phase2_summary.json").open() as f:
        summary = json.load(f)
    outcomes = {p["paper_id"]: p["outcome"] for p in summary["per_paper"]}

    # Classify each paper by its winner's model class
    def classify(winner: str) -> str:
        w = winner.lower()
        if "hybrid" in w or "garch-" in w or "-garch" in w or "midas-lstm" in w or "realized garch" in w:
            return "hybrid"
        if any(t in w for t in ["lstm", "gru", "cnn", "rnn", "transformer",
                                 "neural network", "random forest", "svr", "ann",
                                 "attention", "graph", "gat", "tcn", "autoencoder", "narx"]):
            return "pure ML/DL"
        if any(t in w for t in ["garch", "har", "ewma", "arima", "stochastic", "bekk"]):
            return "pure econometric"
        return "methodological"

    bucket: dict[str, dict[str, int]] = {
        "hybrid": {"survives": 0, "partial": 0, "fails": 0, "not_reproduced": 0},
        "pure ML/DL": {"survives": 0, "partial": 0, "fails": 0, "not_reproduced": 0},
        "pure econometric": {"survives": 0, "partial": 0, "fails": 0, "not_reproduced": 0},
        "methodological": {"survives": 0, "partial": 0, "fails": 0, "not_reproduced": 0},
    }
    for pid, h in headline.items():
        cat = classify(h["headline_winner"])
        out = outcomes.get(pid, "not_reproduced")
        bucket[cat][out] += 1

    cats = ["hybrid", "pure ML/DL", "pure econometric", "methodological"]
    repro_counts = [
        sum(bucket[c][k] for k in ("survives", "partial", "fails")) for c in cats
    ]
    survives = [bucket[c]["survives"] for c in cats]
    partial = [bucket[c]["partial"] for c in cats]
    fails = [bucket[c]["fails"] for c in cats]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cats))
    width = 0.7

    p1 = ax.bar(x, fails, width, color="#c44e52", edgecolor="black", linewidth=0.5, label="Fails")
    p2 = ax.bar(x, partial, width, bottom=fails, color="#dd8452",
                edgecolor="black", linewidth=0.5, label="Partial")
    p3 = ax.bar(x, survives, width, bottom=[f + p for f, p in zip(fails, partial)],
                color="#55a868", edgecolor="black", linewidth=0.5, label="Survives")

    # Annotate each bar with the per-category total reproduced
    for i, (c, n) in enumerate(zip(cats, repro_counts)):
        total = bucket[c]["survives"] + bucket[c]["partial"] + bucket[c]["fails"] + bucket[c]["not_reproduced"]
        ax.text(i, repro_counts[i] + 0.15, f"{n}/{total}\nreproduced", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylabel("Number of papers", fontsize=10)
    ax.set_title("Phase 2 Fig. 3: Survival outcome by methodology class", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(repro_counts) + 2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGS / "phase2_outcome_by_methodology.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")
    print("  Methodology class breakdown:")
    for c in cats:
        print(f"  {c:<20}  reproduced={repro_counts[cats.index(c)]:<2}  "
              f"survives={bucket[c]['survives']}  partial={bucket[c]['partial']}  "
              f"fails={bucket[c]['fails']}  not_reproduced={bucket[c]['not_reproduced']}")


def main():
    print("=== Fig. 2 (year-by-year RDS) ===")
    fig_year_rds()
    print()
    print("=== Fig. 3 (outcome by methodology) ===")
    fig_outcome_by_methodology()


if __name__ == "__main__":
    main()
