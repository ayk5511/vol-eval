"""Phase 2 driver: apply vol-eval to a sample of published vol-forecasting papers.

This script is a *scaffold* — it expects per-paper forecast data to have been
collected and stored under studies/per_paper/<paper_id>/forecasts.parquet
following the schema documented in PHASE_2_AND_3.md and the template at
studies/02_paper_collection_template.csv.

Once a paper has its forecasts.parquet (with columns: actual, winner, runner_up)
the script will apply vol-eval's full battery and write per-paper results to
studies/per_paper/<paper_id>/voleval_result.json.

After all papers are processed, the script aggregates into a single tabulation
of which "wins" survive significance testing.

This file is intentionally separate from the v0 paper so that the v0 paper
ships without depending on Phase 2 data collection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vol_eval import dm_test, model_confidence_set, qlike, spa_test

ROOT = Path(__file__).resolve().parent
PER_PAPER = ROOT / "per_paper"
RESULTS = ROOT / "results"
PER_PAPER.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

REQUIRED_COLUMNS = ("actual", "winner", "runner_up")


def analyze_paper(paper_dir: Path, h: int = 5) -> dict:
    """Apply vol-eval to one paper's forecast panel.

    Expected schema in <paper_dir>/forecasts.parquet:
        actual:    realized volatility (proxy)
        winner:    the author-claimed best-forecaster series
        runner_up: the author-claimed runner-up

    Returns a dict suitable for JSON serialization with full vol-eval output.
    """
    fp = paper_dir / "forecasts.parquet"
    if not fp.exists():
        return {"error": "missing forecasts.parquet", "paper_id": paper_dir.name}

    df = pd.read_parquet(fp)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return {"error": f"missing required columns {missing}", "paper_id": paper_dir.name}

    actual = df["actual"].values
    winner = df["winner"].values
    runner_up = df["runner_up"].values

    out = {
        "paper_id": paper_dir.name,
        "n_observations": int(len(actual)),
        "qlike_winner": qlike(actual, winner),
        "qlike_runner_up": qlike(actual, runner_up),
    }

    # DM on the headline pairwise comparison
    dm = dm_test(actual, winner, runner_up, loss="qlike", h=h)
    out["dm"] = {
        "mean_diff": dm.mean_diff,
        "se": dm.se,
        "t_stat": dm.t_stat,
        "p_value": dm.p_value,
        "winner": dm.winner,
        "rejected_at_5pct": dm.p_value < 0.05,
    }

    # MCS on winner + runner_up + (other models if present in the panel)
    other_models = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if other_models:
        forecasts = {"winner": winner, "runner_up": runner_up}
        for m in other_models:
            forecasts[m] = df[m].values
        rng = np.random.default_rng(2026)
        mcs = model_confidence_set(actual, forecasts, loss="qlike", h=h, alpha=0.10, n_bootstrap=2000, rng=rng)
        out["mcs"] = {
            "alpha": mcs.alpha,
            "survivors": mcs.survivors,
            "winner_survives": "winner" in mcs.survivors,
            "n_survivors": len(mcs.survivors),
            "n_total": len(forecasts),
        }

        # SPA with the winner as benchmark vs all others
        rng = np.random.default_rng(2026)
        competitors = {k: v for k, v in forecasts.items() if k != "winner"}
        spa = spa_test(actual, winner, competitors, loss="qlike", h=h, n_bootstrap=2000, rng=rng)
        out["spa_winner_as_benchmark"] = {
            "statistic": spa.statistic,
            "p_value_consistent": spa.p_value_consistent,
            "rejected_at_5pct": spa.p_value_consistent < 0.05,
        }

    # Persist
    (paper_dir / "voleval_result.json").write_text(json.dumps(out, indent=2))
    return out


def aggregate(per_paper_results: list[dict]) -> dict:
    """Tabulate what fraction of headline 'wins' survive each test."""
    n_total = len(per_paper_results)
    n_with_dm = sum(1 for r in per_paper_results if "dm" in r)
    n_with_mcs = sum(1 for r in per_paper_results if "mcs" in r)

    n_dm_significant = sum(1 for r in per_paper_results if r.get("dm", {}).get("rejected_at_5pct", False))
    n_winner_in_mcs = sum(1 for r in per_paper_results if r.get("mcs", {}).get("winner_survives", False))
    n_spa_rejected = sum(
        1
        for r in per_paper_results
        if r.get("spa_winner_as_benchmark", {}).get("rejected_at_5pct", False)
    )

    return {
        "n_papers_analyzed": n_total,
        "dm": {
            "n_with_dm_run": n_with_dm,
            "n_significant_at_5pct": n_dm_significant,
            "fraction_significant": n_dm_significant / n_with_dm if n_with_dm else None,
        },
        "mcs": {
            "n_with_mcs_run": n_with_mcs,
            "n_winner_in_90pct_mcs": n_winner_in_mcs,
            "fraction_winner_in_mcs": n_winner_in_mcs / n_with_mcs if n_with_mcs else None,
        },
        "spa": {
            "n_winner_as_benchmark_rejected": n_spa_rejected,
            "fraction_winner_rejected": n_spa_rejected / n_with_mcs if n_with_mcs else None,
        },
    }


def main():
    # Discover per-paper directories
    paper_dirs = sorted([d for d in PER_PAPER.iterdir() if d.is_dir()])
    if not paper_dirs:
        print("Phase 2 scaffolding: no per_paper/ subdirectories yet.")
        print("To run Phase 2, populate studies/per_paper/<paper_id>/forecasts.parquet")
        print("for each paper in your sample. See PHASE_2_AND_3.md for the schema.")
        return 0

    print(f"Found {len(paper_dirs)} paper directories. Analyzing each...")
    per_paper_results = []
    for d in paper_dirs:
        print(f"  - {d.name}")
        per_paper_results.append(analyze_paper(d))

    # Save
    (RESULTS / "phase2_per_paper.json").write_text(json.dumps(per_paper_results, indent=2))
    summary = aggregate(per_paper_results)
    (RESULTS / "phase2_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
