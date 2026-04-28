"""Apply vol-eval (DM/MCS/SPA) to a per-paper forecasts.parquet and save
voleval_result.json. Generic across reproductions.

Usage:
  python3 studies/12_apply_voleval_per_paper.py <paper_id> [--h H]

Looks for studies/per_paper/<paper_id>/forecasts.parquet (or forecasts_*.parquet)
and writes studies/per_paper/<paper_id>/voleval_result.json.

Schema requirements for forecasts.parquet:
  - 'actual' column (float)
  - 'winner' column (paper-claimed winner forecasts)
  - 'runner_up' column (paper-claimed runner-up forecasts)
  - any other model columns (used in MCS / SPA)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vol_eval import dm_test, model_confidence_set, qlike, spa_test


def analyze_one(parquet_path: Path, h: int = 1) -> dict:
    df = pd.read_parquet(parquet_path)
    # Drop helper aliases for analysis (keep them only as metadata)
    helpers = [c for c in ["winner", "runner_up"] if c in df.columns]
    base = df.drop(columns=helpers)
    base = base.dropna()
    if len(base) == 0:
        return {"error": "no usable rows after dropna"}

    actual = base["actual"].values
    model_cols = [c for c in base.columns if c != "actual"]
    models = {c: base[c].values for c in model_cols}

    # QLIKE per model
    qlikes = {n: float(qlike(actual, f)) for n, f in models.items()}
    best = min(qlikes, key=qlikes.get)

    # Identify paper-claimed winner / runner-up via the dropped helper columns
    paper_winner = paper_runner = None
    if "winner" in df.columns:
        # The winner column is identical to one of the model columns
        wvals = df["winner"].dropna().values
        for c in model_cols:
            if np.allclose(df[c].dropna().values[:len(wvals)], wvals[:len(df[c].dropna())], rtol=1e-9, equal_nan=True):
                paper_winner = c
                break
    if "runner_up" in df.columns:
        rvals = df["runner_up"].dropna().values
        for c in model_cols:
            if c != paper_winner and np.allclose(df[c].dropna().values[:len(rvals)], rvals[:len(df[c].dropna())], rtol=1e-9, equal_nan=True):
                paper_runner = c
                break

    out: dict = {
        "n_observations": int(len(base)),
        "model_columns": model_cols,
        "qlike_by_model": qlikes,
        "best_by_qlike": best,
        "paper_claimed_winner": paper_winner,
        "paper_claimed_runner_up": paper_runner,
    }

    # DM: paper winner vs runner-up
    if paper_winner and paper_runner:
        dm = dm_test(actual, models[paper_winner], models[paper_runner], loss="qlike", h=h)
        out["dm_winner_vs_runner_up"] = {
            "models": f"{paper_winner} vs {paper_runner}",
            "mean_diff": float(dm.mean_diff),
            "t_stat": float(dm.t_stat),
            "p_value": float(dm.p_value),
            "winner": dm.winner,
            "rejected_at_5pct": bool(dm.p_value < 0.05),
        }

    # MCS at 90%
    rng = np.random.default_rng(2026)
    mcs = model_confidence_set(actual, models, loss="qlike", h=h, alpha=0.10, n_bootstrap=2000, rng=rng)
    out["mcs_at_90pct"] = {
        "alpha": 0.10,
        "n_bootstrap": 2000,
        "survivors": list(mcs.survivors),
        "n_survivors": len(mcs.survivors),
        "n_total": len(model_cols),
        "winner_survives": paper_winner in mcs.survivors if paper_winner else None,
    }

    # SPA: paper-claimed winner as benchmark
    if paper_winner:
        rng2 = np.random.default_rng(2026)
        competitors = {n: f for n, f in models.items() if n != paper_winner}
        spa = spa_test(actual, models[paper_winner], competitors, loss="qlike", h=h, n_bootstrap=2000, rng=rng2)
        out["spa_winner_as_benchmark"] = {
            "benchmark": paper_winner,
            "statistic": float(spa.statistic),
            "p_value_consistent": float(spa.p_value_consistent),
            "rejected_at_5pct": bool(spa.p_value_consistent < 0.05),
        }

    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 12_apply_voleval_per_paper.py <paper_id> [--h H]", file=sys.stderr)
        return 1
    pid = sys.argv[1]
    h = 1
    if "--h" in sys.argv:
        h = int(sys.argv[sys.argv.index("--h") + 1])
    paper_dir = Path(__file__).resolve().parent / "per_paper" / pid
    parquets = list(paper_dir.glob("forecasts*.parquet"))
    if not parquets:
        print(f"No forecasts*.parquet found in {paper_dir}", file=sys.stderr)
        return 1
    result = {"paper_id": pid, "h": h, "datasets": {}}
    for p in parquets:
        label = p.stem.replace("forecasts_", "").replace("forecasts", "default")
        print(f"  Analyzing {p.name}...")
        result["datasets"][label] = analyze_one(p, h=h)

    out_path = paper_dir / "voleval_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
