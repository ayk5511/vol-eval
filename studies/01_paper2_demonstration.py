"""Apply vol-eval to Paper 2's seven-model forecast panel.

This is the self-contained empirical demonstration that anchors the v0
companion paper. Phase 2 of the project (re-analysis of 30 published
volatility-forecasting papers) is queued separately; this script demonstrates
that vol-eval recovers the headline findings of Paper 2 and produces every
ancillary statistic the paper needs.

Inputs:
    /Users/akmbp/Documents/EB1A-Profile/papers/paper2-volatility/results/forecasts_5d.parquet

Outputs:
    studies/results/paper2_full_sample.json
    studies/results/paper2_dm_pairs.json
    studies/results/paper2_mcs.json
    studies/results/paper2_spa.json
    studies/results/paper2_subperiod.json

All numerical values reported in the companion paper trace back to one of the
above JSON files. The audit script (paper/audit.py) re-derives them from
forecasts_5d.parquet and verifies agreement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vol_eval import (
    dm_test,
    mae,
    model_confidence_set,
    mse,
    mz_r2,
    qlike,
    rmse,
    spa_test,
)

PAPER2_FORECASTS = Path(
    "/Users/akmbp/Documents/EB1A-Profile/papers/paper2-volatility/results/forecasts_5d.parquet"
)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["GARCH", "EGARCH", "GJR-GARCH", "HAR-RV", "LightGBM", "XGBoost", "Ensemble"]
H_FORECAST = 5


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PAPER2_FORECASTS)
    return df


def full_sample_metrics(df: pd.DataFrame) -> dict:
    actual = df["actual"].values
    out = {}
    for m in MODELS:
        f = df[m].values
        out[m] = {
            "QLIKE": qlike(actual, f),
            "MSE": mse(actual, f),
            "MAE": mae(actual, f),
            "RMSE": rmse(actual, f),
            "MZ_R2": mz_r2(actual, f),
            "N": int(len(actual)),
        }
    return out


def dm_all_pairs(df: pd.DataFrame, loss: str = "qlike") -> dict:
    actual = df["actual"].values
    out = {}
    for i, a in enumerate(MODELS):
        for j, b in enumerate(MODELS):
            if i >= j:
                continue
            r = dm_test(actual, df[a].values, df[b].values, loss=loss, h=H_FORECAST)
            out[f"{a}_vs_{b}"] = {
                "mean_diff": r.mean_diff,
                "se": r.se,
                "t_stat": r.t_stat,
                "p_value": r.p_value,
                "winner": r.winner,
                "n": r.n,
            }
    return out


def mcs_full_panel(df: pd.DataFrame, loss: str = "qlike") -> dict:
    actual = df["actual"].values
    forecasts = {m: df[m].values for m in MODELS}
    rng = np.random.default_rng(2026)
    res = model_confidence_set(
        actual,
        forecasts,
        loss=loss,
        h=H_FORECAST,
        alpha=0.10,
        n_bootstrap=2000,
        rng=rng,
    )
    return {
        "loss": res.loss,
        "alpha": res.alpha,
        "n_bootstrap": res.n_bootstrap,
        "statistic": res.statistic,
        "survivors": res.survivors,
        "eliminated": res.eliminated,
        "eliminated_p_values": res.eliminated_p_values,
    }


def spa_each_model_as_benchmark(df: pd.DataFrame, loss: str = "qlike") -> dict:
    actual = df["actual"].values
    out = {}
    for benchmark_name in MODELS:
        bench = df[benchmark_name].values
        competitors = {m: df[m].values for m in MODELS if m != benchmark_name}
        rng = np.random.default_rng(2026)
        res = spa_test(
            actual, bench, competitors, loss=loss, h=H_FORECAST, n_bootstrap=2000, rng=rng
        )
        out[benchmark_name] = {
            "n_competitors": res.n_competitors,
            "statistic": res.statistic,
            "p_value_lower": res.p_value_lower,
            "p_value_consistent": res.p_value_consistent,
            "p_value_upper": res.p_value_upper,
        }
    return out


def subperiod_metrics(df: pd.DataFrame) -> dict:
    """Calendar-year split (2022 vs 2023+) with full vol-eval treatment."""
    actual_full = df["actual"].values
    is_2022 = np.asarray(df.index.year == 2022)
    is_other = ~is_2022

    out = {"meta": {"n_2022": int(is_2022.sum()), "n_2023plus": int(is_other.sum())}}
    for label, mask in [("2022_high_vol", is_2022), ("2023_2025_lower_vol", is_other)]:
        actual = actual_full[mask]
        models = {}
        for m in MODELS:
            f = df[m].values[mask]
            models[m] = {
                "QLIKE": qlike(actual, f),
                "MSE": mse(actual, f),
                "MAE": mae(actual, f),
                "RMSE": rmse(actual, f),
                "MZ_R2": mz_r2(actual, f),
                "N": int(len(actual)),
            }
        # MCS within subperiod
        rng = np.random.default_rng(2026)
        forecasts = {m: df[m].values[mask] for m in MODELS}
        mcs = model_confidence_set(actual, forecasts, loss="qlike", h=H_FORECAST, alpha=0.10, n_bootstrap=1000, rng=rng)
        out[label] = {
            "models": models,
            "mcs_survivors": mcs.survivors,
            "mcs_eliminated": mcs.eliminated,
        }
    return out


def main():
    df = load_panel()
    print(f"Loaded forecasts: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")

    print("\n[1/5] Full-sample metrics on all 7 models...")
    full = full_sample_metrics(df)
    (RESULTS_DIR / "paper2_full_sample.json").write_text(json.dumps(full, indent=2))

    print("[2/5] Diebold-Mariano on all 21 ordered pairs (loss=QLIKE, h=5)...")
    dm = dm_all_pairs(df, loss="qlike")
    (RESULTS_DIR / "paper2_dm_pairs.json").write_text(json.dumps(dm, indent=2))

    print("[3/5] Model Confidence Set (alpha=0.10, n_bootstrap=2000)...")
    mcs = mcs_full_panel(df, loss="qlike")
    (RESULTS_DIR / "paper2_mcs.json").write_text(json.dumps(mcs, indent=2))

    print("[4/5] SPA test with each model as benchmark in turn...")
    spa = spa_each_model_as_benchmark(df, loss="qlike")
    (RESULTS_DIR / "paper2_spa.json").write_text(json.dumps(spa, indent=2))

    print("[5/5] Subperiod analysis (calendar split + within-subperiod MCS)...")
    sub = subperiod_metrics(df)
    (RESULTS_DIR / "paper2_subperiod.json").write_text(json.dumps(sub, indent=2))

    print(f"\nAll outputs written to {RESULTS_DIR}")
    print("\nKey findings (full-sample QLIKE ranking, lowest is best):")
    rows = [(m, full[m]["QLIKE"]) for m in MODELS]
    rows.sort(key=lambda x: x[1])
    for i, (m, q) in enumerate(rows, start=1):
        print(f"  {i}. {m:<12s}  QLIKE = {q:.4f}")

    print(f"\nMCS survivors (90% confidence): {mcs['survivors']}")

    print("\nKey DM pairs (most cited from Paper 2):")
    for pair in ["GJR-GARCH_vs_GARCH", "Ensemble_vs_GJR-GARCH", "LightGBM_vs_HAR-RV", "XGBoost_vs_LightGBM"]:
        if pair not in dm:
            # Try reverse
            a, b = pair.split("_vs_")
            pair_rev = f"{b}_vs_{a}"
            if pair_rev in dm:
                d = dm[pair_rev]
                print(f"  {b} vs {a}: t={d['t_stat']:+.3f}, p={d['p_value']:.4f}, winner={d['winner']}")
        else:
            d = dm[pair]
            print(f"  {pair.replace('_', ' ')}: t={d['t_stat']:+.3f}, p={d['p_value']:.4f}, winner={d['winner']}")


if __name__ == "__main__":
    main()
