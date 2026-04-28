"""vol-eval quickstart: a five-minute walkthrough using simulated data.

This script walks through the entire vol-eval API in roughly the order a
user would encounter it. Run with:

    python examples/quickstart.py

For a notebook version see examples/quickstart.ipynb.
"""
from __future__ import annotations

import numpy as np

from vol_eval import (
    dm_test,
    mae,
    model_confidence_set,
    mse,
    mz_r2,
    qlike,
    reality_check,
    rmse,
    spa_test,
)


def simulate_volatility_panel(n: int = 1000, seed: int = 2026) -> tuple:
    """Simulate a realized-volatility series + four forecasters.

    The forecasters span a quality spectrum: tight, medium, biased, loose.
    """
    rng = np.random.default_rng(seed)
    # Persistent log-vol process
    log_vol = np.cumsum(rng.normal(0, 0.05, n)) - 1.5
    actual = np.exp(log_vol) * 0.1 + 0.01
    # Forecasters with different noise / bias profiles
    forecasts = {
        "tight": np.clip(actual + rng.normal(0, 0.005, n), 1e-4, None),
        "medium": np.clip(actual + rng.normal(0, 0.012, n), 1e-4, None),
        "biased": np.clip(actual * 1.3 + rng.normal(0, 0.008, n), 1e-4, None),
        "loose": np.clip(actual + rng.normal(0, 0.025, n), 1e-4, None),
    }
    return actual, forecasts


def main() -> None:
    actual, forecasts = simulate_volatility_panel(n=1000)
    print("Simulated panel:")
    print(f"  T = {len(actual)} observations")
    print(f"  Models: {list(forecasts.keys())}")
    print()

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    print("Loss functions (lower is better, except MZ R^2):")
    print(f"  {'model':<10s}  {'QLIKE':>8s}  {'MSE':>8s}  {'MAE':>8s}  {'RMSE':>8s}  {'MZ R^2':>8s}")
    for name, fc in forecasts.items():
        print(
            f"  {name:<10s}  {qlike(actual, fc):>8.4f}  {mse(actual, fc):>8.6f}  "
            f"{mae(actual, fc):>8.4f}  {rmse(actual, fc):>8.4f}  {mz_r2(actual, fc):>8.4f}"
        )
    print()

    # ------------------------------------------------------------------
    # Diebold-Mariano: pairwise comparison
    # ------------------------------------------------------------------
    print("Diebold-Mariano on QLIKE (h=5):")
    pairs = [
        ("tight", "medium"),
        ("tight", "loose"),
        ("medium", "loose"),
        ("medium", "biased"),
    ]
    for a, b in pairs:
        r = dm_test(actual, forecasts[a], forecasts[b], loss="qlike", h=5)
        print(f"  {a:<8s} vs {b:<8s}  diff={r.mean_diff:+.5f}  t={r.t_stat:+.3f}  "
              f"p={r.p_value:.4f}  winner={r.winner}")
    print()

    # ------------------------------------------------------------------
    # Model Confidence Set: which models survive at 90% confidence?
    # ------------------------------------------------------------------
    print("Model Confidence Set on QLIKE (alpha=0.10):")
    mcs = model_confidence_set(actual, forecasts, loss="qlike", alpha=0.10, n_bootstrap=500)
    print(f"  Survivors:  {mcs.survivors}")
    print(f"  Eliminated: {mcs.eliminated}")
    if mcs.eliminated_p_values:
        print(f"  Elimination p-values: {[f'{p:.4f}' for p in mcs.eliminated_p_values]}")
    print()

    # ------------------------------------------------------------------
    # Hansen SPA: is the medium forecaster not inferior to its competitors?
    # ------------------------------------------------------------------
    print("Hansen SPA, benchmark='medium', testing against tight/loose/biased:")
    benchmark = forecasts["medium"]
    competitors = {k: v for k, v in forecasts.items() if k != "medium"}
    spa = spa_test(actual, benchmark, competitors, loss="qlike", n_bootstrap=500)
    print(f"  Statistic:        {spa.statistic:.3f}")
    print(f"  p_lower:          {spa.p_value_lower:.4f}  (liberal)")
    print(f"  p_consistent:     {spa.p_value_consistent:.4f}  (recommended)")
    print(f"  p_upper:          {spa.p_value_upper:.4f}  (conservative; equals Reality Check)")
    print()

    # ------------------------------------------------------------------
    # White Reality Check
    # ------------------------------------------------------------------
    print("White Reality Check (equivalent to SPA's upper-bound p-value):")
    rc = reality_check(actual, benchmark, competitors, loss="qlike", n_bootstrap=500)
    print(f"  p-value: {rc.p_value:.4f}")
    print()

    print("Done. See README.md for a full reference of the API.")


if __name__ == "__main__":
    main()
