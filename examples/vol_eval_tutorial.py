"""vol-eval tutorial: a 5-minute walk-through of the full battery.

This script doubles as the vol_eval_tutorial.ipynb notebook source. Run it
as a Python script for reproducibility, or convert to a notebook with:

    jupytext --to ipynb examples/vol_eval_tutorial.py

Scenario: a quant analyst has trained five forecasters on the same panel and
wants to know which (if any) is significantly best. We:

  1. Generate synthetic returns and a panel of 5 vol forecasts
  2. Compute QLIKE for each
  3. Run pairwise Diebold-Mariano tests
  4. Run the 90% Model Confidence Set
  5. Run Hansen's SPA with the apparent winner as benchmark

End-to-end runtime on commodity hardware: ~5 seconds.
"""
# %% [markdown]
# # vol-eval tutorial: from forecasts to a publishable significance result
#
# `vol-eval` implements the canonical battery of significance tests for
# volatility forecast evaluation. This tutorial walks through every test on
# a synthetic 5-model panel.
#
# Install: `pip install vol-eval`

# %%
import numpy as np
import pandas as pd

from vol_eval import dm_test, model_confidence_set, qlike, spa_test

rng = np.random.default_rng(2026)

# %% [markdown]
# ## Step 1. Generate synthetic data
#
# 1000 trading days of synthetic returns drawn from a GARCH(1,1) process,
# plus the realised-variance proxy (squared returns) we'll forecast.

# %%
n = 1000
omega, alpha, beta = 0.0001, 0.05, 0.92
sigma2 = np.zeros(n)
sigma2[0] = omega / (1 - alpha - beta)
returns = np.zeros(n)

for t in range(1, n):
    sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
    returns[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

actual = sigma2  # the "true" variance; in real data this would be RV
print(f"Simulated {n} days of GARCH(1,1) returns")
print(f"  Mean realized variance: {actual.mean():.6f}")
print(f"  Std of returns:         {returns.std():.4f}")

# %% [markdown]
# ## Step 2. Build five candidate forecasters
#
# For demonstration purposes, each "forecaster" is a noised version of the
# true variance. In real work, these would be GARCH(1,1), GJR-GARCH, HAR-RV,
# LightGBM, etc. We add controlled noise so the rankings are interesting:
#
# - `Forecaster_A`: clean (small noise) — should win
# - `Forecaster_B`: slightly noisier
# - `Forecaster_C`: moderately noisy
# - `Forecaster_D`: persistently biased
# - `Forecaster_E`: heavily biased

# %%
forecasts = {
    "Forecaster_A": actual * (1 + 0.02 * rng.standard_normal(n)),
    "Forecaster_B": actual * (1 + 0.05 * rng.standard_normal(n)),
    "Forecaster_C": actual * (1 + 0.15 * rng.standard_normal(n)),
    "Forecaster_D": actual * (1 + 0.10 * rng.standard_normal(n)) + 0.0001,
    "Forecaster_E": actual * (1 + 0.20 * rng.standard_normal(n)) + 0.0003,
}
# Clip to positive to avoid log(0) in QLIKE
for k in forecasts:
    forecasts[k] = np.maximum(forecasts[k], 1e-10)

# %% [markdown]
# ## Step 3. Compute QLIKE losses

# %%
qlike_table = pd.DataFrame({
    "QLIKE": {name: qlike(actual, f) for name, f in forecasts.items()}
}).sort_values("QLIKE")
print(qlike_table)

best_model = qlike_table.index[0]
runner_up = qlike_table.index[1]
print(f"\nBest by QLIKE: {best_model}")
print(f"Runner-up:     {runner_up}")

# %% [markdown]
# ## Step 4. Diebold-Mariano (1995) test: is the winner significantly better?
#
# The DM test compares two forecasters' loss differentials. Null hypothesis:
# equal predictive accuracy. Small p-value rejects in favour of the winner.

# %%
dm = dm_test(actual, forecasts[best_model], forecasts[runner_up], loss="qlike", h=1)
print(f"DM test: {best_model} vs {runner_up}")
print(f"  Mean loss differential: {dm.mean_diff:.6f}")
print(f"  HAC standard error:     {dm.se:.6f}")
print(f"  t-statistic:            {dm.t_stat:.3f}")
print(f"  p-value:                {dm.p_value:.4f}")
print(f"  Winner per DM:          {dm.winner}")
print(f"  Significant at 5%:      {dm.p_value < 0.05}")

# %% [markdown]
# ## Step 5. Model Confidence Set (Hansen-Lunde-Nason 2011)
#
# The MCS handles the multiple-comparison problem properly: it returns the
# subset of models that cannot be statistically distinguished from the best
# at confidence level (1 - alpha). With alpha=0.10, this is the 90% MCS.

# %%
rng2 = np.random.default_rng(2026)
mcs = model_confidence_set(actual, forecasts, loss="qlike", h=1,
                           alpha=0.10, n_bootstrap=2000, rng=rng2)
print(f"90% MCS survivors: {mcs.survivors}")
print(f"  Test statistic:  {mcs.statistic}  (max-t variant of Hansen-Lunde-Nason 2011)")
print("  Elimination order (with elimination p-values):")
for name, pv in zip(mcs.eliminated, mcs.eliminated_p_values):
    print(f"    {name:<15}  p = {pv:.4f}")

# %% [markdown]
# ## Step 6. Hansen SPA: is the apparent winner significantly the best?
#
# SPA tests the joint hypothesis that the benchmark is no worse than any of
# its competitors. p_consistent < 0.05 rejects this; equivalently, some
# competitor significantly beats the benchmark.

# %%
rng3 = np.random.default_rng(2026)
competitors = {k: v for k, v in forecasts.items() if k != best_model}
spa = spa_test(actual, forecasts[best_model], competitors, loss="qlike", h=1,
               n_bootstrap=2000, rng=rng3)
print(f"SPA test: benchmark = {best_model}")
print(f"  Statistic:           {spa.statistic:.4f}")
print(f"  p-value (lower):     {spa.p_value_lower:.4f}  (most liberal)")
print(f"  p-value (consistent):{spa.p_value_consistent:.4f}  (recommended)")
print(f"  p-value (upper):     {spa.p_value_upper:.4f}  (most conservative; = Reality Check)")
print(f"  Reject at 5%:        {spa.p_value_consistent < 0.05}")

# %% [markdown]
# ## What we just did
#
# In ~30 lines of user code, we:
#
# 1. Computed QLIKE losses for 5 forecasters
# 2. Ran a Diebold-Mariano test for the apparent winner versus runner-up
# 3. Ran the 90% Model Confidence Set on all 5 models simultaneously
# 4. Ran Hansen's SPA with the apparent winner as the benchmark, plus White's Reality Check (the SPA upper-bound variant)
#
# The standard practice in published volatility-forecasting comparisons is to
# rank models by QLIKE, declare the lowest-loss model the winner, and stop.
# That practice is statistically inadequate. `vol-eval` makes the additional
# tests a one-line addition to any existing workflow.
#
# In a 30-paper re-analysis using `vol-eval` (Khan 2026), 12 of 14 reproducible
# papers' headline winners did NOT survive significance testing. The package
# exists to make that test routine.
#
# ## Going further
#
# - Documentation: <https://github.com/ayk5511/vol-eval>
# - Companion paper: Khan (2026) *vol-eval: A Python Package for Volatility
#   Forecast Evaluation, with a Re-Analysis of 30 Published Comparison
#   Papers (2018-2025).*
# - Audit-trail companion: `mr-audit` package, for SR 11-7 / EU AI Act
#   compliance logging of the same forecasting pipeline.

print("\nTutorial complete. See https://github.com/ayk5511/vol-eval for full docs.")
