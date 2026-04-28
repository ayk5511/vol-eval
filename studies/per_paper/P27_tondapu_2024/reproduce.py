"""Reproduce P27 Tondapu 2024 - GARCH/EWMA/IV horse race on FX volatility.

Paper claim (verbatim from abstract):
  "for the GBP/USD pair, the most accurate volatility forecasts stem from the
  utilization of GARCH models employing a rolling window methodology.
  Conversely, for the EUR/GBP pair, optimal forecasts are derived from GARCH
  models and Ordinary Least Squares (OLS) models incorporating the annualized
  implied volatility of the exchange rate as an independent variable."

Data: Yahoo Finance (per the paper's Section 3.1: "we gathered data from Yahoo
Finance, which included the daily market prices of currency pairs"). Period:
2018-06-15 to 2023-06-15 (per the abstract: "from June 15, 2018, to June 15, 2023").

Predictand: 20-day-ahead realized variance (sum of squared daily log returns
over the next 20 days), per the paper's "20-day variation in the pairs' daily
returns".

Models reproduced:
  - GARCH(1,1) with rolling window of 250 trading days
  - EWMA with lambda = 0.94 (RiskMetrics convention)
  - GARCH(1,1)-t (Student-t innovations) with rolling window
  - GJR-GARCH(1,1) with rolling window

We do NOT reproduce the IV model because we have no FX implied-volatility
data source; the paper's source for IV is unclear. This is documented in
the README of this directory.

Output:
  forecasts_GBPUSD.parquet  - per-day forecasts for GBP/USD
  forecasts_EURGBP.parquet  - per-day forecasts for EUR/GBP
  reproduction_log.json     - what we ran, what we couldn't reproduce, why
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent
START = "2018-06-15"
END = "2023-06-15"
WINDOW = 250    # rolling window length for GARCH (~ 1 year)
H = 20          # 20-day forward forecast horizon
EWMA_LAMBDA = 0.94  # RiskMetrics


def download_fx(ticker: str) -> pd.DataFrame:
    """Yahoo Finance daily FX. Adjust for missing days."""
    df = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df["log_ret"] = np.log(df["Close"]).diff()
    df = df.dropna()
    return df


def realized_var_forward(log_returns: pd.Series, h: int) -> pd.Series:
    """Sum of squared returns over next h days (the 'actual' for QLIKE)."""
    sq = log_returns ** 2
    # rolling sum from t+1 to t+h
    rv = sq.rolling(h, min_periods=h).sum().shift(-h)
    return rv


def garch_rolling_forecast(returns: pd.Series, window: int, h: int, dist: str = "normal", model: str = "GARCH") -> pd.Series:
    """Fit GARCH(1,1) on rolling window, forecast h-step-ahead variance.

    Returns a series indexed by the forecast origin date (i.e., forecast for
    days [t+1, t+h] is recorded at index t).
    """
    forecasts = pd.Series(index=returns.index, dtype=float)
    n = len(returns)
    # Use 100x scaling per arch package convention to avoid numerical issues
    scaled_ret = returns * 100.0
    for t in range(window, n - h):
        sample = scaled_ret.iloc[t - window:t]
        try:
            am = arch_model(sample, mean="Zero", vol=model, p=1, q=1, dist=dist)
            res = am.fit(disp="off", show_warning=False)
            f = res.forecast(horizon=h, reindex=False)
            # Sum of variance forecasts over h-step horizon
            var_h = float(f.variance.values[0].sum())
            # Unscale: variance scaled by 100^2 = 10000
            forecasts.iloc[t] = var_h / 10000.0
        except Exception:
            forecasts.iloc[t] = np.nan
    return forecasts


def ewma_rolling_forecast(returns: pd.Series, lam: float, h: int) -> pd.Series:
    """RiskMetrics EWMA volatility forecast, h-step-ahead variance.

    EWMA: sigma2_t = lambda * sigma2_{t-1} + (1-lambda) * r_{t-1}^2
    h-step forecast = h * sigma2_t (constant variance assumption).
    """
    sq = returns ** 2
    sigma2 = pd.Series(index=returns.index, dtype=float)
    sigma2.iloc[0] = sq.iloc[0]
    for t in range(1, len(returns)):
        sigma2.iloc[t] = lam * sigma2.iloc[t - 1] + (1 - lam) * sq.iloc[t - 1]
    # h-step forecast as multiple of 1-step variance
    return sigma2 * h


def historical_avg_forecast(returns: pd.Series, h: int) -> pd.Series:
    """Backward-looking 20-day average squared return as predictor."""
    sq = returns ** 2
    return sq.rolling(h, min_periods=h).sum()


def main():
    summary = {"models_run": [], "models_skipped": []}
    for ticker, label in [("GBPUSD=X", "GBPUSD"), ("EURGBP=X", "EURGBP")]:
        print(f"\n=== {label} ({ticker}) ===")
        df = download_fx(ticker)
        print(f"  Downloaded {len(df)} daily observations from Yahoo Finance")
        ret = df["log_ret"]

        # Actual: sum of squared returns over next H days
        actual = realized_var_forward(ret, H)

        # Models
        print(f"  Fitting GARCH(1,1) rolling (window={WINDOW})...")
        garch_n = garch_rolling_forecast(ret, WINDOW, H, dist="normal", model="GARCH")
        print("  Fitting GARCH(1,1)-t rolling...")
        garch_t = garch_rolling_forecast(ret, WINDOW, H, dist="t", model="GARCH")
        print("  Fitting GJR-GARCH(1,1) rolling...")
        gjr = garch_rolling_forecast(ret, WINDOW, H, dist="normal", model="GJR-GARCH")
        print(f"  Computing EWMA(λ={EWMA_LAMBDA})...")
        ewma = ewma_rolling_forecast(ret, EWMA_LAMBDA, H)
        print("  Computing historical average...")
        hist = historical_avg_forecast(ret, H)

        # Combine
        out = pd.DataFrame({
            "actual": actual,
            "GARCH_normal_rolling": garch_n,
            "GARCH_t_rolling": garch_t,
            "GJR_GARCH_rolling": gjr,
            "EWMA": ewma,
            "Historical_Avg": hist,
        })
        # Drop rows where actual is NaN (last H days where we don't have realized future)
        out = out.dropna(subset=["actual"])
        out = out.dropna(subset=["GARCH_normal_rolling"])  # also drop early period before window fits
        print(f"  Final usable observations: {len(out)}")

        # Per the paper:
        # GBP/USD: winner=GARCH (rolling). Take GARCH_normal_rolling.
        # EUR/GBP: winner=GARCH or GARCH+OLS+IV. We don't have IV; take GARCH_normal_rolling.
        # Runner-up: EWMA (paper compares against EWMA explicitly)
        out["winner"] = out["GARCH_normal_rolling"]
        out["runner_up"] = out["EWMA"]

        # Save
        path = OUT_DIR / f"forecasts_{label}.parquet"
        out.to_parquet(path)
        print(f"  Wrote {path} ({len(out)} rows)")
        summary["models_run"].append(f"{label}: GARCH_normal, GARCH_t, GJR_GARCH, EWMA, Historical")

    summary["models_skipped"].append(
        "IV models: paper's source for FX implied volatility is unclear. "
        "Yahoo Finance does not provide FX option-implied vol for these pairs. "
        "Reproduction excludes IV; ranking among GARCH-family + EWMA is what we test."
    )
    summary["paper_claim"] = {
        "GBPUSD": "GARCH (rolling) wins per paper abstract",
        "EURGBP": "GARCH or GARCH+OLS+IV wins per paper abstract",
    }
    summary["our_test"] = (
        "Apply DM/MCS/SPA to {GARCH, EWMA, GJR-GARCH, GARCH-t, Historical} "
        "to test whether GARCH wins are statistically significant."
    )

    log_path = OUT_DIR / "reproduction_log.json"
    log_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Wrote {log_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
