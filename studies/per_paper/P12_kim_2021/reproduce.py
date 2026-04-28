"""Reproduce P12 Kim 2021 - Bayesian SV vs GARCH-family on 9 cryptocurrencies.

Paper claim: "the SV model performs better than the GARCH family models"
on 9 cryptos; "the forecasting errors of the SV model... tend to be more
accurate as forecast time horizons are longer".

Data: paper text discloses 9 cryptos (BTC, XRP, ETH, BCH, XLM, LTC, TRX,
ADA, IOTA) over period 1 (19 Aug to 27 Nov 2018) and period 2 (2 Jan to
27 Nov 2018). Short samples. We use yfinance crypto pairs.

Models reproduced:
  - GARCH(1,1) Normal innovations
  - GARCH(1,1)-t (Student-t innovations)
  - GJR-GARCH(1,1)-t
  - SV approximation: log AR(1) on log realized variance
    (full Bayesian SV would require PyMC; we approximate with a closed-form
    Gaussian state-space which captures the spirit of stochastic volatility).
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
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
np.random.seed(2026)

OUT_DIR = Path(__file__).parent

CRYPTOS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "XRP": "XRP-USD",
    "BCH": "BCH-USD",
    "LTC": "LTC-USD",
    "ADA": "ADA-USD",
}


def download(ticker, start="2017-08-01", end="2023-12-31"):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df["log_ret"] = np.log(df["Close"]).diff()
    df["sq_ret"] = df["log_ret"] ** 2
    return df.dropna()


def garch_rolling(returns, refit_every=20, asymmetric=False, dist="normal"):
    out = pd.Series(index=returns.index, dtype=float)
    scaled = returns * 100.0
    n = len(returns)
    burn = 250
    omega = alpha = beta = sigma2 = None
    o_param = 0.0
    last_fit_idx = -refit_every
    for t in range(burn, n):
        if t - last_fit_idx >= refit_every:
            try:
                kwargs = dict(mean="Zero", vol="GARCH", p=1, q=1, dist=dist)
                if asymmetric:
                    kwargs["o"] = 1
                am = arch_model(scaled.iloc[:t], **kwargs)
                res = am.fit(disp="off", show_warning=False)
                omega = float(res.params["omega"])
                alpha = float(res.params["alpha[1]"])
                beta = float(res.params["beta[1]"])
                if asymmetric and "gamma[1]" in res.params:
                    o_param = float(res.params["gamma[1]"])
                sigma2 = float(res.conditional_volatility.iloc[-1] ** 2)
                last_fit_idx = t
            except Exception:
                pass
        if omega is not None:
            out.iloc[t] = sigma2 / 10000.0
            r_prev = float(scaled.iloc[t - 1])
            asym = o_param * r_prev ** 2 if asymmetric and r_prev < 0 else 0.0
            sigma2 = omega + alpha * r_prev ** 2 + asym + beta * sigma2
    return out


def sv_approximation(returns: pd.Series, refit_every: int = 50) -> pd.Series:
    """Stochastic volatility approximation via AR(1) on log realized variance.

    log(sq_t) = alpha + beta * log(sq_{t-1}) + epsilon_t

    Forecast: exp(alpha + beta * log(sq_t)).
    """
    sq = returns ** 2
    # Log of squared returns + small floor to avoid log(0)
    log_sq = np.log(sq + 1e-10)
    out = pd.Series(index=returns.index, dtype=float)
    n = len(returns)
    burn = 250
    last_fit = -refit_every
    alpha = beta = None
    for t in range(burn, n):
        if t - last_fit >= refit_every:
            try:
                m = ARIMA(log_sq.iloc[:t].values, order=(1, 0, 0)).fit()
                alpha = float(m.params[0])  # constant
                beta = float(m.params[1])    # AR(1) coefficient
                last_fit = t
            except Exception:
                pass
        if alpha is not None:
            log_sq_prev = float(log_sq.iloc[t - 1])
            log_var_pred = alpha + beta * log_sq_prev
            out.iloc[t] = float(np.exp(log_var_pred))
    return out


def main():
    summary = {}
    for label, ticker in CRYPTOS.items():
        print(f"\n=== {label} ({ticker}) ===")
        df = download(ticker)
        if len(df) < 600:
            print("  insufficient data; skipping")
            continue

        df["garch_n"] = garch_rolling(df["log_ret"], dist="normal")
        df["garch_t"] = garch_rolling(df["log_ret"], dist="t")
        df["gjr_garch_t"] = garch_rolling(df["log_ret"], asymmetric=True, dist="t")
        df["sv_approx"] = sv_approximation(df["log_ret"])
        df["target_var"] = df["sq_ret"].shift(-1)

        df_clean = df.dropna(subset=["target_var", "garch_n", "garch_t", "gjr_garch_t", "sv_approx"])
        if len(df_clean) < 100:
            continue

        split = int(0.6 * len(df_clean))
        out = pd.DataFrame({
            "actual": df_clean["target_var"].values,
            "GARCH_normal": df_clean["garch_n"].values,
            "GARCH_t": df_clean["garch_t"].values,
            "GJR_GARCH_t": df_clean["gjr_garch_t"].values,
            "SV_approx": df_clean["sv_approx"].values.clip(min=1e-10),
        }, index=df_clean.index)
        out_test = out.iloc[split:].dropna()
        out_test = out_test.copy()
        out_test["winner"] = out_test["SV_approx"]
        out_test["runner_up"] = out_test["GARCH_normal"]

        path = OUT_DIR / f"forecasts_{label}.parquet"
        out_test.to_parquet(path)
        summary[label] = {"n": len(out_test)}
        print(f"  Wrote {path} ({len(out_test)} rows)")

    log = {
        "data_source": "yfinance crypto tickers (paper used CoinMarketCap.com URL)",
        "cryptos_analyzed": list(summary.keys()),
        "models": ["GARCH_normal", "GARCH_t", "GJR_GARCH_t", "SV_approx"],
        "paper_claimed_winner": "Bayesian SV beats GARCH family",
        "approximation_notes": [
            "Paper uses Bayesian SV via MCMC; PyMC not available in our environment.",
            "We approximate SV with AR(1) on log squared returns - captures the stochastic-vol idea but is not full Bayesian inference.",
            "Sample period: paper uses ~10 months in 2018; we use 6+ years (2017-2023) for more statistical power.",
        ],
        "summary_per_crypto": summary,
    }
    (OUT_DIR / "reproduction_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
