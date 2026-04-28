"""Reproduce P09 Rubio 2022 - ARIMA-SVR hybrid vs ARIMA vs SVR on Colombian shares.

Paper claim: hybrid ARIMA-SVR outperforms ARIMA and SVR for forecasting daily
and cumulative returns of selected Colombian companies on NYSE (Bancolombia,
Ecopetrol, Tecnoglass, Grupo Aval).

Note: paper actually targets RETURNS not volatility - but the paper claims
'volatility forecasting' in our literature search. We'll forecast volatility
(squared returns) as our common Phase 2 metric, while documenting that
the paper's actual predictand is returns.

Data: yfinance Colombian ADRs (paper used Yahoo Finance API for Python).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
np.random.seed(2026)

OUT_DIR = Path(__file__).parent

TICKERS = {
    "CIB": "Bancolombia ADR",       # NYSE: CIB
    "EC": "Ecopetrol ADR",
    "TGLS": "Tecnoglass",
    "AVAL": "Grupo Aval ADR",
}


def download(ticker, start="2010-01-01", end="2022-01-01"):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df["log_ret"] = np.log(df["Close"]).diff()
    df["sq_ret"] = df["log_ret"] ** 2
    return df.dropna()


def arima_rolling_var_forecast(returns: pd.Series, order=(1, 0, 1), refit_every: int = 50, train_size: int = 500) -> pd.Series:
    """ARIMA on returns; squared residual variance as 'volatility'.

    For volatility comparison we use ARIMA's residual standard deviation.
    """
    out = pd.Series(index=returns.index, dtype=float)
    n = len(returns)
    last_fit = -refit_every
    sigma2 = float(returns.var())
    for t in range(train_size, n):
        if t - last_fit >= refit_every:
            try:
                m = ARIMA(returns.iloc[:t].values, order=order).fit()
                resid = m.resid
                sigma2 = float(resid.var())
                last_fit = t
            except Exception:
                pass
        out.iloc[t] = sigma2
    return out


def svr_forecast_var(returns: pd.Series, lookback: int = 20, train_frac: float = 0.6) -> pd.Series:
    """SVR predicts squared returns from lagged squared returns."""
    sq = returns ** 2
    df = pd.DataFrame({"target": sq.shift(-1)})
    for lag in range(1, lookback + 1):
        df[f"sq_lag{lag}"] = sq.shift(lag)
    df = df.dropna()
    n = len(df)
    split = int(n * train_frac)
    X_train = df.iloc[:split].drop(columns="target").values
    y_train = df.iloc[:split]["target"].values
    X_all = df.drop(columns="target").values
    sc = StandardScaler().fit(X_train)
    Xn_train = sc.transform(X_train)
    Xn_all = sc.transform(X_all)
    model = SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=1e-6)
    model.fit(Xn_train, y_train)
    pred = model.predict(Xn_all)
    out = pd.Series(np.nan, index=returns.index)
    out.loc[df.index] = pred
    return out


def hybrid_arima_svr_forecast(returns: pd.Series, lookback: int = 20, train_frac: float = 0.6) -> pd.Series:
    """ARIMA-SVR hybrid: ARIMA fits the linear part on returns, SVR fits residual squared."""
    n = len(returns)
    split = int(n * train_frac)
    try:
        m = ARIMA(returns.iloc[:split].values, order=(1, 0, 1)).fit()
        # Get residuals over the FULL sample using the fitted model
        all_resid = returns - m.predict(start=0, end=n - 1)
    except Exception:
        all_resid = returns - returns.mean()
    sq_resid = all_resid ** 2
    df = pd.DataFrame({"target": sq_resid.shift(-1).reindex(returns.index)})
    for lag in range(1, lookback + 1):
        df[f"sq_lag{lag}"] = sq_resid.shift(lag).reindex(returns.index)
    df = df.dropna()
    train_idx = df.index[:int(len(df) * train_frac)]
    X_train = df.loc[train_idx].drop(columns="target").values
    y_train = df.loc[train_idx]["target"].values
    X_all = df.drop(columns="target").values
    sc = StandardScaler().fit(X_train)
    Xn_train = sc.transform(X_train)
    Xn_all = sc.transform(X_all)
    model = SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=1e-6)
    model.fit(Xn_train, y_train)
    pred = model.predict(Xn_all)
    out = pd.Series(np.nan, index=returns.index)
    out.loc[df.index] = pred
    return out


def main():
    summary = {}
    for ticker, name in TICKERS.items():
        print(f"\n=== {ticker} ({name}) ===")
        df = download(ticker)
        if len(df) < 600:
            print(f"  insufficient data ({len(df)}); skipping")
            continue
        print(f"  {len(df)} observations")

        df["arima_var"] = arima_rolling_var_forecast(df["log_ret"])
        df["svr_var"] = svr_forecast_var(df["log_ret"])
        df["hybrid_var"] = hybrid_arima_svr_forecast(df["log_ret"])
        df["target_var"] = df["sq_ret"].shift(-1)

        df_clean = df.dropna(subset=["target_var", "arima_var", "svr_var", "hybrid_var"])
        if len(df_clean) < 100:
            continue

        split = int(0.6 * len(df_clean))
        out = pd.DataFrame({
            "actual": df_clean["target_var"].values,
            "ARIMA": df_clean["arima_var"].values,
            "SVR": df_clean["svr_var"].values.clip(min=1e-9),
            "Hybrid_ARIMA_SVR": df_clean["hybrid_var"].values.clip(min=1e-9),
        }, index=df_clean.index)
        out_test = out.iloc[split:].dropna()
        out_test = out_test.copy()
        out_test["winner"] = out_test["Hybrid_ARIMA_SVR"]
        out_test["runner_up"] = out_test["ARIMA"]

        path = OUT_DIR / f"forecasts_{ticker}.parquet"
        out_test.to_parquet(path)
        summary[ticker] = {"n": len(out_test)}
        print(f"  Wrote {path} ({len(out_test)} rows)")

    log = {
        "data_source": "yfinance (paper: Yahoo Finance API for Python)",
        "tickers_analyzed": list(summary.keys()),
        "models": ["ARIMA", "SVR", "Hybrid_ARIMA_SVR"],
        "paper_claimed_winner": "Hybrid_ARIMA_SVR",
        "approximation_notes": [
            "Paper's predictand is returns, not volatility; we forecast squared returns to fit Phase 2's vol-eval pipeline.",
            "ARIMA(1,0,1); paper does not specify exact order.",
            "SVR uses RBF kernel default hyperparameters; paper does not disclose hyperparams.",
            "Hybrid: ARIMA on returns, SVR on residual squared.",
        ],
        "summary_per_ticker": summary,
    }
    (OUT_DIR / "reproduction_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
