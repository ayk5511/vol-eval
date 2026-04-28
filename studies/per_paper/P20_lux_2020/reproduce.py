"""Reproduce P20 Lux 2020 - SVR-GARCH-KDE vs parametric GARCH for VaR/vol forecasting.

Paper claim: SVR-GARCH-KDE (nonparametric SVR-based GARCH with KDE for VaR
quantile) outperforms parametric GARCH-Normal and GARCH-t for VaR.

Data: yfinance ^GDAXI (DAX 30) - paper used Yahoo Finance via R's quantmod.

We test the variance-forecast component (not the VaR quantile) since QLIKE
is the standard volatility loss; the SVR-fitted variance vs parametric GARCH
variance comparison is the relevant horse race for vol-eval.

Models:
  - GARCH-Normal (parametric)
  - GARCH-t
  - SVR-GARCH (SVR predicts squared return from lagged squared returns,
    using GARCH residuals as input — approximation of paper's nonparametric
    variance equation)
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
np.random.seed(2026)

OUT_DIR = Path(__file__).parent


def download():
    df = yf.download("^GDAXI", start="2010-01-01", end="2020-01-01", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df["log_ret"] = np.log(df["Close"]).diff()
    df["sq_ret"] = df["log_ret"] ** 2
    return df.dropna()


def garch_rolling(returns, refit_every=20, dist="normal"):
    out = pd.Series(index=returns.index, dtype=float)
    scaled = returns * 100.0
    n = len(returns)
    burn = 500
    omega = alpha = beta = sigma2 = None
    last_fit_idx = -refit_every
    for t in range(burn, n):
        if t - last_fit_idx >= refit_every:
            try:
                am = arch_model(scaled.iloc[:t], mean="Zero", vol="GARCH", p=1, q=1, dist=dist)
                res = am.fit(disp="off", show_warning=False)
                omega = float(res.params["omega"])
                alpha = float(res.params["alpha[1]"])
                beta = float(res.params["beta[1]"])
                sigma2 = float(res.conditional_volatility.iloc[-1] ** 2)
                last_fit_idx = t
            except Exception:
                pass
        if omega is not None:
            out.iloc[t] = sigma2 / 10000.0
            r2 = float(scaled.iloc[t - 1] ** 2)
            sigma2 = omega + alpha * r2 + beta * sigma2
    return out


def svr_garch_variance(returns, lookback=10, train_frac=0.6):
    """SVR predicts squared returns using lagged sq_ret + lagged returns."""
    sq = returns ** 2
    df = pd.DataFrame({"target": sq.shift(-1)})
    for lag in range(1, lookback + 1):
        df[f"sq_lag{lag}"] = sq.shift(lag)
        df[f"r_lag{lag}"] = returns.shift(lag)
    df = df.dropna()
    n = len(df)
    split = int(n * train_frac)
    X_train = df.iloc[:split].drop(columns="target").values
    y_train = df.iloc[:split]["target"].values
    X_all = df.drop(columns="target").values
    sc = StandardScaler().fit(X_train)
    Xn_train = sc.transform(X_train)
    Xn_all = sc.transform(X_all)
    model = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=1e-7)
    model.fit(Xn_train, y_train)
    pred = model.predict(Xn_all).clip(min=1e-10)
    out = pd.Series(np.nan, index=returns.index)
    out.loc[df.index] = pred
    return out


def main():
    print("Downloading ^GDAXI...")
    df = download()
    print(f"  {len(df)} observations")

    print("Fitting GARCH-Normal...")
    df["garch_n"] = garch_rolling(df["log_ret"], dist="normal")
    print("Fitting GARCH-t...")
    df["garch_t"] = garch_rolling(df["log_ret"], dist="t")
    print("Fitting SVR-GARCH...")
    df["svr_garch"] = svr_garch_variance(df["log_ret"])
    df["target_var"] = df["sq_ret"].shift(-1)
    df_clean = df.dropna(subset=["target_var", "garch_n", "garch_t", "svr_garch"])
    print(f"Usable: {len(df_clean)}")

    split = int(0.6 * len(df_clean))
    out = pd.DataFrame({
        "actual": df_clean["target_var"].values,
        "GARCH_normal": df_clean["garch_n"].values,
        "GARCH_t": df_clean["garch_t"].values,
        "SVR_GARCH": df_clean["svr_garch"].values,
    }, index=df_clean.index)
    out_test = out.iloc[split:].dropna()
    out_test = out_test.copy()
    out_test["winner"] = out_test["SVR_GARCH"]
    out_test["runner_up"] = out_test["GARCH_normal"]

    path = OUT_DIR / "forecasts.parquet"
    out_test.to_parquet(path)
    print(f"Wrote {path} ({len(out_test)} rows)")

    log = {
        "data_source": "yfinance ^GDAXI (paper: Yahoo Finance via quantmod R package)",
        "sample_period": f"{df.index[0].date()} to {df.index[-1].date()}",
        "n_test_used": int(len(out_test)),
        "models": ["GARCH_normal", "GARCH_t", "SVR_GARCH"],
        "paper_claimed_winner": "SVR-GARCH-KDE (nonparametric); we test variance-forecast component",
        "approximation_notes": [
            "Paper's KDE quantile estimation for VaR is omitted; we test the variance-forecast horse race only.",
            "SVR with RBF kernel and lagged squared returns + lagged returns as features.",
            "Hyperparameters: C=10, gamma='scale', epsilon=1e-7. Paper does not disclose exact hyperparams.",
        ],
    }
    (OUT_DIR / "reproduction_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
