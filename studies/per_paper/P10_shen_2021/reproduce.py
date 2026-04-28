"""Reproduce P10 Shen 2021 - Bitcoin RNN vs GARCH/EWMA.

Paper claim: RNN outperforms GARCH and EWMA in average MSE forecasting,
but is less efficient at capturing extreme events (worse at VaR).

Data: paper says CoinMarketCap.com (free, public). The CoinMarketCap public
API now requires a key; we use yfinance BTC-USD as a near-equivalent
substitute (Bitcoin OHLC is commodity-level public data).

Period: paper says 30 April 2010 to 2 August 2020 (in-sample) +
3 August 2020+ (OOS). But Bitcoin daily prices on CoinMarketCap only go back
to mid-2013, and yfinance BTC-USD starts 2014-09-17. We use available range.

Models:
  - GARCH(1,1)
  - EWMA (lambda=0.94, RiskMetrics)
  - RNN (1 layer, 50 hidden, lookback=20) — simple recurrent network

Predictand: 1-day-ahead realized variance proxy = squared log return at t+1.

Output: forecasts.parquet, reproduction_log.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from arch import arch_model

warnings.filterwarnings("ignore")
torch.manual_seed(2026)
np.random.seed(2026)

OUT_DIR = Path(__file__).parent
LOOKBACK = 20
HIDDEN = 50
EPOCHS = 30
BATCH = 64
LR = 1e-3
TARGET_SCALE = 1e4
EWMA_LAMBDA = 0.94


def download_btc() -> pd.DataFrame:
    df = yf.download("BTC-USD", start="2014-09-17", end="2023-12-31", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df["log_ret"] = np.log(df["Close"]).diff()
    df["sq_ret"] = df["log_ret"] ** 2
    return df.dropna()


def garch_rolling_variance(returns: pd.Series, refit_every: int = 20) -> pd.Series:
    out = pd.Series(index=returns.index, dtype=float)
    scaled = returns * 100.0
    n = len(returns)
    burn = 500
    omega = alpha = beta = sigma2 = None
    last_fit_idx = -refit_every
    for t in range(burn, n):
        if t - last_fit_idx >= refit_every:
            try:
                am = arch_model(scaled.iloc[:t], mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
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


def ewma_variance(returns: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    sq = returns ** 2
    sigma2 = pd.Series(index=returns.index, dtype=float)
    sigma2.iloc[0] = float(sq.iloc[0])
    for t in range(1, len(returns)):
        sigma2.iloc[t] = lam * sigma2.iloc[t - 1] + (1 - lam) * float(sq.iloc[t - 1])
    return sigma2


class RNN(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN):
        super().__init__()
        self.rnn = nn.RNN(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return nn.functional.softplus(self.fc(out[:, -1, :]).squeeze(-1)) + 1e-6


def make_seq(features, target, lookback):
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(target[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(X_train, y_train, n_features, epochs=EPOCHS):
    model = RNN(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train)
    n = len(X_t)
    for epoch in range(epochs):
        idx = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            b = idx[i:i + BATCH]
            opt.zero_grad()
            pred = model(X_t[b])
            loss = loss_fn(pred, y_t[b])
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def main():
    print("Downloading BTC-USD from yfinance...")
    df = download_btc()
    print(f"  {len(df)} daily observations from {df.index[0].date()} to {df.index[-1].date()}")

    print("Fitting GARCH(1,1)...")
    df["garch_var"] = garch_rolling_variance(df["log_ret"])
    print("Computing EWMA...")
    df["ewma_var"] = ewma_variance(df["log_ret"])
    df["target_var"] = df["sq_ret"].shift(-1)
    df = df.dropna(subset=["target_var", "garch_var"])
    print(f"  Usable: {len(df)}")

    split = int(0.6 * len(df))
    target = df["target_var"].values * TARGET_SCALE

    feats = ["log_ret", "sq_ret"]
    train_feats = df[feats].iloc[:split].values
    mean = train_feats.mean(axis=0)
    std = train_feats.std(axis=0) + 1e-8
    all_feats = (df[feats].values - mean) / std
    X_seq, y_seq = make_seq(all_feats, target, LOOKBACK)
    train_end = split - LOOKBACK
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]

    print("Training RNN...")
    model = train_model(X_train, y_train, n_features=len(feats))
    pred_all = predict(model, X_seq) / TARGET_SCALE
    rnn_full = np.concatenate([np.full(LOOKBACK, np.nan), pred_all])

    out = pd.DataFrame({
        "actual": df["target_var"].values,
        "GARCH": df["garch_var"].values,
        "EWMA": df["ewma_var"].values,
        "RNN": rnn_full,
    }, index=df.index)
    out_test = out.iloc[split:].dropna()
    print(f"Test: {len(out_test)}")

    out_test = out_test.copy()
    out_test["winner"] = out_test["RNN"]
    out_test["runner_up"] = out_test["GARCH"]

    path = OUT_DIR / "forecasts.parquet"
    out_test.to_parquet(path)
    print(f"Wrote {path}")

    log = {
        "data_source": "yfinance BTC-USD (paper said CoinMarketCap; CoinMarketCap public API now paid)",
        "sample_period": f"{df.index[0].date()} to {df.index[-1].date()}",
        "n_train": int(split),
        "n_test_used": int(len(out_test)),
        "models": ["GARCH", "EWMA", "RNN"],
        "paper_claimed_winner": "RNN",
        "paper_claimed_runner_up": "GARCH or EWMA",
        "approximation_notes": [
            "Paper period is 2010-2020+; yfinance BTC-USD starts 2014.",
            "RNN: 1-layer simple RNN, 50 hidden units, lookback=20, 30 epochs.",
            "GARCH: GARCH(1,1) with refit every 20 days + manual recursion.",
            "EWMA: lambda=0.94 RiskMetrics standard.",
        ],
    }
    (OUT_DIR / "reproduction_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
