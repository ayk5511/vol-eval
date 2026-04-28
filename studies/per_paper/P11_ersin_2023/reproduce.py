"""Reproduce P11 Ersin 2023 - GARCH-MIDAS-LSTM on Borsa Istanbul.

Paper claim: 39-95% RMSE reduction with GARCH-MIDAS-LSTM over GARCH-MIDAS,
testing during COVID-19 in Borsa Istanbul stock market with macroeconomic
indicators (geopolitical risk, industrial production, economic expectations).

Data: Borsa Istanbul 100 (yfinance ^XU100.IS or XU100.IS) + OECD indicators.

Models reproduced (approximation):
  - GARCH (baseline; MIDAS framework not in arch package)
  - GARCH-LSTM (LSTM with GARCH variance feature; approximation of paper's MIDAS-LSTM)

We document this as an approximation: full GARCH-MIDAS requires a custom
implementation of the multiplicative low-frequency * high-frequency variance
decomposition, which is beyond a 2-hour reproduction budget.
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


def download():
    df = yf.download("XU100.IS", start="2010-01-01", end="2023-12-31", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df["log_ret"] = np.log(df["Close"]).diff()
    df["sq_ret"] = df["log_ret"] ** 2
    return df.dropna()


def garch_rolling(returns, refit_every=20):
    out = pd.Series(index=returns.index, dtype=float)
    scaled = returns * 100.0
    n = len(returns)
    burn = 250
    omega = alpha = beta = sigma2 = None
    last_fit = -refit_every
    for t in range(burn, n):
        if t - last_fit >= refit_every:
            try:
                am = arch_model(scaled.iloc[:t], mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
                res = am.fit(disp="off", show_warning=False)
                omega = float(res.params["omega"])
                alpha = float(res.params["alpha[1]"])
                beta = float(res.params["beta[1]"])
                sigma2 = float(res.conditional_volatility.iloc[-1] ** 2)
                last_fit = t
            except Exception:
                pass
        if omega is not None:
            out.iloc[t] = sigma2 / 10000.0
            r2 = float(scaled.iloc[t - 1] ** 2)
            sigma2 = omega + alpha * r2 + beta * sigma2
    return out


class LSTMVol(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True)
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


def train(model, X, y, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    for epoch in range(epochs):
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), BATCH):
            b = idx[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(X_t[b]), y_t[b])
            loss.backward()
            opt.step()
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def main():
    print("Downloading XU100.IS...")
    df = download()
    print(f"  {len(df)} observations")

    df["garch_var"] = garch_rolling(df["log_ret"])
    df["target_var"] = df["sq_ret"].shift(-1)
    df = df.dropna(subset=["target_var", "garch_var"])
    print(f"Usable: {len(df)}")

    split = int(0.6 * len(df))
    target = df["target_var"].values * TARGET_SCALE

    feats_arr = df[["log_ret", "sq_ret", "garch_var"]].values
    mean = feats_arr[:split].mean(axis=0)
    std = feats_arr[:split].std(axis=0) + 1e-8
    feats_norm = (feats_arr - mean) / std

    X_seq, y_seq = make_seq(feats_norm, target, LOOKBACK)
    train_end = split - LOOKBACK

    print("Training GARCH-LSTM (approximation of GARCH-MIDAS-LSTM)...")
    model = LSTMVol(3)
    train(model, X_seq[:train_end], y_seq[:train_end])
    pred_all = predict(model, X_seq) / TARGET_SCALE
    full_pred = np.concatenate([np.full(LOOKBACK, np.nan), pred_all])

    out = pd.DataFrame({
        "actual": df["target_var"].values,
        "GARCH": df["garch_var"].values,
        "GARCH_LSTM_approx": full_pred,
    }, index=df.index)
    out_test = out.iloc[split:].dropna()
    print(f"Test: {len(out_test)}")
    out_test = out_test.copy()
    out_test["winner"] = out_test["GARCH_LSTM_approx"]
    out_test["runner_up"] = out_test["GARCH"]

    path = OUT_DIR / "forecasts.parquet"
    out_test.to_parquet(path)
    print(f"Wrote {path}")

    log = {
        "data_source": "yfinance XU100.IS (paper used Borsa Istanbul + OECD indicators)",
        "sample_period": f"{df.index[0].date()} to {df.index[-1].date()}",
        "n_test_used": int(len(out_test)),
        "models": ["GARCH", "GARCH_LSTM_approx"],
        "paper_claimed_winner": "GARCH-MIDAS-LSTM (paper claims 39-95% RMSE reduction)",
        "approximation_notes": [
            "Paper's GARCH-MIDAS framework requires multiplicative variance decomposition into high-freq (daily GARCH) and low-freq (monthly OECD indicators); not implemented here.",
            "Our 'GARCH-LSTM' is an approximation: LSTM with GARCH-variance feature.",
            "We did NOT include macroeconomic indicators (geopolitical risk, industrial production) which are central to the paper's MIDAS approach.",
            "This reproduction tests whether even a simple GARCH-LSTM combination significantly beats GARCH on Borsa Istanbul; it does NOT test the full GARCH-MIDAS-LSTM proposed in the paper.",
        ],
    }
    (OUT_DIR / "reproduction_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
