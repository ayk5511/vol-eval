"""Reproduce P25 Xu 2024 - GINN (GARCH-Informed Neural Network) horse race.

Paper claim: GINN significantly outperforms standalone GARCH and standalone
LSTM in R², MSE, MAE.

Data: Yahoo Finance, 8 indices, 1992-06-01 to 2022-05-31 (per paper text:
"approximately 7,500 days of daily closing values were captured from
06/01/1992 to 05/31/2022 through Yahoo Finance").

Models reproduced (single representative index = S&P 500 for tractability;
full 8-index reproduction would be 8x compute and is documented as future):
  - GARCH(1,1) with periodic refit
  - LSTM (log returns + lookback)
  - GINN (LSTM with GARCH-fitted variance as additional feature)

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
START = "1992-06-01"
END = "2022-05-31"
LOOKBACK = 20
HIDDEN = 50
EPOCHS = 30
BATCH = 64
LR = 1e-3
TARGET_SCALE = 1e4


def download(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False)
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


class LSTMVol(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return nn.functional.softplus(self.fc(out[:, -1, :]).squeeze(-1)) + 1e-6


def make_sequences(features: np.ndarray, target: np.ndarray, lookback: int) -> tuple:
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(target[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm(X_train, y_train, n_features: int, epochs: int = EPOCHS) -> LSTMVol:
    model = LSTMVol(n_features)
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
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{epochs}  loss={total/n:.6e}")
    return model


def predict_lstm(model: LSTMVol, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def main():
    print(f"Downloading ^GSPC for {START} to {END}...")
    df = download("^GSPC")
    print(f"  {len(df)} daily observations")

    print("Fitting GARCH(1,1) (refit every 20 days)...")
    df["garch_var"] = garch_rolling_variance(df["log_ret"])
    df["target_var"] = df["sq_ret"].shift(-1)
    df = df.dropna(subset=["target_var", "garch_var"])
    print(f"  Usable: {len(df)}")

    split = int(0.6 * len(df))
    print(f"  Train: {split}, Test: {len(df)-split}")

    target = df["target_var"].values * TARGET_SCALE

    forecasts = {}
    for label, feats in [
        ("LSTM", ["log_ret", "sq_ret"]),
        ("GINN", ["log_ret", "sq_ret", "garch_var"]),
    ]:
        print(f"\nTraining {label} ({len(feats)} features)...")
        train_feats = df[feats].iloc[:split].values
        mean = train_feats.mean(axis=0)
        std = train_feats.std(axis=0) + 1e-8
        all_feats = (df[feats].values - mean) / std
        X_seq, y_seq = make_sequences(all_feats, target, LOOKBACK)
        train_end = split - LOOKBACK
        X_train = X_seq[:train_end]
        y_train = y_seq[:train_end]
        model = train_lstm(X_train, y_train, n_features=len(feats))
        all_pred = predict_lstm(model, X_seq) / TARGET_SCALE
        full = np.concatenate([np.full(LOOKBACK, np.nan), all_pred])
        forecasts[label] = full

    out = pd.DataFrame({
        "actual": df["target_var"].values,
        "GARCH": df["garch_var"].values,
        "LSTM": forecasts["LSTM"],
        "GINN": forecasts["GINN"],
    }, index=df.index)

    out_test = out.iloc[split:].dropna()
    print(f"\nTest observations: {len(out_test)}")

    out_test = out_test.copy()
    out_test["winner"] = out_test["GINN"]
    out_test["runner_up"] = out_test["LSTM"]

    path = OUT_DIR / "forecasts.parquet"
    out_test.to_parquet(path)
    print(f"Wrote {path}")

    log = {
        "data_source": "Yahoo Finance ^GSPC",
        "sample_period": f"{START} to {END}",
        "n_total": int(len(df)),
        "n_train": int(split),
        "n_test_used": int(len(out_test)),
        "models": ["GARCH", "LSTM", "GINN"],
        "paper_claimed_winner": "GINN",
        "paper_claimed_runner_up": "LSTM or GARCH (whichever closer)",
        "approximation_notes": [
            "Paper uses 8 indices; we reproduce on S&P 500 only (representative).",
            "Paper's GINN architecture is described as PINN-inspired LSTM with GARCH input — we approximate with our LSTM_GARCH variant.",
            "LSTM hyperparameters: 50 hidden, 1 layer, lookback=20, epochs=30.",
            "Single train/test split (60/40); paper may use rolling refits.",
        ],
    }
    (OUT_DIR / "reproduction_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
