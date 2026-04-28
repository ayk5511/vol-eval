"""Reproduce P28 Wei 2025 - GARCH-GRU vs GARCH-LSTM vs Transformer vs GARCH.

Paper claim: GARCH-GRU consistently outperforms classical GARCH, pipeline-style
hybrids, and Transformer baselines (GARCH-GRU strongest accuracy-efficiency
tradeoff; GARCH-LSTM also superior).

Data: Yahoo Finance per the paper's Data Availability Statement
("openly available at Yahoo Finance"). Period per text: "January 1, 2010 to
December 31, 2020". Indices: S&P 500, Dow Jones Industrial Average (DJI),
NASDAQ Composite (IXIC).

Models reproduced (representative: S&P 500):
  - GARCH(1,1) classical
  - LSTM
  - GRU
  - GARCH-LSTM (LSTM with GARCH-fitted variance as feature)
  - GARCH-GRU
  - Transformer (small encoder-only)
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


def download(ticker):
    df = yf.download(ticker, start="2010-01-01", end="2020-12-31", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df["log_ret"] = np.log(df["Close"]).diff()
    df["sq_ret"] = df["log_ret"] ** 2
    return df.dropna()


def garch_rolling_variance(returns, refit_every=20):
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


class CellModel(nn.Module):
    def __init__(self, n_features, hidden, cell="LSTM"):
        super().__init__()
        if cell == "LSTM":
            self.rnn = nn.LSTM(n_features, hidden, batch_first=True)
        elif cell == "GRU":
            self.rnn = nn.GRU(n_features, hidden, batch_first=True)
        else:
            self.rnn = nn.RNN(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return nn.functional.softplus(self.fc(out[:, -1, :]).squeeze(-1)) + 1e-6


class TinyTransformer(nn.Module):
    """Small encoder-only transformer for sequence -> scalar prediction."""
    def __init__(self, n_features, hidden=64, nhead=4, nlayers=2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=hidden*2, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.enc(h)
        return nn.functional.softplus(self.fc(h[:, -1, :]).squeeze(-1)) + 1e-6


def make_seq(features, target, lookback):
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(target[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train(model, X_train, y_train, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train)
    n = len(X_t)
    for epoch in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, BATCH):
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
    print("Downloading ^GSPC...")
    df = download("^GSPC")
    print(f"  {len(df)} observations")

    df["garch_var"] = garch_rolling_variance(df["log_ret"])
    df["target_var"] = df["sq_ret"].shift(-1)
    df = df.dropna(subset=["target_var", "garch_var"])
    print(f"Usable: {len(df)}")

    split = int(0.6 * len(df))
    target = df["target_var"].values * TARGET_SCALE

    forecasts = {}
    configs = [
        ("LSTM", ["log_ret", "sq_ret"], "lstm_cell"),
        ("GRU", ["log_ret", "sq_ret"], "gru_cell"),
        ("GARCH_LSTM", ["log_ret", "sq_ret", "garch_var"], "lstm_cell"),
        ("GARCH_GRU", ["log_ret", "sq_ret", "garch_var"], "gru_cell"),
        ("Transformer", ["log_ret", "sq_ret"], "transformer"),
    ]
    for label, feats, kind in configs:
        print(f"Training {label}...")
        train_feats = df[feats].iloc[:split].values
        mean = train_feats.mean(axis=0)
        std = train_feats.std(axis=0) + 1e-8
        all_feats = (df[feats].values - mean) / std
        X_seq, y_seq = make_seq(all_feats, target, LOOKBACK)
        train_end = split - LOOKBACK
        if kind == "lstm_cell":
            model = CellModel(len(feats), HIDDEN, "LSTM")
        elif kind == "gru_cell":
            model = CellModel(len(feats), HIDDEN, "GRU")
        else:
            model = TinyTransformer(len(feats))
        train(model, X_seq[:train_end], y_seq[:train_end])
        full = np.concatenate([np.full(LOOKBACK, np.nan), predict(model, X_seq) / TARGET_SCALE])
        forecasts[label] = full

    out = pd.DataFrame({
        "actual": df["target_var"].values,
        "GARCH": df["garch_var"].values,
        **forecasts,
    }, index=df.index)
    out_test = out.iloc[split:].dropna()
    print(f"Test: {len(out_test)}")
    out_test = out_test.copy()
    out_test["winner"] = out_test["GARCH_GRU"]
    out_test["runner_up"] = out_test["GARCH_LSTM"]

    path = OUT_DIR / "forecasts.parquet"
    out_test.to_parquet(path)
    print(f"Wrote {path}")

    log = {
        "data_source": "yfinance ^GSPC (paper: Yahoo Finance via DAS)",
        "sample_period": f"{df.index[0].date()} to {df.index[-1].date()}",
        "n_test_used": int(len(out_test)),
        "models": ["GARCH", "LSTM", "GRU", "GARCH_LSTM", "GARCH_GRU", "Transformer"],
        "paper_claimed_winner": "GARCH_GRU",
        "paper_claimed_runner_up": "GARCH_LSTM",
        "approximation_notes": [
            "We approximate paper's 'embedded GARCH dynamics' with feature-augmented LSTM/GRU.",
            "Single train/test split (60/40); paper does walk-forward.",
            "Transformer is a small 2-layer encoder; paper Transformer architecture not fully disclosed.",
        ],
    }
    (OUT_DIR / "reproduction_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
