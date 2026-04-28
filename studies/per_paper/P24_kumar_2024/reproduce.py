"""Reproduce P24 Kumar 2024 - Temporal Graph Attention Network on 8 indices.

Paper claim: Temporal GAT > GARCH and other ML methods on 8 global indices.

Data: yfinance for 8 indices (S&P 500, DAX, CAC 40, FTSE, Nifty 50,
Nikkei 225, KOSPI, Hang Seng).

Approximation:
  Full Temporal Graph Attention Network requires PyTorch Geometric;
  not available in our environment. We approximate the multi-index
  cross-attention with a multi-channel LSTM that takes all 8 index
  log returns as features. This captures the 'multi-index information'
  spirit but is NOT a true graph attention network.

Models compared (per index):
  - GARCH(1,1) on individual index returns (paper baseline)
  - LSTM with all-8-indices as features (multi-channel approximation)
  - Multi_LSTM = approximation of Temporal GAT
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

INDICES = {
    "GSPC": "^GSPC",       # S&P 500
    "GDAXI": "^GDAXI",     # DAX
    "FCHI": "^FCHI",       # CAC 40
    "FTSE": "^FTSE",       # FTSE 100
    "NSEI": "^NSEI",       # Nifty 50
    "N225": "^N225",       # Nikkei 225
    "KS11": "^KS11",       # KOSPI
    "HSI": "^HSI",         # Hang Seng
}


def download_all():
    frames = {}
    for label, ticker in INDICES.items():
        df = yf.download(ticker, start="2007-11-01", end="2022-06-30", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close"]].dropna()
        df["log_ret"] = np.log(df["Close"]).diff()
        frames[label] = df.dropna()
    # Align all to common date range
    common_idx = None
    for label, fr in frames.items():
        common_idx = fr.index if common_idx is None else common_idx.intersection(fr.index)
    aligned = {label: fr.reindex(common_idx) for label, fr in frames.items()}
    print(f"  Common date range: {len(common_idx)} days from {common_idx[0].date()} to {common_idx[-1].date()}")
    return aligned


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
                am = arch_model(scaled.iloc[:t].dropna(), mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
                res = am.fit(disp="off", show_warning=False)
                omega = float(res.params["omega"])
                alpha = float(res.params["alpha[1]"])
                beta = float(res.params["beta[1]"])
                sigma2 = float(res.conditional_volatility.iloc[-1] ** 2)
                last_fit = t
            except Exception:
                pass
        if omega is not None and not np.isnan(scaled.iloc[t - 1] if t > 0 else 0):
            out.iloc[t] = sigma2 / 10000.0
            r2 = float(scaled.iloc[t - 1] ** 2) if not np.isnan(scaled.iloc[t - 1]) else 0.0
            sigma2 = omega + alpha * r2 + beta * sigma2
    return out


class MultiIndexLSTM(nn.Module):
    """LSTM taking all 8 indices' log returns + sq returns as features."""
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


def train_model(model, X, y, epochs=EPOCHS):
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
    print("Downloading 8 global indices...")
    frames = download_all()
    print(f"  {len(frames)} indices loaded")

    # Build a common feature matrix: log returns of all 8 indices
    all_log_rets = pd.DataFrame({label: fr["log_ret"] for label, fr in frames.items()}).dropna()
    print(f"  Common-date log returns: {len(all_log_rets)} days")

    summary = {}
    for target_idx in ["GSPC"]:  # representative single-target reproduction
        print(f"\n=== {target_idx} ===")
        target_returns = all_log_rets[target_idx]
        target_var = (target_returns ** 2).shift(-1)

        # GARCH baseline on target index alone
        garch_var = garch_rolling(target_returns)

        # LSTM features: all 8 indices' log returns (multi-channel)
        feats = all_log_rets.values.astype(np.float32)
        train_end_idx = int(0.6 * len(all_log_rets))
        train_feats = feats[:train_end_idx]
        mean = train_feats.mean(axis=0)
        std = train_feats.std(axis=0) + 1e-8
        feats_norm = (feats - mean) / std

        target_arr = target_var.values * TARGET_SCALE
        X_seq, y_seq = make_seq(feats_norm, target_arr, LOOKBACK)
        train_seq_end = train_end_idx - LOOKBACK
        # drop sequences with NaN target
        valid = ~np.isnan(y_seq)
        X_train = X_seq[:train_seq_end][valid[:train_seq_end]]
        y_train = y_seq[:train_seq_end][valid[:train_seq_end]]

        print(f"  Training MultiIndex_LSTM on {target_idx}...")
        model = MultiIndexLSTM(8)
        train_model(model, X_train, y_train)
        pred_all = predict(model, X_seq) / TARGET_SCALE
        full_pred = np.concatenate([np.full(LOOKBACK, np.nan), pred_all])

        out = pd.DataFrame({
            "actual": target_var.values,
            "GARCH": garch_var.values,
            "MultiIndex_LSTM_approx_TGAT": full_pred,
        }, index=all_log_rets.index)
        out_test = out.iloc[train_end_idx:].dropna()
        print(f"  Test: {len(out_test)}")
        out_test = out_test.copy()
        out_test["winner"] = out_test["MultiIndex_LSTM_approx_TGAT"]
        out_test["runner_up"] = out_test["GARCH"]
        path = OUT_DIR / f"forecasts_{target_idx}.parquet"
        out_test.to_parquet(path)
        summary[target_idx] = {"n": len(out_test)}

    log = {
        "data_source": "yfinance 8 indices (paper used Yahoo Finance)",
        "indices": list(INDICES.keys()),
        "models": ["GARCH", "MultiIndex_LSTM_approx_TGAT"],
        "paper_claimed_winner": "Temporal GAT (graph attention network)",
        "approximation_notes": [
            "Full Temporal GAT requires PyTorch Geometric (not in our environment).",
            "We approximate the cross-index information aggregation with a multi-channel LSTM that takes all 8 indices' log returns as features.",
            "The graph structure (correlation- or volatility-spillover-based) is REPLACED by simple feature concatenation; this is not equivalent to attention.",
            "Reproduction is for S&P 500 only (representative); paper covers all 8 indices.",
        ],
        "summary": summary,
    }
    (OUT_DIR / "reproduction_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
