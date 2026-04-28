"""Reproduce P18 Lei 2021 - TCN with investor attention vs GARCH/HAR/LSTM.

Paper claim: TCN with investor attention (Baidu search index features) >
TCN without attention > GARCH > HAR-RV > ARFIMA > LSTM-with-attention.

Data: paper's DAS: joinquant.com/data (Chinese stock high-frequency) +
Baidu search index. We use yfinance for SSE Composite (^SSEC) as a
representative Chinese index proxy and OMIT the Baidu attention feature
since that data is not freely available.
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
    df = yf.download("000001.SS", start="2010-01-01", end="2023-12-31", auto_adjust=False, progress=False)
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


def har_rv_forecast(returns, refit_every=50):
    """HAR-RV: AR on daily, weekly, monthly aggregated squared returns."""
    sq = returns ** 2
    out = pd.Series(index=returns.index, dtype=float)
    n = len(returns)
    burn = 250
    last_fit = -refit_every
    coefs = None
    for t in range(burn, n):
        if t - last_fit >= refit_every:
            try:
                rv_d = sq.iloc[:t]
                rv_w = sq.iloc[:t].rolling(5).mean()
                rv_m = sq.iloc[:t].rolling(22).mean()
                X = pd.DataFrame({"d": rv_d.shift(1), "w": rv_w.shift(1), "m": rv_m.shift(1)}).dropna()
                y = rv_d.loc[X.index]
                X_arr = np.column_stack([np.ones(len(X)), X.values])
                coefs, *_ = np.linalg.lstsq(X_arr, y.values, rcond=None)
                last_fit = t
            except Exception:
                pass
        if coefs is not None:
            d = float(sq.iloc[t - 1])
            w = float(sq.iloc[t - 5:t].mean())
            m = float(sq.iloc[t - 22:t].mean())
            out.iloc[t] = float(coefs[0] + coefs[1] * d + coefs[2] * w + coefs[3] * m)
    return out


class TCN(nn.Module):
    """Simple temporal convolutional network."""
    def __init__(self, n_features, channels=32, dilations=(1, 2, 4)):
        super().__init__()
        layers = []
        in_c = n_features
        for d in dilations:
            layers.append(nn.Conv1d(in_c, channels, kernel_size=3, dilation=d, padding=d))
            layers.append(nn.ReLU())
            in_c = channels
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (batch, lookback, features) -> (batch, features, lookback)
        x = x.transpose(1, 2)
        h = self.conv(x)
        h = h[:, :, -1]
        return nn.functional.softplus(self.fc(h).squeeze(-1)) + 1e-6


class LSTMNet(nn.Module):
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
    print("Downloading 000001.SS (SSE Composite)...")
    df = download()
    print(f"  {len(df)} observations")

    df["garch_var"] = garch_rolling(df["log_ret"])
    df["har_var"] = har_rv_forecast(df["log_ret"])
    df["target_var"] = df["sq_ret"].shift(-1)
    df = df.dropna(subset=["target_var", "garch_var", "har_var"])
    print(f"Usable: {len(df)}")

    split = int(0.6 * len(df))
    target = df["target_var"].values * TARGET_SCALE

    feats_arr = df[["log_ret", "sq_ret"]].values
    mean = feats_arr[:split].mean(axis=0)
    std = feats_arr[:split].std(axis=0) + 1e-8
    feats_norm = (feats_arr - mean) / std

    X_seq, y_seq = make_seq(feats_norm, target, LOOKBACK)
    train_end = split - LOOKBACK

    print("Training TCN...")
    tcn = TCN(2)
    train_model(tcn, X_seq[:train_end], y_seq[:train_end])
    tcn_pred = predict(tcn, X_seq) / TARGET_SCALE

    print("Training LSTM...")
    lstm = LSTMNet(2)
    train_model(lstm, X_seq[:train_end], y_seq[:train_end])
    lstm_pred = predict(lstm, X_seq) / TARGET_SCALE

    full_tcn = np.concatenate([np.full(LOOKBACK, np.nan), tcn_pred])
    full_lstm = np.concatenate([np.full(LOOKBACK, np.nan), lstm_pred])

    out = pd.DataFrame({
        "actual": df["target_var"].values,
        "GARCH": df["garch_var"].values,
        "HAR_RV": df["har_var"].values.clip(min=1e-9),
        "TCN": full_tcn,
        "LSTM": full_lstm,
    }, index=df.index)
    out_test = out.iloc[split:].dropna()
    print(f"Test: {len(out_test)}")
    out_test = out_test.copy()
    out_test["winner"] = out_test["TCN"]
    out_test["runner_up"] = out_test["GARCH"]

    path = OUT_DIR / "forecasts.parquet"
    out_test.to_parquet(path)
    print(f"Wrote {path}")

    log = {
        "data_source": "yfinance 000001.SS (paper used JoinQuant Chinese stock high-frequency)",
        "sample_period": f"{df.index[0].date()} to {df.index[-1].date()}",
        "n_test_used": int(len(out_test)),
        "models": ["GARCH", "HAR_RV", "TCN", "LSTM"],
        "paper_claimed_winner": "TCN with investor attention (Baidu search index)",
        "approximation_notes": [
            "Paper uses Baidu search index as 'investor attention' feature; we OMIT this since Baidu Index data is not freely available.",
            "Our reproduction tests TCN/LSTM/GARCH/HAR-RV horse race WITHOUT the attention feature.",
            "Paper's headline claim is specifically that the ATTENTION feature improves TCN; that claim is not testable here.",
        ],
    }
    (OUT_DIR / "reproduction_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
