"""Reproduce P26 Roszyk 2024 - LSTM/GARCH hybrid horse race on S&P 500 with VIX.

Paper claim: machine learning approaches, particularly the hybrid LSTM models,
significantly outperform the traditional GARCH model. Including the VIX index
in the hybrid model further enhances forecasting ability. Hybrid LSTM-GARCH+VIX
is the headline winner; hybrid LSTM-GARCH (no VIX) is the runner-up.

Data: Yahoo Finance ^GSPC (S&P 500) + ^VIX (VIX index). Period 2000-01-03 to
2023-12-21 per paper.

Models reproduced (simplified single train/test split — paper does walk-forward
which is computationally heavy; results are an approximation):
  - GARCH(1,1) baseline
  - LSTM (log returns + past variance)
  - Hybrid LSTM-GARCH (LSTM with GARCH-fitted variance as feature)
  - Hybrid LSTM-GARCH+VIX (above + VIX)

Predictand: 1-day-ahead realized variance proxy = squared log return at t+1.

Output:
  forecasts.parquet
  reproduction_log.json
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
START = "2000-01-03"
END = "2023-12-21"
LOOKBACK = 20      # days of history fed into LSTM
HIDDEN = 50
EPOCHS = 30
BATCH = 64
LR = 1e-3


def download_data() -> pd.DataFrame:
    sp = yf.download("^GSPC", start=START, end=END, auto_adjust=False, progress=False)
    vix = yf.download("^VIX", start=START, end=END, auto_adjust=False, progress=False)
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = sp.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    df = pd.DataFrame({
        "sp_close": sp["Close"],
        "vix_close": vix["Close"],
    }).dropna()
    df["log_ret"] = np.log(df["sp_close"]).diff()
    df["sq_ret"] = df["log_ret"] ** 2
    df = df.dropna()
    return df


def garch_rolling_variance(returns: pd.Series, refit_every: int = 20) -> pd.Series:
    """Fit GARCH(1,1) periodically; manually update conditional variance between
    refits using the recursion sigma²_{t+1} = omega + alpha * r_t² + beta * sigma²_t.

    This produces a true out-of-sample variance forecast each day, where the
    variance reflects the latest return AND the GARCH parameters fitted on
    history up to the most recent refit.
    """
    out = pd.Series(index=returns.index, dtype=float)
    scaled = returns * 100.0
    n = len(returns)
    burn = 500
    omega = alpha = beta = None
    sigma2 = None  # current conditional variance (in scaled units)
    last_fit_idx = -refit_every
    for t in range(burn, n):
        if t - last_fit_idx >= refit_every:
            try:
                am = arch_model(scaled.iloc[:t], mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
                res = am.fit(disp="off", show_warning=False)
                params = res.params
                omega = float(params["omega"])
                alpha = float(params["alpha[1]"])
                beta = float(params["beta[1]"])
                # Reset sigma2 to the fitted residual variance at end of training
                sigma2 = float(res.conditional_volatility.iloc[-1] ** 2)
                last_fit_idx = t
            except Exception:
                pass
        if omega is not None and sigma2 is not None:
            # 1-step-ahead forecast from current state
            out.iloc[t] = sigma2 / 10000.0
            # Update sigma2 using actual return at t-1 (already observed when we forecast t)
            r2 = float(scaled.iloc[t - 1] ** 2) if t - 1 >= 0 else 0.0
            sigma2 = omega + alpha * r2 + beta * sigma2
    return out


TARGET_SCALE = 1e4  # bring squared returns ~1e-4 up to ~1 scale for stable training


class LSTMVol(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Softplus ensures positive output for variance prediction
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
    print("Downloading S&P 500 + VIX...")
    df = download_data()
    print(f"  {len(df)} daily observations")

    # GARCH
    print("\nFitting GARCH(1,1) (refit every 250 days)...")
    garch_var = garch_rolling_variance(df["log_ret"])
    df["garch_var"] = garch_var

    # Predictand: next-day squared log return
    df["target_var"] = df["sq_ret"].shift(-1)
    df = df.dropna(subset=["target_var", "garch_var"])
    print(f"  Usable observations after burn-in: {len(df)}")

    # Train/test split: train on first 60%, test on remaining 40%
    split = int(0.6 * len(df))
    train = df.iloc[:split]
    test = df.iloc[split:]
    print(f"  Train: {len(train)} days  Test: {len(test)} days")

    # Build feature arrays
    # Feature sets:
    # (a) LSTM:           past sq_ret + log_ret
    # (b) LSTM-GARCH:     past sq_ret + log_ret + garch_var
    # (c) LSTM-GARCH-VIX: above + vix_close

    f_lstm = ["sq_ret", "log_ret"]
    f_lstm_garch = ["sq_ret", "log_ret", "garch_var"]
    f_lstm_garch_vix = ["sq_ret", "log_ret", "garch_var", "vix_close"]

    target = df["target_var"].values * TARGET_SCALE  # scale up for training stability

    # Standardize features using train-only mean/std
    def standardize(feats):
        train_feats = df[feats].iloc[:split].values
        mean = train_feats.mean(axis=0)
        std = train_feats.std(axis=0) + 1e-8
        all_feats = (df[feats].values - mean) / std
        return all_feats

    forecasts: dict[str, np.ndarray] = {}
    for label, feats in [
        ("LSTM", f_lstm),
        ("LSTM_GARCH", f_lstm_garch),
        ("LSTM_GARCH_VIX", f_lstm_garch_vix),
    ]:
        print(f"\nTraining {label} ({len(feats)} features)...")
        X = standardize(feats)
        X_seq, y_seq = make_sequences(X, target, LOOKBACK)
        # split adjusts for lookback
        train_end = split - LOOKBACK
        X_train = X_seq[:train_end]
        y_train = y_seq[:train_end]
        X_test = X_seq[train_end:]
        model = train_lstm(X_train, y_train, n_features=len(feats))
        pred_train = predict_lstm(model, X_train) / TARGET_SCALE
        pred_test = predict_lstm(model, X_test) / TARGET_SCALE
        full = np.concatenate([np.full(LOOKBACK, np.nan), pred_train, pred_test])
        forecasts[label] = full

    # Now align indices and assemble final forecast frame for the test period
    out = pd.DataFrame({
        "actual": df["target_var"].values,
        "GARCH": df["garch_var"].values,
        "LSTM": forecasts["LSTM"],
        "LSTM_GARCH": forecasts["LSTM_GARCH"],
        "LSTM_GARCH_VIX": forecasts["LSTM_GARCH_VIX"],
    }, index=df.index)

    # Restrict to test set only
    out_test = out.iloc[split:].dropna()
    print(f"\nTest-period observations after dropping NaN: {len(out_test)}")

    # Per the paper:
    # winner = LSTM_GARCH_VIX
    # runner_up = LSTM_GARCH
    out_test = out_test.copy()
    out_test["winner"] = out_test["LSTM_GARCH_VIX"]
    out_test["runner_up"] = out_test["LSTM_GARCH"]

    path = OUT_DIR / "forecasts.parquet"
    out_test.to_parquet(path)
    print(f"\nWrote {path}")

    log = {
        "data_source": "Yahoo Finance ^GSPC + ^VIX (matches paper)",
        "sample_period": f"{START} to {END}",
        "n_train": int(split),
        "n_test": int(len(test)),
        "models": ["GARCH", "LSTM", "LSTM_GARCH", "LSTM_GARCH_VIX"],
        "paper_claimed_winner": "LSTM_GARCH_VIX",
        "paper_claimed_runner_up": "LSTM_GARCH",
        "approximation_notes": [
            "Single train/test split (60/40) instead of paper's walk-forward.",
            "GARCH refit every 250 days (computational cost).",
            "LSTM hyperparameters set to standard defaults (50 hidden, 1 layer, lookback=20, epochs=30, batch=64).",
            "Specific hyperparameters in paper were not disclosed.",
        ],
    }
    (OUT_DIR / "reproduction_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
