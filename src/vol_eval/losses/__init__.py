"""Volatility forecast loss functions.

All functions accept arrays of realized (proxy) volatility and forecast
volatility, both in the same units (typically annualized standard deviation).
All are mean losses, lower is better, except mz_r2 where higher is better.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import pearsonr

__all__ = ["qlike", "mse", "mae", "rmse", "mz_r2"]


def _validate(actual: ArrayLike, forecast: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    a = np.asarray(actual, dtype=np.float64).ravel()
    p = np.asarray(forecast, dtype=np.float64).ravel()
    if a.shape != p.shape:
        raise ValueError(f"actual and forecast must have same shape, got {a.shape} vs {p.shape}")
    if a.size == 0:
        raise ValueError("inputs must be non-empty")
    mask = ~(np.isnan(a) | np.isnan(p))
    return a[mask], p[mask]


def qlike(actual: ArrayLike, forecast: ArrayLike, eps: float = 1e-8) -> float:
    """Patton (2011) QLIKE loss.

    QLIKE = mean( (sigma^2 / sigmahat^2) - log(sigma^2 / sigmahat^2) - 1 )

    where sigma is the volatility proxy and sigmahat the forecast. QLIKE is
    robust to noise in the volatility proxy (Patton 2011, JoE 160) and is the
    theoretically preferred loss function for volatility forecasting.

    Lower is better. Always non-negative.

    Reference:
        Patton, A.J. (2011). "Volatility Forecast Comparison Using Imperfect
        Volatility Proxies." Journal of Econometrics 160(1), 246-256.
    """
    a, p = _validate(actual, forecast)
    if np.any(p <= 0):
        raise ValueError("forecast must be strictly positive for QLIKE")
    ratio = (a ** 2) / (p ** 2 + eps)
    return float(np.mean(ratio - np.log(ratio + eps) - 1.0))


def mse(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Mean squared error. Lower is better."""
    a, p = _validate(actual, forecast)
    return float(np.mean((a - p) ** 2))


def mae(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Mean absolute error. Lower is better."""
    a, p = _validate(actual, forecast)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Root mean squared error. Lower is better."""
    return float(np.sqrt(mse(actual, forecast)))


def mz_r2(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Mincer-Zarnowitz R-squared.

    The R^2 from regressing actual on forecast. A well-calibrated forecast
    has intercept zero, slope one, and high R^2. Higher is better.

    Reference:
        Mincer, J. and Zarnowitz, V. (1969). "The Evaluation of Economic
        Forecasts." In: Mincer (ed.), Economic Forecasts and Expectations.
    """
    a, p = _validate(actual, forecast)
    if a.size < 2:
        return float("nan")
    corr, _ = pearsonr(a, p)
    return float(corr ** 2)
