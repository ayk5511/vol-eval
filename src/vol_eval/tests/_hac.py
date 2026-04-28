"""HAC (heteroskedasticity- and autocorrelation-consistent) standard errors.

Bartlett-kernel Newey-West estimator of the long-run variance of a mean.
Used by all significance tests in vol_eval.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def newey_west_lrvar(d: NDArray[np.float64], q: int) -> float:
    """Newey-West long-run variance estimate of the mean of d.

    Uses Bartlett kernel weights w_k = 1 - k / (q + 1) for k = 1..q.
    Returns the variance of the *mean* (not the variance of d itself), so
    the matching standard error is sqrt(value).

    Args:
        d: 1-D array of loss differentials (already centered or not; the
           function centers them internally).
        q: Bandwidth (number of autocovariance lags). Rule of thumb:
           h - 1 for an h-step-ahead forecast, where h >= 1.

    Returns:
        Long-run variance of the mean. NaN if the estimate is non-positive.
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    n = d.size
    if n < 2:
        return float("nan")
    centered = d - d.mean()
    gamma0 = float(centered @ centered) / n
    s = gamma0
    for k in range(1, q + 1):
        if k >= n:
            break
        gk = float(centered[k:] @ centered[:-k]) / n
        weight = 1.0 - k / (q + 1.0)
        s += 2.0 * weight * gk
    if s <= 0:
        return float("nan")
    return s / n


def newey_west_se(d: NDArray[np.float64], q: int) -> float:
    """Standard error from the Newey-West long-run variance of the mean."""
    v = newey_west_lrvar(d, q)
    return float(np.sqrt(v)) if not np.isnan(v) else float("nan")
