"""vol-eval: volatility forecast evaluation toolkit.

A small, opinionated package for evaluating volatility forecasts. Covers the
loss functions that the literature actually uses (QLIKE, MSE, MAE,
Mincer-Zarnowitz R^2) plus the significance tests that should be reported
alongside any pairwise comparison (Diebold-Mariano with HAC standard error,
Hansen-Lunde-Nason Model Confidence Set, Hansen SPA, White's Reality Check).

Example:
    >>> import numpy as np
    >>> from vol_eval import qlike, dm_test
    >>> actual = np.abs(np.random.randn(252)) * 0.15
    >>> forecast_a = actual + np.random.randn(252) * 0.02
    >>> forecast_b = actual + np.random.randn(252) * 0.025
    >>> qlike(actual, forecast_a)  # lower is better
    >>> result = dm_test(actual, forecast_a, forecast_b, loss="qlike", h=5)
    >>> result.p_value
"""
from __future__ import annotations

from vol_eval.losses import mae, mse, mz_r2, qlike, rmse
from vol_eval.tests import (
    DMResult,
    MCSResult,
    RealityCheckResult,
    SPAResult,
    dm_test,
    model_confidence_set,
    reality_check,
    spa_test,
)

__version__ = "0.1.0"

__all__ = [
    # Losses
    "qlike",
    "mse",
    "mae",
    "rmse",
    "mz_r2",
    # Tests
    "dm_test",
    "model_confidence_set",
    "spa_test",
    "reality_check",
    # Result types
    "DMResult",
    "MCSResult",
    "SPAResult",
    "RealityCheckResult",
]
