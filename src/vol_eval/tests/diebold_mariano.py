"""Diebold-Mariano (1995) two-sided test for equal forecast accuracy.

Pairwise comparison: H0 that two forecasters have the same expected loss
against a chosen loss function (default QLIKE).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from vol_eval.losses import mae, mse, qlike
from vol_eval.tests._hac import newey_west_se

LossFn = Callable[[ArrayLike, ArrayLike], float]
LossName = Literal["qlike", "mse", "mae"]

LOSS_REGISTRY: dict[str, LossFn] = {
    "qlike": qlike,
    "mse": mse,
    "mae": mae,
}


@dataclass(frozen=True)
class DMResult:
    """Result of a Diebold-Mariano two-sided test.

    Attributes:
        n: Effective sample size (after dropping NaN).
        mean_diff: Sample mean of loss(model_a) - loss(model_b). Negative
            means model_a has lower loss (i.e., model_a wins).
        se: HAC (Newey-West) standard error of the mean differential.
        t_stat: Test statistic mean_diff / se.
        p_value: Two-sided p-value under standard-normal asymptotic null.
        loss: Name of the loss function used.
        h: Forecast horizon used to set the HAC bandwidth (q = h - 1).
        winner: "a", "b", or "tie" depending on sign and significance.
    """

    n: int
    mean_diff: float
    se: float
    t_stat: float
    p_value: float
    loss: str
    h: int

    @property
    def winner(self) -> str:
        """Return 'a' if model_a wins at 5%, 'b' if model_b wins at 5%, else 'tie'."""
        if np.isnan(self.t_stat) or self.p_value > 0.05:
            return "tie"
        return "a" if self.mean_diff < 0 else "b"

    def __repr__(self) -> str:
        return (
            f"DMResult(loss={self.loss!r}, h={self.h}, n={self.n}, "
            f"mean_diff={self.mean_diff:+.5f}, t={self.t_stat:+.3f}, "
            f"p={self.p_value:.4f}, winner={self.winner!r})"
        )


def _pointwise_loss(actual: NDArray[np.float64], forecast: NDArray[np.float64], loss: str, eps: float = 1e-8) -> NDArray[np.float64]:
    """Compute the per-observation loss series (not the mean)."""
    if loss == "qlike":
        if np.any(forecast <= 0):
            raise ValueError("forecast must be strictly positive for QLIKE")
        ratio = (actual ** 2) / (forecast ** 2 + eps)
        return ratio - np.log(ratio + eps) - 1.0
    if loss == "mse":
        return (actual - forecast) ** 2
    if loss == "mae":
        return np.abs(actual - forecast)
    raise ValueError(f"unsupported loss: {loss!r}; choose from {list(LOSS_REGISTRY)}")


def dm_test(
    actual: ArrayLike,
    forecast_a: ArrayLike,
    forecast_b: ArrayLike,
    *,
    loss: LossName = "qlike",
    h: int = 1,
    eps: float = 1e-8,
) -> DMResult:
    """Diebold-Mariano two-sided test on a chosen loss function.

    Args:
        actual: 1-D array of realized (proxy) volatility.
        forecast_a: Forecasts from model A.
        forecast_b: Forecasts from model B.
        loss: One of "qlike", "mse", "mae". Default "qlike".
        h: Forecast horizon in periods. The HAC bandwidth is set to h - 1
           (rule of thumb). For 1-step-ahead, h = 1, bandwidth 0.
        eps: Small constant added to QLIKE denominators for numerical
             stability when forecasts are near zero.

    Returns:
        DMResult with t-statistic, p-value, and winner classification.

    Reference:
        Diebold, F.X. and Mariano, R.S. (1995). "Comparing Predictive
        Accuracy." Journal of Business & Economic Statistics 13(3), 253-263.
    """
    a = np.asarray(actual, dtype=np.float64).ravel()
    pa = np.asarray(forecast_a, dtype=np.float64).ravel()
    pb = np.asarray(forecast_b, dtype=np.float64).ravel()
    if not (a.shape == pa.shape == pb.shape):
        raise ValueError("actual, forecast_a, forecast_b must have the same shape")
    mask = ~(np.isnan(a) | np.isnan(pa) | np.isnan(pb))
    a, pa, pb = a[mask], pa[mask], pb[mask]
    n = a.size
    if n < 10:
        return DMResult(n=n, mean_diff=float("nan"), se=float("nan"), t_stat=float("nan"), p_value=float("nan"), loss=loss, h=h)

    la = _pointwise_loss(a, pa, loss, eps=eps)
    lb = _pointwise_loss(a, pb, loss, eps=eps)
    d = la - lb

    se = newey_west_se(d, q=max(0, h - 1))
    if np.isnan(se) or se == 0:
        return DMResult(n=n, mean_diff=float(d.mean()), se=float("nan"), t_stat=float("nan"), p_value=float("nan"), loss=loss, h=h)

    t = float(d.mean() / se)
    p = float(2.0 * (1.0 - norm.cdf(abs(t))))
    return DMResult(n=n, mean_diff=float(d.mean()), se=se, t_stat=t, p_value=p, loss=loss, h=h)
