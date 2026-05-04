"""Giacomini-White (2006) Conditional Predictive Ability test.

GW06 extends Diebold-Mariano in two ways:

1. It tests Conditional rather than Unconditional Predictive Ability:
   the null is that the loss differential at time t+h is unpredictable
   given information available at time t.
2. It explicitly accommodates parameter-estimation uncertainty by
   conditioning on the forecasting method (data + estimation procedure)
   rather than on the population model.

The unconditional version is asymptotically equivalent to DM. The
conditional version is strictly more informative when the loss
differential has serial structure: a non-rejection of DM's
unconditional H0 can coexist with a rejection of GW's conditional H0,
and the conditional rejection points to systematic predictability
that DM misses.

Reference:
    Giacomini, R. and White, H. (2006). "Tests of Conditional Predictive
    Ability." Econometrica 74(6), 1545-1578.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from vol_eval.tests.diebold_mariano import _pointwise_loss

InstrumentSpec = Literal["constant", "constant_and_lagged_d"]


@dataclass(frozen=True)
class GWResult:
    """Result of a Giacomini-White (2006) Conditional Predictive Ability test.

    Attributes:
        n: Effective sample size after dropping NaN and applying the
            instrument lag.
        q: Dimension of the test-instrument vector h_t. q=1 corresponds
            to the unconditional test (asymptotically equivalent to DM).
            q=2 is the standard conditional test with h_t = (1, d_{t-1}).
        instruments: Description of the instrument set used.
        statistic: GW test statistic. Asymptotically chi-squared with q
            degrees of freedom under the null of equal conditional
            predictive ability.
        p_value: Right-tail p-value from chi2(q).
        mean_diff: Sample mean of loss(model_a) - loss(model_b). Negative
            means model_a has lower loss.
        loss: Name of the loss function used.
        h: Forecast horizon; sets the HAC bandwidth (q_HAC = h - 1).
        winner: "a", "b", or "tie" classification at 5%.
    """

    n: int
    q: int
    instruments: str
    statistic: float
    p_value: float
    mean_diff: float
    loss: str
    h: int

    @property
    def winner(self) -> str:
        """Return 'a' if model_a wins at 5%, 'b' if model_b wins at 5%, else 'tie'.

        Note that GW06 is a two-sided test in nature (the null is
        joint orthogonality, not a sign hypothesis); the winner
        attribute uses the sign of the unconditional mean differential
        as a tie-breaker once the test rejects.
        """
        if np.isnan(self.statistic) or self.p_value > 0.05:
            return "tie"
        return "a" if self.mean_diff < 0 else "b"

    def __repr__(self) -> str:
        return (
            f"GWResult(loss={self.loss!r}, h={self.h}, n={self.n}, q={self.q}, "
            f"instruments={self.instruments!r}, statistic={self.statistic:.3f}, "
            f"p={self.p_value:.4f}, winner={self.winner!r})"
        )


def _build_instruments(
    d: NDArray[np.float64], spec: InstrumentSpec
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, str]:
    """Construct the test-instrument matrix Z (T_eff x q) for a loss differential d.

    Returns (Z, d_aligned, q, description). `d_aligned` is `d` truncated to
    rows that have valid instrument values (e.g., dropping the first row
    when the instrument includes d_{t-1}).
    """
    if spec == "constant":
        # Unconditional case: instrument is the constant 1.
        z = np.ones((d.size, 1), dtype=np.float64)
        return z, d, 1, "constant"
    if spec == "constant_and_lagged_d":
        # Standard conditional case: h_t = (1, d_{t-1}).
        z = np.column_stack([np.ones(d.size - 1), d[:-1]])
        d_aligned = d[1:]
        return z, d_aligned, 2, "constant + lag-1 d"
    raise ValueError(
        f"unsupported instruments: {spec!r}; choose from "
        f"{['constant', 'constant_and_lagged_d']}"
    )


def _hac_long_run_variance(z_d: NDArray[np.float64], q_hac: int) -> NDArray[np.float64]:
    """Newey-West HAC estimator of the long-run variance of z*d (q-dim vector).

    Uses Bartlett kernel with bandwidth q_hac. Reduces to the sample
    covariance matrix when q_hac == 0.
    """
    n, k = z_d.shape
    if n < 2:
        return np.full((k, k), np.nan)
    centered = z_d - z_d.mean(axis=0, keepdims=True)
    omega = (centered.T @ centered) / n
    for lag in range(1, q_hac + 1):
        weight = 1.0 - lag / (q_hac + 1.0)
        gamma = (centered[lag:].T @ centered[:-lag]) / n
        omega = omega + weight * (gamma + gamma.T)
    return omega


def gw_test(
    actual: ArrayLike,
    forecast_a: ArrayLike,
    forecast_b: ArrayLike,
    *,
    loss: str = "qlike",
    h: int = 1,
    instruments: InstrumentSpec = "constant_and_lagged_d",
    eps: float = 1e-8,
) -> GWResult:
    """Giacomini-White (2006) Conditional Predictive Ability test.

    The test asks: is the loss differential at time t+h orthogonal to
    a vector of test instruments h_t observable at time t? Under the
    null of equal conditional predictive ability, the GW statistic is
    asymptotically chi-squared with q degrees of freedom, where q is
    the dimension of h_t.

    Args:
        actual: 1-D array of realized (proxy) volatility.
        forecast_a: Forecasts from model A.
        forecast_b: Forecasts from model B.
        loss: Loss function name, "qlike" (default), "mse", or "mae".
        h: Forecast horizon in periods. The HAC bandwidth is q_HAC = h - 1.
        instruments: Test-instrument specification. Default
            "constant_and_lagged_d" (h_t = (1, d_{t-1}); q = 2). Use
            "constant" for the unconditional GW test (q = 1).
        eps: Small constant for QLIKE numerical stability.

    Returns:
        GWResult with statistic, chi-squared p-value, and winner.

    Reference:
        Giacomini, R. and White, H. (2006). Econometrica 74(6), 1545-1578.
    """
    a = np.asarray(actual, dtype=np.float64).ravel()
    pa = np.asarray(forecast_a, dtype=np.float64).ravel()
    pb = np.asarray(forecast_b, dtype=np.float64).ravel()
    if not (a.shape == pa.shape == pb.shape):
        raise ValueError("actual, forecast_a, forecast_b must have the same shape")
    mask = ~(np.isnan(a) | np.isnan(pa) | np.isnan(pb))
    a, pa, pb = a[mask], pa[mask], pb[mask]
    n_raw = a.size
    if n_raw < 10:
        return GWResult(
            n=n_raw, q=0, instruments=instruments, statistic=float("nan"),
            p_value=float("nan"), mean_diff=float("nan"), loss=loss, h=h,
        )

    la = _pointwise_loss(a, pa, loss, eps=eps)
    lb = _pointwise_loss(a, pb, loss, eps=eps)
    d_full = la - lb

    z, d, q, descr = _build_instruments(d_full, instruments)
    n_eff = d.size
    if n_eff < 10:
        return GWResult(
            n=n_eff, q=q, instruments=descr, statistic=float("nan"),
            p_value=float("nan"), mean_diff=float(d_full.mean()), loss=loss, h=h,
        )

    # z*d as a (T x q) array of products
    zd = z * d[:, None]
    # Sample mean of z*d, dim q
    mean_zd = zd.mean(axis=0)
    # HAC long-run variance, q x q
    omega = _hac_long_run_variance(zd, q_hac=max(0, h - 1))
    # GW test statistic: n * mean_zd' * inv(omega) * mean_zd
    # Equivalent to a Wald statistic on the coefficient vector mean_zd
    try:
        omega_inv = np.linalg.inv(omega)
    except np.linalg.LinAlgError:
        return GWResult(
            n=n_eff, q=q, instruments=descr, statistic=float("nan"),
            p_value=float("nan"), mean_diff=float(d_full.mean()), loss=loss, h=h,
        )
    statistic = float(n_eff * mean_zd @ omega_inv @ mean_zd)
    p_value = float(1.0 - chi2.cdf(statistic, df=q))

    return GWResult(
        n=n_eff, q=q, instruments=descr,
        statistic=statistic, p_value=p_value,
        mean_diff=float(d_full.mean()), loss=loss, h=h,
    )
