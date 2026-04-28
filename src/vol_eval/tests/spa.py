"""Hansen (2005) Superior Predictive Ability (SPA) test.

Tests the null that a benchmark forecast is not inferior to any of a set of
competitor forecasts. More powerful than White's Reality Check because it
uses studentized statistics and an estimate of the indicator function for
non-binding moments.

Reference:
    Hansen, P.R. (2005). "A Test for Superior Predictive Ability." Journal of
    Business & Economic Statistics 23(4), 365-380.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from vol_eval.tests.diebold_mariano import _pointwise_loss
from vol_eval.tests.model_confidence_set import _stationary_bootstrap_indices


@dataclass(frozen=True)
class SPAResult:
    """Hansen (2005) SPA test result.

    Per Hansen (2005, p.371), the three p-values satisfy
    p_lower <= p_consistent <= p_upper in the population. The lower variant is
    the liberal one (smallest p-value, easiest to reject); the upper variant is
    the conservative one (largest p-value, equivalent to White's Reality Check).

    Attributes:
        n: Effective sample size.
        n_competitors: Number of competitor models tested against the benchmark.
        statistic: Sample value of the SPA test statistic.
        p_value_consistent: Bootstrap p-value using the consistent estimator
            of the indicator (Hansen 2005, Section 3.3). Recommended for
            inference.
        p_value_lower: Liberal lower-bound estimate. Smallest of the three.
        p_value_upper: Conservative upper-bound estimate. Equals White's
            Reality Check p-value.
        loss: Loss function used.
    """

    n: int
    n_competitors: int
    statistic: float
    p_value_consistent: float
    p_value_lower: float
    p_value_upper: float
    loss: str

    def __repr__(self) -> str:
        return (
            f"SPAResult(loss={self.loss!r}, n={self.n}, k={self.n_competitors}, "
            f"stat={self.statistic:.3f}, p_consistent={self.p_value_consistent:.4f}, "
            f"p_lower={self.p_value_lower:.4f}, p_upper={self.p_value_upper:.4f})"
        )


def spa_test(
    actual: ArrayLike,
    benchmark: ArrayLike,
    competitors: dict[str, ArrayLike],
    *,
    loss: str = "qlike",
    h: int = 1,
    n_bootstrap: int = 1000,
    block_size: int | None = None,
    eps: float = 1e-8,
    rng: np.random.Generator | None = None,
) -> SPAResult:
    """Hansen SPA test: H0 that benchmark is not inferior to any competitor.

    Args:
        actual: Realized volatility.
        benchmark: Benchmark forecast series.
        competitors: Mapping of competitor model name to forecast array.
        loss: Loss function name.
        h: Forecast horizon (sets bandwidth and block size defaults).
        n_bootstrap: Bootstrap replications.
        block_size: Stationary-bootstrap mean block length. If None, defaults
            to max(2, ceil(n^{1/3})).
        eps: QLIKE denominator floor.
        rng: Optional numpy Generator.

    Returns:
        SPAResult with the test statistic and three p-value variants.
    """
    if rng is None:
        rng = np.random.default_rng(2026)

    a = np.asarray(actual, dtype=np.float64).ravel()
    pb = np.asarray(benchmark, dtype=np.float64).ravel()
    comp_names = list(competitors.keys())
    if len(comp_names) < 1:
        raise ValueError("SPA requires at least 1 competitor")
    L_b = _pointwise_loss(a, pb, loss, eps=eps)
    L_c = np.column_stack([_pointwise_loss(a, np.asarray(competitors[n], dtype=np.float64).ravel(), loss, eps=eps) for n in comp_names])

    # Loss differentials d_{k,t} = L_{benchmark,t} - L_{competitor_k,t}
    # Positive d means benchmark is worse; SPA tests H0: max_k E[d_k] <= 0
    D = L_b[:, None] - L_c  # shape (n, k)
    mask = ~np.any(np.isnan(D), axis=1)
    D = D[mask]
    n, k = D.shape
    if n < 30:
        return SPAResult(n=n, n_competitors=k, statistic=float("nan"), p_value_consistent=float("nan"), p_value_lower=float("nan"), p_value_upper=float("nan"), loss=loss)

    if block_size is None:
        block_size = max(2, int(np.ceil(n ** (1.0 / 3.0))))

    d_bar = D.mean(axis=0)  # shape (k,)
    # Bootstrap variance (HAC-equivalent)
    boot_idx = _stationary_bootstrap_indices(n, n_bootstrap, block_size, rng)
    boot_d_bar = np.empty((n_bootstrap, k))
    for b in range(n_bootstrap):
        boot_d_bar[b] = D[boot_idx[b]].mean(axis=0)
    omega = boot_d_bar.var(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.nan)
    # Studentized statistic
    t = d_bar / np.sqrt(omega)
    statistic = float(max(0.0, np.nanmax(t)))

    # Three centering variants for the bootstrap distribution.
    # Following Hansen (2005, eq. 9-11) and the arch package's reference:
    #   Lower (l):       g_l = max(d_bar, 0)  - never recenter models that beat
    #                    benchmark (positive d_bar, where benchmark is worse);
    #                    recenter (zero out) models that lose to benchmark
    #   Consistent (c):  g_c = d_bar where d_bar >= -A_n; 0 otherwise
    #   Upper (u):       g_u = d_bar          - always recenter (Reality Check)
    # where A_n = sqrt(2 * log log n / n) * sqrt(omega_k).
    # Hansen shows g_l <= g_c <= g_u for all k, so p_lower >= p_consistent >= p_upper.
    A_n = np.sqrt(2.0 * np.log(np.log(max(n, 3))) / n) * np.sqrt(omega)
    g_lower = np.maximum(d_bar, 0.0)
    g_consistent = np.where(d_bar >= -A_n, d_bar, 0.0)
    g_upper = d_bar

    centered = boot_d_bar  # shape (n_bootstrap, k)
    boot_t_lower = (centered - g_lower) / np.sqrt(omega)
    boot_t_consistent = (centered - g_consistent) / np.sqrt(omega)
    boot_t_upper = (centered - g_upper) / np.sqrt(omega)

    boot_stats_lower = np.maximum(0.0, np.nanmax(boot_t_lower, axis=1))
    boot_stats_consistent = np.maximum(0.0, np.nanmax(boot_t_consistent, axis=1))
    boot_stats_upper = np.maximum(0.0, np.nanmax(boot_t_upper, axis=1))

    p_lower = float(np.mean(boot_stats_lower >= statistic))
    p_consistent = float(np.mean(boot_stats_consistent >= statistic))
    p_upper = float(np.mean(boot_stats_upper >= statistic))

    return SPAResult(
        n=n,
        n_competitors=k,
        statistic=statistic,
        p_value_consistent=p_consistent,
        p_value_lower=p_lower,
        p_value_upper=p_upper,
        loss=loss,
    )
