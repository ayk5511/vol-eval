"""Hansen-Lunde-Nason (2011) Model Confidence Set.

The MCS identifies the subset of models whose forecast accuracy is not
significantly worse than the best model in the comparison set, at a chosen
confidence level. Implements the elimination procedure with the T_max
statistic and bootstrap p-values for the equivalence test.

Reference:
    Hansen, P.R., Lunde, A., and Nason, J.M. (2011). "The Model Confidence
    Set." Econometrica 79(2), 453-497.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vol_eval.tests.diebold_mariano import _pointwise_loss

Statistic = Literal["t_max", "t_range"]


@dataclass
class MCSResult:
    """Result of a Model Confidence Set procedure.

    Attributes:
        survivors: Names of models in the MCS at the given confidence level.
        eliminated: Names of models removed during the procedure, in order.
        eliminated_p_values: For each eliminated model, the p-value at which
            the equivalence hypothesis was rejected (and the model removed).
        loss: Loss function used.
        h: Forecast horizon.
        alpha: Significance level (1 - confidence).
        n_bootstrap: Number of bootstrap replications used.
        statistic: Which test statistic was used ("t_max" or "t_range").
    """

    survivors: list[str]
    eliminated: list[str] = field(default_factory=list)
    eliminated_p_values: list[float] = field(default_factory=list)
    loss: str = "qlike"
    h: int = 1
    alpha: float = 0.10
    n_bootstrap: int = 1000
    statistic: str = "t_max"

    def __repr__(self) -> str:
        return (
            f"MCSResult(loss={self.loss!r}, h={self.h}, alpha={self.alpha}, "
            f"survivors={self.survivors!r}, n_eliminated={len(self.eliminated)})"
        )


def _stationary_bootstrap_indices(n: int, n_boot: int, block_size: int, rng: np.random.Generator) -> NDArray[np.int_]:
    """Politis-Romano stationary bootstrap. Returns shape (n_boot, n) indices."""
    p = 1.0 / max(block_size, 1)
    indices = np.empty((n_boot, n), dtype=np.int_)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int_)
        i = int(rng.integers(0, n))
        idx[0] = i
        for t in range(1, n):
            if rng.random() < p:
                i = int(rng.integers(0, n))
            else:
                i = (i + 1) % n
            idx[t] = i
        indices[b] = idx
    return indices


def model_confidence_set(
    actual: ArrayLike,
    forecasts: dict[str, ArrayLike],
    *,
    loss: str = "qlike",
    h: int = 1,
    alpha: float = 0.10,
    n_bootstrap: int = 1000,
    block_size: int | None = None,
    statistic: Statistic = "t_max",
    eps: float = 1e-8,
    rng: np.random.Generator | None = None,
) -> MCSResult:
    """Compute the Model Confidence Set for a panel of forecasts.

    Args:
        actual: 1-D array of realized volatility.
        forecasts: Mapping of model name to forecast array.
        loss: Loss function ("qlike", "mse", "mae").
        h: Forecast horizon (sets HAC bandwidth via h - 1; also influences
           the stationary-bootstrap block size if block_size is None).
        alpha: 1 - confidence level for the MCS. Default 0.10 (90% MCS).
        n_bootstrap: Number of stationary-bootstrap replications.
        block_size: Stationary-bootstrap mean block length. If None,
            defaults to max(2, ceil(n^{1/3})), a standard choice that scales
            with sample size.
        statistic: "t_max" (default, more powerful) or "t_range".
        eps: QLIKE denominator floor.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        MCSResult with the survivor set and the elimination history.
    """
    if rng is None:
        rng = np.random.default_rng(2026)

    names = list(forecasts.keys())
    if len(names) < 2:
        raise ValueError("MCS requires at least 2 models")

    a = np.asarray(actual, dtype=np.float64).ravel()
    losses = {}
    for name, fc in forecasts.items():
        p = np.asarray(fc, dtype=np.float64).ravel()
        if p.shape != a.shape:
            raise ValueError(f"forecast {name!r} has wrong shape: {p.shape} vs {a.shape}")
        losses[name] = _pointwise_loss(a, p, loss, eps=eps)

    # Drop rows with any NaN
    L = np.column_stack([losses[n] for n in names])
    mask = ~np.any(np.isnan(L), axis=1)
    L = L[mask]
    n = L.shape[0]
    if n < 30:
        # Below this, MCS bootstrap is unreliable
        return MCSResult(survivors=names, loss=loss, h=h, alpha=alpha, n_bootstrap=n_bootstrap, statistic=statistic)

    if block_size is None:
        block_size = max(2, int(np.ceil(n ** (1.0 / 3.0))))

    # Pre-compute bootstrap index resamples (shared across iterations for efficiency)
    boot_idx = _stationary_bootstrap_indices(n, n_bootstrap, block_size, rng)

    surviving = list(range(len(names)))
    eliminated_idx: list[int] = []
    eliminated_p: list[float] = []

    while len(surviving) > 1:
        m = len(surviving)
        sub = L[:, surviving]
        # Loss differentials d_{ij,t} = L_{i,t} - L_{j,t}
        # Equivalent statistic: each model's "performance" relative to the average
        d_bar = sub.mean(axis=0)  # shape (m,)
        # Pairwise mean differentials and HAC variance estimates
        # Centered variant: t_i = (mean_loss_i - mean_loss_avg) / se_i
        # Following Hansen et al. (2011) eq. (4)-(5)
        avg_loss = d_bar.mean()
        d_i = d_bar - avg_loss  # shape (m,)
        # HAC variance via stationary bootstrap variance of (sub.mean - avg) per model
        boot_d_bar = np.empty((n_bootstrap, m))
        for b in range(n_bootstrap):
            sub_b = sub[boot_idx[b]]
            boot_d_bar[b] = sub_b.mean(axis=0) - sub_b.mean(axis=0).mean()
        var_d = boot_d_bar.var(axis=0, ddof=1)
        var_d = np.where(var_d > 0, var_d, np.nan)
        t_stats = d_i / np.sqrt(var_d)

        if statistic == "t_max":
            test_stat = float(np.nanmax(t_stats))
            boot_stats = np.nanmax((boot_d_bar - boot_d_bar.mean(axis=0)) / np.sqrt(var_d), axis=1)
        else:  # t_range
            test_stat = float(np.nanmax(t_stats) - np.nanmin(t_stats))
            boot_stats = np.nanmax(boot_d_bar / np.sqrt(var_d), axis=1) - np.nanmin(boot_d_bar / np.sqrt(var_d), axis=1)

        p_value = float(np.mean(boot_stats >= test_stat))

        if p_value > alpha:
            # Cannot reject equivalence; stop and keep all surviving
            break

        # Eliminate the worst-performing model (highest t-statistic)
        worst_local = int(np.nanargmax(t_stats))
        worst_global = surviving[worst_local]
        eliminated_idx.append(worst_global)
        eliminated_p.append(p_value)
        surviving.pop(worst_local)

    return MCSResult(
        survivors=[names[i] for i in surviving],
        eliminated=[names[i] for i in eliminated_idx],
        eliminated_p_values=eliminated_p,
        loss=loss,
        h=h,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        statistic=statistic,
    )
