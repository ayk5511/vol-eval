"""White (2000) Reality Check for data snooping.

Tests the null that a benchmark forecast is not outperformed by any of a set
of competitor forecasts. Equivalent to the upper-bound variant of Hansen's
SPA test (more conservative); included separately for users who want the
canonical reference.

Reference:
    White, H. (2000). "A Reality Check for Data Snooping." Econometrica 68(5),
    1097-1126.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from vol_eval.tests.spa import spa_test


@dataclass(frozen=True)
class RealityCheckResult:
    """White (2000) Reality Check result. The Reality Check p-value is
    equivalent to the upper-bound p-value of Hansen's SPA test."""

    n: int
    n_competitors: int
    statistic: float
    p_value: float
    loss: str

    def __repr__(self) -> str:
        return (
            f"RealityCheckResult(loss={self.loss!r}, n={self.n}, k={self.n_competitors}, "
            f"stat={self.statistic:.3f}, p_value={self.p_value:.4f})"
        )


def reality_check(
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
) -> RealityCheckResult:
    """White (2000) Reality Check.

    Returns the upper-bound (Reality Check) p-value from the SPA test. This
    is the most conservative of the three SPA p-values and is the canonical
    Reality Check statistic.
    """
    spa = spa_test(
        actual,
        benchmark,
        competitors,
        loss=loss,
        h=h,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        eps=eps,
        rng=rng,
    )
    return RealityCheckResult(
        n=spa.n,
        n_competitors=spa.n_competitors,
        statistic=spa.statistic,
        p_value=spa.p_value_upper,
        loss=loss,
    )
