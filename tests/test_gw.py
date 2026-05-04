"""Unit tests for the Giacomini-White (2006) Conditional Predictive Ability test."""
from __future__ import annotations

import numpy as np
import pytest

from vol_eval import gw_test
from vol_eval.tests.giacomini_white import GWResult


def _toy_panel(n: int = 500, seed: int = 2026) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a small synthetic panel where forecast_a is clearly better than forecast_b."""
    rng = np.random.default_rng(seed)
    actual = np.abs(rng.standard_normal(n)) * 0.15 + 0.05
    # forecast_a is close to truth; forecast_b is deliberately badly biased.
    # QLIKE requires positive forecasts; clip both to a strict positive floor.
    forecast_a = np.clip(actual + rng.standard_normal(n) * 0.01, 1e-3, None)
    forecast_b = np.clip(actual + rng.standard_normal(n) * 0.06 + 0.05, 1e-3, None)
    return actual, forecast_a, forecast_b


def test_gw_returns_gwresult() -> None:
    actual, fa, fb = _toy_panel()
    result = gw_test(actual, fa, fb, loss="qlike", h=1)
    assert isinstance(result, GWResult)
    assert result.q == 2  # default conditional case
    assert result.instruments == "constant + lag-1 d"


def test_gw_unconditional_q_is_one() -> None:
    actual, fa, fb = _toy_panel()
    result = gw_test(actual, fa, fb, loss="qlike", h=1, instruments="constant")
    assert result.q == 1
    assert result.instruments == "constant"


def test_gw_rejects_when_a_clearly_better() -> None:
    """When model A is clearly better, the GW test should reject."""
    actual, fa, fb = _toy_panel(n=2000, seed=2026)
    result = gw_test(actual, fa, fb, loss="qlike", h=1)
    assert result.p_value < 0.05
    assert result.mean_diff < 0  # A has lower loss
    assert result.winner == "a"


def test_gw_does_not_reject_when_models_identical() -> None:
    """When forecasts are identical, the GW test should not reject.

    Identical forecasts yield a degenerate (singular) HAC variance
    matrix, so the test gracefully returns NaN instead of a finite
    statistic. The winner classification remains 'tie'.
    """
    rng = np.random.default_rng(2026)
    actual = np.abs(rng.standard_normal(500)) * 0.15 + 0.05
    forecast = np.clip(actual + rng.standard_normal(500) * 0.01, 1e-3, None)
    result = gw_test(actual, forecast, forecast, loss="qlike", h=1)
    # Loss differential is identically zero; HAC variance is singular.
    assert result.mean_diff == 0.0
    assert np.isnan(result.statistic) or result.statistic == 0.0
    assert result.winner == "tie"


def test_gw_handles_short_sample() -> None:
    """The test should return NaN values gracefully on small samples."""
    rng = np.random.default_rng(2026)
    actual = np.abs(rng.standard_normal(5)) * 0.1
    fa = actual + 0.001
    fb = actual + 0.002
    result = gw_test(actual, fa, fb, loss="qlike", h=1)
    assert np.isnan(result.statistic)
    assert np.isnan(result.p_value)


def test_gw_validates_shape() -> None:
    """The test should raise on shape mismatch between inputs."""
    rng = np.random.default_rng(2026)
    a = rng.standard_normal(100)
    b = rng.standard_normal(50)
    with pytest.raises(ValueError, match="same shape"):
        gw_test(a, b, a, loss="mse")


def test_gw_validates_instruments() -> None:
    """The test should raise on an unsupported instruments spec."""
    actual, fa, fb = _toy_panel()
    with pytest.raises(ValueError, match="unsupported instruments"):
        gw_test(actual, fa, fb, loss="qlike", instruments="not_a_spec")  # type: ignore[arg-type]


def test_gw_unconditional_close_to_dm() -> None:
    """GW with q=1 (constant instrument) should give a similar p-value to DM.

    The two are not numerically identical (DM is two-sided normal, GW is
    chi2-with-1-df, and the variance estimators differ slightly), but
    they should agree on whether to reject.
    """
    from vol_eval import dm_test

    actual, fa, fb = _toy_panel(n=1000, seed=2026)
    dm = dm_test(actual, fa, fb, loss="qlike", h=1)
    gw = gw_test(actual, fa, fb, loss="qlike", h=1, instruments="constant")
    # Both should reject at 5% on this strongly-separated panel
    assert dm.p_value < 0.05
    assert gw.p_value < 0.05


def test_gw_repr() -> None:
    actual, fa, fb = _toy_panel()
    result = gw_test(actual, fa, fb, loss="qlike", h=1)
    rep = repr(result)
    assert "GWResult" in rep
    assert "qlike" in rep
    assert "winner" in rep
