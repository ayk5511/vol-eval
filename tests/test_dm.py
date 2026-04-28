"""Tests for Diebold-Mariano test."""
from __future__ import annotations

import numpy as np
import pytest

from vol_eval import dm_test


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 500
    # Simulate persistent volatility process
    log_vol = np.cumsum(rng.normal(0, 0.05, n)) - 1.5
    actual = np.exp(log_vol) * 0.1 + 0.01
    # Two forecasters with different noise levels
    forecast_a = actual + rng.normal(0, 0.005, n)  # tighter
    forecast_b = actual + rng.normal(0, 0.015, n)  # noisier
    forecast_a = np.clip(forecast_a, 1e-4, None)
    forecast_b = np.clip(forecast_b, 1e-4, None)
    return actual, forecast_a, forecast_b


def test_dm_returns_result_object(synthetic_data):
    a, fa, fb = synthetic_data
    result = dm_test(a, fa, fb, loss="qlike", h=1)
    assert hasattr(result, "t_stat")
    assert hasattr(result, "p_value")
    assert hasattr(result, "winner")


def test_dm_picks_better_model_with_correct_sign(synthetic_data):
    a, fa, fb = synthetic_data
    # forecast_a is tighter; should win on QLIKE
    result = dm_test(a, fa, fb, loss="qlike", h=1)
    assert result.mean_diff < 0, "Tighter forecast should have lower QLIKE"
    if result.p_value < 0.05:
        assert result.winner == "a"


def test_dm_p_value_in_unit_interval(synthetic_data):
    a, fa, fb = synthetic_data
    result = dm_test(a, fa, fb, loss="qlike")
    assert 0.0 <= result.p_value <= 1.0


def test_dm_identical_forecasts_give_zero_stat():
    rng = np.random.default_rng(0)
    a = np.abs(rng.normal(0.1, 0.02, 200))
    p = np.abs(rng.normal(0.1, 0.02, 200))
    result = dm_test(a, p, p, loss="mse")  # both forecasters are the same series
    assert result.mean_diff == pytest.approx(0.0)
    # t-stat is undefined (0/0); should be NaN
    assert np.isnan(result.t_stat) or result.t_stat == 0.0


def test_dm_different_loss_functions(synthetic_data):
    a, fa, fb = synthetic_data
    for loss in ["qlike", "mse", "mae"]:
        result = dm_test(a, fa, fb, loss=loss)
        assert not np.isnan(result.mean_diff)
        assert 0.0 <= result.p_value <= 1.0


def test_dm_horizon_changes_bandwidth(synthetic_data):
    a, fa, fb = synthetic_data
    r1 = dm_test(a, fa, fb, loss="qlike", h=1)
    r5 = dm_test(a, fa, fb, loss="qlike", h=5)
    # Same point estimate, different SE
    assert r1.mean_diff == pytest.approx(r5.mean_diff)
    assert r1.h == 1 and r5.h == 5


def test_dm_handles_short_sample():
    a = np.array([0.1, 0.12, 0.11])
    fa = np.array([0.1, 0.11, 0.10])
    fb = np.array([0.12, 0.13, 0.12])
    # Below n=10 threshold; returns NaN result
    result = dm_test(a, fa, fb)
    assert np.isnan(result.t_stat)


def test_dm_unsupported_loss_raises():
    a = np.abs(np.random.default_rng(0).normal(0.1, 0.02, 100))
    with pytest.raises(ValueError):
        dm_test(a, a, a, loss="rmse")  # rmse not in DM registry
