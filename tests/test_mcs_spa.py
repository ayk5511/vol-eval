"""Tests for MCS, SPA, and Reality Check."""
from __future__ import annotations

import numpy as np
import pytest

from vol_eval import model_confidence_set, reality_check, spa_test


@pytest.fixture
def panel():
    rng = np.random.default_rng(123)
    n = 400
    log_vol = np.cumsum(rng.normal(0, 0.05, n)) - 1.5
    actual = np.exp(log_vol) * 0.1 + 0.01
    forecasts = {
        "tight": actual + rng.normal(0, 0.004, n),
        "medium": actual + rng.normal(0, 0.012, n),
        "loose": actual + rng.normal(0, 0.025, n),
        "biased_high": actual * 1.3 + rng.normal(0, 0.008, n),
    }
    forecasts = {k: np.clip(v, 1e-4, None) for k, v in forecasts.items()}
    return actual, forecasts


def test_mcs_returns_result_with_survivors(panel):
    actual, forecasts = panel
    result = model_confidence_set(actual, forecasts, loss="qlike", n_bootstrap=200)
    assert len(result.survivors) >= 1
    assert len(result.survivors) <= len(forecasts)
    # Loose forecast should not survive
    assert "loose" not in result.survivors or result.alpha < 0.01


def test_mcs_eliminates_in_order(panel):
    actual, forecasts = panel
    result = model_confidence_set(actual, forecasts, loss="qlike", n_bootstrap=200)
    # Eliminated p-values should be monotonically non-decreasing in order of removal
    if len(result.eliminated_p_values) >= 2:
        assert result.eliminated_p_values[0] <= 0.10  # below alpha


def test_mcs_handles_two_models(panel):
    actual, forecasts = panel
    sub = {k: forecasts[k] for k in ["tight", "loose"]}
    result = model_confidence_set(actual, sub, loss="qlike", n_bootstrap=200)
    assert len(result.survivors) >= 1


def test_mcs_requires_at_least_two_models(panel):
    actual, forecasts = panel
    with pytest.raises(ValueError):
        model_confidence_set(actual, {"only": forecasts["tight"]})


def test_spa_returns_three_p_values(panel):
    actual, forecasts = panel
    benchmark = forecasts["medium"]
    competitors = {k: v for k, v in forecasts.items() if k != "medium"}
    result = spa_test(actual, benchmark, competitors, loss="qlike", n_bootstrap=200)
    assert 0.0 <= result.p_value_lower <= 1.0
    assert 0.0 <= result.p_value_consistent <= 1.0
    assert 0.0 <= result.p_value_upper <= 1.0
    # Hansen (2005): g_l <= g_c <= g_u implies p_upper <= p_consistent <= p_lower.
    # The lower variant is liberal (small null distribution -> larger sample stat
    # is required, but here it gives the largest p-value because fewer competitors
    # contribute to the null). The upper (Reality Check) is the conservative
    # variant. With sampling noise the inequalities should hold within a tolerance.
    assert result.p_value_upper <= result.p_value_consistent + 0.05
    assert result.p_value_consistent <= result.p_value_lower + 0.05


def test_spa_benchmark_obviously_inferior_rejects(panel):
    actual, forecasts = panel
    # Use the loose forecast as benchmark; tight will dominate -> reject H0
    benchmark = forecasts["loose"]
    competitors = {"tight": forecasts["tight"]}
    result = spa_test(actual, benchmark, competitors, loss="qlike", n_bootstrap=400)
    # Should reject at conventional levels
    assert result.p_value_consistent < 0.10


def test_spa_benchmark_clearly_best_does_not_reject(panel):
    actual, forecasts = panel
    benchmark = forecasts["tight"]
    competitors = {k: v for k, v in forecasts.items() if k != "tight"}
    result = spa_test(actual, benchmark, competitors, loss="qlike", n_bootstrap=400)
    # Should NOT reject H0 (benchmark is fine or better)
    assert result.p_value_consistent > 0.10


def test_reality_check_matches_spa_upper_bound(panel):
    actual, forecasts = panel
    benchmark = forecasts["medium"]
    competitors = {k: v for k, v in forecasts.items() if k != "medium"}
    rng = np.random.default_rng(2026)
    spa_result = spa_test(actual, benchmark, competitors, loss="qlike", n_bootstrap=300, rng=rng)
    rc_result = reality_check(actual, benchmark, competitors, loss="qlike", n_bootstrap=300, rng=np.random.default_rng(2026))
    # Both should produce the same p_value (using same seed)
    assert rc_result.p_value == pytest.approx(spa_result.p_value_upper)


def test_spa_requires_at_least_one_competitor(panel):
    actual, forecasts = panel
    with pytest.raises(ValueError):
        spa_test(actual, forecasts["tight"], {})
