"""Tests for loss functions."""
from __future__ import annotations

import numpy as np
import pytest

from vol_eval import mae, mse, mz_r2, qlike, rmse


def test_qlike_zero_when_perfect():
    a = np.array([0.1, 0.2, 0.15, 0.18])
    assert qlike(a, a) == pytest.approx(0.0, abs=1e-6)


def test_qlike_positive_when_imperfect():
    a = np.array([0.1, 0.2, 0.15, 0.18])
    p = a + 0.05
    assert qlike(a, p) > 0


def test_mse_zero_when_perfect():
    a = np.array([0.1, 0.2, 0.15])
    assert mse(a, a) == 0.0


def test_mse_matches_manual():
    a = np.array([1.0, 2.0, 3.0])
    p = np.array([1.5, 2.5, 2.5])
    expected = np.mean(np.array([0.25, 0.25, 0.25]))
    assert mse(a, p) == pytest.approx(expected)


def test_rmse_is_sqrt_of_mse():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.05, 0.4, 100)
    p = a + rng.normal(0, 0.02, 100)
    assert rmse(a, p) == pytest.approx(np.sqrt(mse(a, p)))


def test_mae_zero_when_perfect():
    a = np.array([0.1, 0.2, 0.15])
    assert mae(a, a) == 0.0


def test_mz_r2_one_when_perfect_linear():
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    p = 2 * a + 0.05  # perfectly linear in a
    assert mz_r2(a, p) == pytest.approx(1.0, abs=1e-9)


def test_mz_r2_zero_when_uncorrelated():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, 1000)
    p = rng.uniform(0, 1, 1000)
    assert mz_r2(a, p) == pytest.approx(0.0, abs=0.05)


def test_qlike_rejects_nonpositive_forecast():
    a = np.array([0.1, 0.2])
    p = np.array([0.1, 0.0])
    with pytest.raises(ValueError):
        qlike(a, p)


def test_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        mse(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_loss_handles_nan():
    a = np.array([0.1, np.nan, 0.2, 0.15])
    p = np.array([0.12, 0.18, np.nan, 0.16])
    # Should compute over the 2 fully-valid pairs (indices 0 and 3)
    assert mse(a, p) == pytest.approx(np.mean([(0.1 - 0.12) ** 2, (0.15 - 0.16) ** 2]))
