"""Significance tests for forecast comparison.

Diebold-Mariano (pairwise), Model Confidence Set (set of survivors from
multiple comparisons), Hansen SPA (one model vs. a set of competitors), and
White's Reality Check (a more conservative ancestor of SPA).
"""
from __future__ import annotations

from vol_eval.tests.diebold_mariano import DMResult, dm_test
from vol_eval.tests.giacomini_white import GWResult, gw_test
from vol_eval.tests.model_confidence_set import MCSResult, model_confidence_set
from vol_eval.tests.reality_check import RealityCheckResult, reality_check
from vol_eval.tests.spa import SPAResult, spa_test

__all__ = [
    "dm_test",
    "DMResult",
    "gw_test",
    "GWResult",
    "model_confidence_set",
    "MCSResult",
    "spa_test",
    "SPAResult",
    "reality_check",
    "RealityCheckResult",
]
