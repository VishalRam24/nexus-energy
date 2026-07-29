"""Phase 7 — TDR error-bound reporting.

`representative_period_error` returns NRMSE + duration-curve L1 between
the rep-day reconstruction of the input series and the original. With
k = (number of distinct day shapes), error is ~0; with k=1 on a
multi-shape year, error is large.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_energy.temporal import (
    aggregate_to_representative_days,
    representative_period_error,
)


def _two_shape(n_days: int = 14, hpd: int = 24, seed: int = 0):
    hours = np.arange(hpd)
    a = 50 + 50 * np.exp(-((hours - 13) ** 2) / 30.0)
    b = 60 * np.ones(hpd)
    is_b = np.array([(i % 7) in (5, 6) for i in range(n_days)])
    arr = np.where(is_b[:, None], b, a)
    return arr.flatten()


class TestAggregationError:
    def test_two_shape_with_k2_is_zero_error(self):
        load = _two_shape(n_days=14)
        rep = aggregate_to_representative_days({"load": load}, n_days=2)
        err = representative_period_error({"load": load}, rep)
        assert err.overall_nrmse < 1e-9
        assert err.duration_curve_l1["load"] < 1e-9

    def test_k1_on_two_shape_has_nontrivial_error(self):
        load = _two_shape(n_days=14)
        rep = aggregate_to_representative_days({"load": load}, n_days=1)
        err = representative_period_error({"load": load}, rep)
        assert err.overall_nrmse > 0.05
        assert err.duration_curve_l1["load"] > 0.5

    def test_error_decreases_with_more_periods(self):
        rng = np.random.default_rng(0)
        # 30 distinct random day shapes.
        load = rng.normal(100, 20, size=30 * 24)
        e2 = representative_period_error(
            {"load": load},
            aggregate_to_representative_days({"load": load}, n_days=2)).overall_nrmse
        e8 = representative_period_error(
            {"load": load},
            aggregate_to_representative_days({"load": load}, n_days=8)).overall_nrmse
        assert e8 < e2

    def test_per_feature_breakdown(self):
        load = _two_shape(n_days=14)
        rng = np.random.default_rng(1)
        noisy = load + rng.normal(0, 5, size=load.shape)
        rep = aggregate_to_representative_days(
            {"load": load, "wind": noisy}, n_days=2)
        err = representative_period_error(
            {"load": load, "wind": noisy}, rep)
        assert "load" in err.nrmse and "wind" in err.nrmse
        # Noisy series should have higher per-feature error than the clean one.
        assert err.nrmse["wind"] > err.nrmse["load"]
