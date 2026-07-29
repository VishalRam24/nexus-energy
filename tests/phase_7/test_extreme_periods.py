"""Phase 7 — Extreme-period preservation.

K-medoids picks "typical" days. The peak demand or zero-renewables day
gets averaged into a cluster and disappears, breaking capacity adequacy.
``aggregate_to_representative_days(extreme_periods=...)`` force-includes
those days as standalone representatives with weight=1.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.temporal import (
    aggregate_to_representative_days,
    apply_representative_days,
)


def _year_with_one_peak(n_days: int = 30, hours_per_day: int = 24,
                        peak_day: int = 17, peak_load: float = 500.0):
    """Mostly-flat 100 MW load with a single 500 MW peak on one day."""
    base = 100.0 * np.ones((n_days, hours_per_day))
    base[peak_day, 12] = peak_load  # one ferocious peak hour
    return base.flatten()


class TestExtremePeriods:
    def test_extreme_day_gets_weight_one(self):
        load = _year_with_one_peak(n_days=20, peak_day=11, peak_load=400.0)
        rep = aggregate_to_representative_days(
            {"load": load},
            n_days=4,  # 3 k-medoids + 1 extreme
            extreme_periods=[("max", "load")],
        )
        # The peak day should appear among the medoids with weight 1.
        peak_slot = np.argmax(rep.profiles[:, :, 0].max(axis=1))
        assert rep.weights[peak_slot] == pytest.approx(1.0)

    def test_total_weight_preserved(self):
        load = _year_with_one_peak(n_days=25, peak_day=8, peak_load=350.0)
        rep = aggregate_to_representative_days(
            {"load": load},
            n_days=5,
            extreme_periods=[("max", "load")],
        )
        # Sum of period weights = total number of original days.
        assert rep.weights.sum() == pytest.approx(25.0)

    def test_no_extremes_matches_legacy(self):
        """Calling without extreme_periods preserves prior behaviour."""
        load = _year_with_one_peak(n_days=14, peak_day=3, peak_load=400.0)
        rep = aggregate_to_representative_days({"load": load}, n_days=3)
        assert rep.n_periods == 3
        assert rep.weights.sum() == pytest.approx(14.0)

    def test_capacity_adequacy_with_peak_day(self):
        """The peak day MUST be served. Without extreme inclusion, the
        reduced LP can drop the peak and under-build capacity. With
        ``extreme_periods=[('max', 'load')]`` the peak survives and the
        build covers it."""
        load = _year_with_one_peak(n_days=30, peak_day=13, peak_load=500.0)
        rep = aggregate_to_representative_days(
            {"load": load},
            n_days=4,
            extreme_periods=[("max", "load")],
        )
        sys = ne.EnergySystem("adq")
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=0.0)
        sys.add_generator("gas", bus=bus, capacity=0,
                          marginal_cost=20, capital_cost=1_000,
                          extendable=True, max_capacity=1_000)
        apply_representative_days(sys, rep, {"load": "ld"})

        result = sys.optimise()
        assert result.status == "optimal"
        # Built capacity must clear the 500 MW peak (no LOL allowed).
        gas_cap = result.capacity_additions["gas"]
        assert gas_cap >= 500.0 - 1e-3

    def test_min_extreme_for_renewables(self):
        """For solar / wind, the worst day is the LOWEST output day —
        the day a peaker is most likely to be needed."""
        n_days, hpd = 10, 24
        solar = 0.5 * np.ones((n_days, hpd))
        solar[6, :] = 0.0  # one fully-cloudy day
        rep = aggregate_to_representative_days(
            {"load": np.ones(n_days * hpd) * 100, "solar_cf": solar.flatten()},
            n_days=3,
            extreme_periods=[("min", "solar_cf")],
        )
        # The cloudy day should appear with weight=1.
        solar_min_per_period = rep.profiles[:, :, 1].min(axis=1)
        cloudy_slot = int(np.argmin(solar_min_per_period))
        assert rep.profiles[cloudy_slot, :, 1].max() == pytest.approx(0.0)
        assert rep.weights[cloudy_slot] == pytest.approx(1.0)
