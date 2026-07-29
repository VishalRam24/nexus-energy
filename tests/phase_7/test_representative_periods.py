"""Phase 7 — Representative-period API with weighted solve.

`apply_representative_days` should:
  1. Set timesteps = n_periods × period_length.
  2. Replace load / generator profiles with the rep-day profiles.
  3. Install per-timestep snapshot weights (each hour of period p gets
     weight = rep.weights[p], the count of original days mapped to p).

The reduced solve must produce a cost in the same ballpark as a full
8760 solve when the input series has only a few distinct day shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.temporal import (
    aggregate_to_representative_days,
    apply_representative_days,
)


def _two_shape_year(n_days: int = 30, hours_per_day: int = 24,
                    seed: int = 0) -> dict[str, np.ndarray]:
    """Return synthetic load / solar series with exactly TWO distinct
    daily shapes. K-medoids with k=2 should recover both perfectly,
    making the reduced solve == full solve."""
    rng = np.random.default_rng(seed)
    # Shape A: weekday — high midday peak.
    hours = np.arange(hours_per_day)
    weekday_load = 50 + 50 * np.exp(-((hours - 13) ** 2) / 30.0)
    weekday_solar = np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) ** 2
    # Shape B: weekend — flat.
    weekend_load = 60 * np.ones(hours_per_day)
    weekend_solar = 0.5 * weekday_solar
    # 5 weekdays then 2 weekend days, repeat.
    is_weekend = np.array([(i % 7) in (5, 6) for i in range(n_days)])
    load_days = np.where(is_weekend[:, None], weekend_load, weekday_load)
    solar_days = np.where(is_weekend[:, None], weekend_solar, weekday_solar)
    return {
        "load": load_days.flatten(),
        "solar_cf": solar_days.flatten(),
        "_n_weekdays": int((~is_weekend).sum()),
        "_n_weekends": int(is_weekend.sum()),
    }


class TestRepresentativePeriods:
    def test_apply_sets_timesteps_and_weights(self):
        ts = _two_shape_year(n_days=14)
        rep = aggregate_to_representative_days(
            {"load": ts["load"], "solar_cf": ts["solar_cf"]},
            n_days=2,
        )
        sys = ne.EnergySystem("rp")
        bus = sys.add_bus("e")
        sys.add_load("load", bus=bus, amount=0.0)
        sys.add_generator("solar", bus=bus, capacity=200,
                          marginal_cost=0, tech="solar")
        sys.add_generator("gas", bus=bus, capacity=300, marginal_cost=40)

        apply_representative_days(sys, rep, {"load": "load",
                                              "solar_cf": "solar"})

        assert sys._timesteps == 2 * 24
        assert sys._snapshot_weights is not None
        assert len(sys._snapshot_weights) == 48
        # Per-period weight is broadcast across the 24 hours of that period.
        for p in range(rep.n_periods):
            block = sys._snapshot_weights[p * 24:(p + 1) * 24]
            assert np.allclose(block, rep.weights[p])
        # Total weighted hours = original number of days.
        assert sys._snapshot_weights.sum() == pytest.approx(14 * 24)

    def test_two_shape_year_reduces_exactly(self):
        """With only 2 distinct day shapes, k=2 representative days should
        reproduce the full-year cost (within solver tolerance)."""
        ts = _two_shape_year(n_days=14)
        n_weekdays = ts["_n_weekdays"]
        n_weekends = ts["_n_weekends"]
        load_full = ts["load"]
        solar_full = ts["solar_cf"]
        T_full = len(load_full)

        # --- Full 14-day solve.
        sys_full = ne.EnergySystem("full")
        sys_full.set_timesteps(T_full)
        bus = sys_full.add_bus("e")
        sys_full.add_load("load", bus=bus, amount=load_full)
        sys_full.add_generator("solar", bus=bus, capacity=120,
                               marginal_cost=0, carrier_factor=solar_full,
                               tech="solar")
        sys_full.add_generator("gas", bus=bus, capacity=200, marginal_cost=40)
        rfull = sys_full.optimise()
        assert rfull.status == "optimal"

        # --- Reduced 2-day solve via representative-day API.
        rep = aggregate_to_representative_days(
            {"load": load_full, "solar_cf": solar_full},
            n_days=2,
        )
        sys_red = ne.EnergySystem("red")
        bus2 = sys_red.add_bus("e")
        sys_red.add_load("load", bus=bus2, amount=0.0)
        sys_red.add_generator("solar", bus=bus2, capacity=120,
                              marginal_cost=0, tech="solar")
        sys_red.add_generator("gas", bus=bus2, capacity=200, marginal_cost=40)
        apply_representative_days(sys_red, rep,
                                  {"load": "load", "solar_cf": "solar"})
        rred = sys_red.optimise()
        assert rred.status == "optimal"

        # k-medoids should pick one weekday + one weekend (perfectly
        # separable). Weights split as (n_weekdays, n_weekends).
        weight_set = sorted(rep.weights.tolist())
        assert weight_set == sorted([float(n_weekdays), float(n_weekends)])

        # Reduced cost must match full cost (no aggregation error since the
        # input is exactly 2 shapes).
        assert rred.total_cost == pytest.approx(rfull.total_cost, rel=1e-6)
