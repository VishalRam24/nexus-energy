"""Phase 6 — 24/7 hourly clean matching (GenX HourlyMatching).

For a designated "data-center" style load, the qualifying clean
generation dispatched into that load's bus must cover its demand at
every timestep, not just on average.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


class TestHourlyMatching:
    def test_hourly_matching_forces_247_clean(self):
        # 3 timesteps, demand [50, 100, 50]. Wind availability [1.0, 0.3, 1.0]
        # so wind cap must be ≥ 100 / 0.3 = 333.3 MW to cover t=1 hour-by-
        # hour. Without 24/7 matching, LP could average wind + gas; with
        # it, wind cap gets sized up.
        sys = ne.EnergySystem("247")
        sys.set_timesteps(3, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=np.array([50.0, 100.0, 50.0]))
        sys.add_generator("gas", bus=bus, capacity=200,
                          marginal_cost=10, tech="gas")
        sys.add_generator("wind", bus=bus, capacity=0,
                          marginal_cost=0, capital_cost=100_000,
                          tech="wind", extendable=True, max_capacity=1000,
                          carrier_factor=np.array([1.0, 0.3, 1.0]))
        sys.set_hourly_matching(load_name="ld", qualifying_techs=["wind"])
        result = sys.optimise()
        assert result.status == "optimal"
        wind_cap = result.capacity_additions["wind"]
        # Must size wind to cover the t=1 shortfall: cap × 0.3 ≥ 100.
        assert wind_cap >= 100.0 / 0.3 - 1e-3
        # Gas dispatches 0 because wind must match hourly on its own.
        gas = result.generator_dispatch["gas"]
        assert all(abs(g) < 1e-3 for g in gas)
