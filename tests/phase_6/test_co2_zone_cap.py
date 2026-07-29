"""Phase 6 — per-zone CO2 mass cap.

PyPSA / GenX let the user bind emissions per zone (bus or bus-group).
Plumbs `set_co2_zone_cap(bus, limit)` that sums emission_factor ×
p[gen_in_zone, t] across the horizon and caps it.
"""

from __future__ import annotations

import pytest

import nexus_energy as ne


class TestCO2ZoneCap:
    def test_zone_cap_binds_only_that_zone(self):
        # Two zones: zone_dirty has cheap gas + a zone cap of 20 tCO2;
        # zone_clean has wind; they're connected by a free link.
        sys = ne.EnergySystem("zonecap")
        sys.set_timesteps(1, dt=1.0)
        dirty = sys.add_bus("dirty")
        clean = sys.add_bus("clean")
        sys.add_generator("gas", bus=dirty, capacity=200,
                          marginal_cost=10, emission_factor=0.4)
        sys.add_generator("wind", bus=clean, capacity=200,
                          marginal_cost=30, emission_factor=0.0)
        sys.add_load("ld", bus=dirty, amount=100.0)
        sys.add_link("tie", bus_from=clean, bus_to=dirty,
                     capacity=200, bidirectional=True)
        sys.set_co2_zone_cap(dirty, limit=20.0)  # ≤ 50 MWh of gas
        result = sys.optimise()
        assert result.status == "optimal"
        gas = float(result.generator_dispatch["gas"][0])
        wind = float(result.generator_dispatch["wind"][0])
        # gas emissions = 0.4 × gas_MWh ≤ 20 → gas ≤ 50.
        assert gas == pytest.approx(50.0, abs=1e-3)
        assert wind == pytest.approx(50.0, abs=1e-3)
