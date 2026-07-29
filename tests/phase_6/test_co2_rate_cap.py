"""Phase 6 — CO2 rate cap (GenX CO2Cap: 2 / 3 analogue).

A rate cap bounds emissions *per MWh of energy delivered* rather than
total emissions. Written as ``sum(E × p) ≤ rate × sum(p)`` which is
still linear in the dispatch variables.
"""

from __future__ import annotations

import pytest

import nexus_energy as ne


class TestCO2RateCap:
    def test_rate_cap_forces_cleaner_mix(self):
        # 100 MWh demand. Gas (mc=10, E=0.4 tCO2/MWh) is cheapest; wind
        # (mc=30, E=0) is clean. With a rate cap of 0.2 tCO2/MWh
        # delivered the LP must blend 50/50: wind_MWh * 0 + gas_MWh * 0.4
        # ≤ 0.2 * 100 → gas ≤ 50 MWh.
        sys = ne.EnergySystem("ratecap")
        sys.set_timesteps(1, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=100.0)
        sys.add_generator("gas", bus=bus, capacity=100,
                          marginal_cost=10, emission_factor=0.4)
        sys.add_generator("wind", bus=bus, capacity=100,
                          marginal_cost=30, emission_factor=0.0)
        sys.set_co2_rate_cap(0.2)  # tCO2 per MWh delivered
        result = sys.optimise()
        assert result.status == "optimal"
        gas = float(result.generator_dispatch["gas"][0])
        wind = float(result.generator_dispatch["wind"][0])
        # Gas capped at 50 MWh; wind fills remainder.
        assert gas == pytest.approx(50.0, abs=1e-3)
        assert wind == pytest.approx(50.0, abs=1e-3)
        # Total cost = 50*10 + 50*30 = $2000
        assert result.total_cost == pytest.approx(2000.0, abs=1e-2)

    def test_rate_cap_inactive_when_loose(self):
        # Loose cap (1.0 tCO2/MWh) lets gas serve everything.
        sys = ne.EnergySystem("loose")
        sys.set_timesteps(1, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=100.0)
        sys.add_generator("gas", bus=bus, capacity=100,
                          marginal_cost=10, emission_factor=0.4)
        sys.add_generator("wind", bus=bus, capacity=100,
                          marginal_cost=30, emission_factor=0.0)
        sys.set_co2_rate_cap(1.0)
        result = sys.optimise()
        assert result.status == "optimal"
        assert float(result.generator_dispatch["gas"][0]) == pytest.approx(100.0, abs=1e-3)
