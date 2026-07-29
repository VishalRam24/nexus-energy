"""Phase 6 — RPS (Renewable Portfolio Standard) + CES.

RPS: sum(p[qualifying, t]) ≥ fraction × sum(load[t]).
CES: same shape but each tech has a cleanliness score in [0,1].
"""

from __future__ import annotations

import pytest

import nexus_energy as ne


class TestRPS:
    def test_rps_forces_minimum_renewable_share(self):
        sys = ne.EnergySystem("rps")
        sys.set_timesteps(1, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=100.0)
        sys.add_generator("gas", bus=bus, capacity=100,
                          marginal_cost=10, tech="gas")
        sys.add_generator("wind", bus=bus, capacity=100,
                          marginal_cost=30, tech="wind")
        sys.set_rps(fraction=0.6, qualifying_techs=["wind"])
        result = sys.optimise()
        assert result.status == "optimal"
        # Wind must serve ≥ 60 MWh; gas fills remainder.
        wind = float(result.generator_dispatch["wind"][0])
        gas = float(result.generator_dispatch["gas"][0])
        assert wind == pytest.approx(60.0, abs=1e-3)
        assert gas == pytest.approx(40.0, abs=1e-3)


class TestCES:
    def test_ces_weights_clean_fraction(self):
        # Gas (score 0), nuclear (score 1), wind (score 1). 70% clean
        # required. Cheapest mix: 70 MWh wind (cheapest clean) + 30 gas.
        sys = ne.EnergySystem("ces")
        sys.set_timesteps(1, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=100.0)
        sys.add_generator("gas", bus=bus, capacity=100,
                          marginal_cost=10, tech="gas")
        sys.add_generator("wind", bus=bus, capacity=100,
                          marginal_cost=30, tech="wind")
        sys.add_generator("nuclear", bus=bus, capacity=100,
                          marginal_cost=50, tech="nuclear")
        sys.set_ces(fraction=0.7,
                    scores={"wind": 1.0, "nuclear": 1.0, "gas": 0.0})
        result = sys.optimise()
        assert result.status == "optimal"
        wind = float(result.generator_dispatch["wind"][0])
        nuclear = float(result.generator_dispatch["nuclear"][0])
        gas = float(result.generator_dispatch["gas"][0])
        # Clean dispatch ≥ 70 MWh
        assert wind + nuclear >= 70.0 - 1e-3
        # Least-cost mix fills cheap clean first (wind, mc=30)
        assert wind == pytest.approx(70.0, abs=1e-3)
        assert gas == pytest.approx(30.0, abs=1e-3)
        assert nuclear == pytest.approx(0.0, abs=1e-3)
