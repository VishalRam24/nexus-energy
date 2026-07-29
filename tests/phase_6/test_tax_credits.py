"""Phase 6 — ITC (investment tax credit) + PTC (production tax credit).

- ITC: per-tech fraction of capital_cost reimbursed. Pushes build toward
  qualifying techs without changing dispatch economics.
- PTC: per-tech $/MWh subtracted from marginal cost for every MWh
  dispatched. Pushes dispatch toward qualifying techs.
"""

from __future__ import annotations

import pytest

import nexus_energy as ne


class TestITC:
    def test_itc_reduces_effective_capital_cost(self):
        # Two techs competing for 100 MW demand in a greenfield build.
        # Solar capex=$200k/MW, wind=$150k/MW. Without ITC wind wins
        # on capex. 50% ITC on solar drops its effective capex to
        # $100k → solar wins.
        sys = ne.EnergySystem("itc")
        sys.set_timesteps(1, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=100.0)
        sys.add_generator("solar", bus=bus, capacity=0,
                          marginal_cost=0.0, capital_cost=200_000,
                          tech="solar", extendable=True, max_capacity=200)
        sys.add_generator("wind", bus=bus, capacity=0,
                          marginal_cost=0.0, capital_cost=150_000,
                          tech="wind", extendable=True, max_capacity=200)
        sys.set_itc({"solar": 0.5})  # 50% credit on solar capex
        result = sys.optimise()
        assert result.status == "optimal"
        solar = result.capacity_additions.get("solar", 0.0)
        wind = result.capacity_additions.get("wind", 0.0)
        # Solar effective capex = $100k beats wind $150k → all solar.
        assert solar == pytest.approx(100.0, abs=1e-3)
        assert wind == pytest.approx(0.0, abs=1e-3)
        # Total cost = 100 MW × $100k = $10M.
        assert result.total_cost == pytest.approx(10_000_000.0, rel=1e-4)


class TestPTC:
    def test_ptc_lowers_effective_marginal_cost(self):
        # 100 MWh demand. Gas (mc=10), wind (mc=30). Wind gets $25/MWh
        # PTC → effective wind mc = $5 < gas → all wind.
        sys = ne.EnergySystem("ptc")
        sys.set_timesteps(1, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=100.0)
        sys.add_generator("gas", bus=bus, capacity=100,
                          marginal_cost=10, tech="gas")
        sys.add_generator("wind", bus=bus, capacity=100,
                          marginal_cost=30, tech="wind")
        sys.set_ptc({"wind": 25.0})  # $25/MWh credit on wind
        result = sys.optimise()
        assert result.status == "optimal"
        wind = float(result.generator_dispatch["wind"][0])
        gas = float(result.generator_dispatch["gas"][0])
        assert wind == pytest.approx(100.0, abs=1e-3)
        assert gas == pytest.approx(0.0, abs=1e-3)
        # Total cost = 100 × ($30 − $25) = $500
        assert result.total_cost == pytest.approx(500.0, abs=1e-2)
