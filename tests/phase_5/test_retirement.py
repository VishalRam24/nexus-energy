"""Phase 5 — endogenous economic retirement + scheduled retirement.

A brownfield asset costs ``fixed_om`` $/MW/year just to keep around,
even when not dispatched. If it can't earn that back from the energy
market, the LP should retire it. Modelled by letting the cap_var go
below the original nameplate (``min_capacity=0``, ``max_capacity=existing``)
and adding fixed_om × cap_var to the objective.

Scheduled retirement (``retire_at_year``) is a no-op in single-stage
mode but must be honored when multi-stage planning is enabled.
"""

from __future__ import annotations

import pytest

import nexus_energy as ne


class TestEconomicRetirement:
    def test_unprofitable_capacity_retires(self):
        # 100 MW coal plant, $50k/MW/year fixed O&M, marginal cost $40.
        # Demand 30 MW served by a $10/MWh gas plant. Coal earns nothing
        # from the market and the LP should retire all 100 MW to dodge
        # the $5M/year fixed cost.
        sys = ne.EnergySystem("retire")
        sys.set_timesteps(1, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=30.0)
        sys.add_generator("gas", bus=bus, capacity=100, marginal_cost=10)
        sys.add_generator(
            "coal", bus=bus, capacity=100,
            marginal_cost=40, fixed_om=50_000,
            extendable=True, min_capacity=0, max_capacity=100,
        )
        result = sys.optimise()
        assert result.status == "optimal"
        # Coal retires: cap_var → 0.
        assert result.capacity_additions["coal"] == pytest.approx(0.0, abs=1e-3)
        # Cost is just gas: 30 MWh × $10/MWh = $300.
        assert result.total_cost == pytest.approx(300.0, abs=1.0)

    def test_profitable_capacity_stays_online(self):
        # Same coal plant but the gas peaker is so expensive ($1000/MWh)
        # that coal's marginal-cost arbitrage covers the fixed O&M.
        sys = ne.EnergySystem("stay")
        sys.set_timesteps(1, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=80.0)
        sys.add_generator("peaker", bus=bus, capacity=100, marginal_cost=1000)
        sys.add_generator(
            "coal", bus=bus, capacity=100,
            marginal_cost=40, fixed_om=500,
            extendable=True, min_capacity=0, max_capacity=100,
        )
        result = sys.optimise()
        assert result.status == "optimal"
        # Coal must stay (and dispatches 80 MW for the load). Saves
        # 80 MW × ($1000 − $40) = $76,800 by displacing the peaker — far
        # more than the 80 MW × $500 = $40,000 retained cost.
        assert result.capacity_additions["coal"] > 79.0
