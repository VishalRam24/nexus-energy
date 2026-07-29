"""Phase 5 — integer / discrete unit investment.

Real generation comes in discrete unit sizes (one CCGT block ≈ 400 MW,
not 387.4). Modelling continuous capacity expansion overstates how
finely the LP can match demand to install size and understates true
CapEx (it ignores the rounding-up cost). Toggling the cap-var to
integer with a `unit_size` field captures this exactly.
"""

from __future__ import annotations

import pytest

import nexus_energy as ne


class TestIntegerInvestment:
    def test_integer_units_round_up_to_nearest_block(self):
        # Demand 250 MW. CCGT unit_size=100 → must build 3 units (300 MW),
        # not 2.5. With continuous extendable, LP would pick 250 MW exactly.
        sys = ne.EnergySystem("int_inv")
        sys.set_timesteps(1, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=250.0)
        sys.add_generator(
            "ccgt", bus=bus, capacity=0,
            marginal_cost=10, capital_cost=100_000,
            extendable=True, max_capacity=500,
            integer_investment=True, unit_size=100,
        )
        result = sys.optimise()
        assert result.status == "optimal"
        # Built capacity must be a multiple of unit_size.
        built = result.capacity_additions["ccgt"]
        assert built == pytest.approx(300.0, abs=1e-3)

    def test_continuous_default_finds_exact_demand(self):
        # Same problem without integer_investment → builds exactly 250.
        sys = ne.EnergySystem("cont_inv")
        sys.set_timesteps(1, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=250.0)
        sys.add_generator(
            "ccgt", bus=bus, capacity=0,
            marginal_cost=10, capital_cost=100_000,
            extendable=True, max_capacity=500,
        )
        result = sys.optimise()
        assert result.status == "optimal"
        assert result.capacity_additions["ccgt"] == pytest.approx(250.0, abs=1e-3)
