"""Phase 5 — economies of scale via piecewise-linear CapEx.

Big plants are cheaper per MW than small plants (engineering,
balance-of-plant, financing). Flat ``capital_cost`` ($/MW) misses
this; PWL CapEx breaks the cap-cost curve into ascending segments
with strictly decreasing $/MW slopes. The LP picks the cheapest
segments first — naturally pushing toward larger builds when demand
justifies it.

Convention: ``capex_segments = [(MW_breakpoint, $/MW), ...]`` sorted
ascending in MW, with slopes strictly decreasing. The cost between
breakpoints i-1 and i is `slope_i * (cap_in_segment)`.
"""

from __future__ import annotations

import pytest

import nexus_energy as ne


class TestPwlCapex:
    def test_decreasing_slope_picks_cheapest_segments_first(self):
        # Two segments: 0-100 MW @ $200k/MW, 100-300 MW @ $100k/MW.
        # Demand=200 MW → built=200 MW. Cost = 100*200k + 100*100k = $30M.
        sys = ne.EnergySystem("pwl_cap")
        sys.set_timesteps(1, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=200.0)
        sys.add_generator(
            "wind", bus=bus, capacity=0,
            marginal_cost=0,
            extendable=True, max_capacity=300,
            capex_segments=[(100, 200_000), (300, 100_000)],
        )
        result = sys.optimise()
        assert result.status == "optimal"
        assert result.capacity_additions["wind"] == pytest.approx(200.0, abs=1e-3)
        # 100 MW × $200k + 100 MW × $100k = $30M
        assert result.total_cost == pytest.approx(30_000_000.0, rel=1e-4)

    def test_flat_capex_unchanged_when_segments_absent(self):
        # No capex_segments → behave like Phase 1 capital_cost.
        sys = ne.EnergySystem("flat_cap")
        sys.set_timesteps(1, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=200.0)
        sys.add_generator(
            "wind", bus=bus, capacity=0,
            marginal_cost=0, capital_cost=150_000,
            extendable=True, max_capacity=300,
        )
        result = sys.optimise()
        assert result.status == "optimal"
        assert result.total_cost == pytest.approx(200 * 150_000, rel=1e-4)
