"""Phase 6 — Capacity Reserve Margin (planning reserve).

Total derated installed capacity must exceed peak load by a margin:

    sum(firm_cf[tech] × cap[gen]) ≥ (1 + margin) × peak_load

where ``firm_cf`` is the firm capacity credit per technology (solar
~0.15, wind ~0.25, gas ~1.0). Binds on the build decision, not on the
hourly dispatch.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


class TestReserveMargin:
    def test_reserve_margin_forces_firm_overbuild(self):
        # Peak load 100 MW; 15% reserve margin → 115 MW of firm cap.
        # Gas firm credit = 1.0; wind = 0.25. Cheapest mix: 115 MW gas
        # OR mix that satisfies 1.0*gas + 0.25*wind ≥ 115.
        # Wind capex is cheaper so LP wants wind ++ but wind only
        # contributes 0.25 per MW to reserve → need 4× the MW of wind
        # to match a MW of gas. Cost trade-off: gas cap cost $100k vs
        # wind cap cost $50k, but 1 MW gas replaces 4 MW wind (price
        # $200k). So gas cheaper for the margin → 115 MW gas.
        sys = ne.EnergySystem("rm")
        sys.set_timesteps(1, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_load("ld", bus=bus, amount=100.0)
        sys.add_generator("gas", bus=bus, capacity=0,
                          marginal_cost=10, capital_cost=100_000,
                          tech="gas", extendable=True, max_capacity=1000)
        sys.add_generator("wind", bus=bus, capacity=0,
                          marginal_cost=0, capital_cost=50_000,
                          tech="wind", extendable=True, max_capacity=1000)
        sys.set_reserve_margin(margin=0.15,
                                firm_credit={"gas": 1.0, "wind": 0.25})
        result = sys.optimise()
        assert result.status == "optimal"
        gas_cap = result.capacity_additions["gas"]
        wind_cap = result.capacity_additions.get("wind", 0.0)
        # Firm derated sum ≥ 115 MW
        firm = 1.0 * gas_cap + 0.25 * wind_cap
        assert firm >= 115.0 - 1e-3
        # Cheapest: 100 MW gas for dispatch + 15 MW extra gas for margin.
        # Wind is never cheaper since gas capex per firm-MW ($100k) <
        # wind capex per firm-MW ($50k / 0.25 = $200k).
        assert gas_cap == pytest.approx(115.0, abs=1e-3)
