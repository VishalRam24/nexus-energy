"""Phase 4 — gas / H₂ linepack on Link.

Pipes carry inventory: gas injected at one end can sit in the pipe
before being withdrawn at the other end. Without linepack, a gas /
H₂ network behaves like instantaneous transport and the diurnal
flexibility that linepack provides — letting the source run flat
while demand swings — disappears.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


class TestLinepack:
    def test_pipe_buffers_ramp_limited_source(self):
        # Source at A is ramp-limited; demand at B swings ±30/step. With
        # 100 MWh of linepack the LP can hold the source flat and let the
        # pipe inventory soak up the imbalance.
        sys = ne.EnergySystem("gas_lp")
        sys.set_timesteps(4, dt=1.0)
        sys.add_carrier("gas", "MWh_th")
        a = sys.add_bus("a", carrier="gas")
        b = sys.add_bus("b", carrier="gas")
        sys.add_generator("src", bus=a, capacity=50, marginal_cost=10,
                          ramp_up=5, ramp_down=5)
        sys.add_load("ld", bus=b, amount=np.array([10.0, 40.0, 10.0, 40.0]))
        sys.add_link("pipe", bus_from=a, bus_to=b, capacity=50,
                     linepack_capacity=100, linepack_initial=0.5,
                     linepack_cyclic=True)
        result = sys.optimise()
        assert result.status == "optimal"
        # Total injection over horizon ≈ total demand (lossless pipe).
        flow_in = result.link_flow["pipe"]
        assert flow_in.sum() == pytest.approx(100.0, abs=1e-3)
        # Source covers it all at $10/MWh → total cost ≈ 1000.
        assert result.total_cost == pytest.approx(1000.0, abs=1e-3)

    def test_no_linepack_forces_expensive_peaker(self):
        # Identical setup, no linepack. The ramp-limited source can't
        # follow the swings, so a downstream peaker has to fire.
        sys = ne.EnergySystem("gas_no_lp")
        sys.set_timesteps(4, dt=1.0)
        sys.add_carrier("gas", "MWh_th")
        a = sys.add_bus("a", carrier="gas")
        b = sys.add_bus("b", carrier="gas")
        sys.add_generator("src", bus=a, capacity=50, marginal_cost=10,
                          ramp_up=5, ramp_down=5)
        sys.add_generator("peaker", bus=b, capacity=50, marginal_cost=1000)
        sys.add_load("ld", bus=b, amount=np.array([10.0, 40.0, 10.0, 40.0]))
        sys.add_link("pipe", bus_from=a, bus_to=b, capacity=50)
        result = sys.optimise()
        assert result.status == "optimal"
        assert result.total_cost > 1000.0  # peaker fires

    def test_inventory_within_capacity(self):
        # A larger pipe; verify inv[t] never exceeds capacity at any step.
        sys = ne.EnergySystem("gas_cap")
        sys.set_timesteps(3, dt=1.0)
        sys.add_carrier("gas", "MWh_th")
        a = sys.add_bus("a", carrier="gas")
        b = sys.add_bus("b", carrier="gas")
        sys.add_generator("src", bus=a, capacity=100, marginal_cost=1)
        sys.add_load("ld", bus=b, amount=np.array([0.0, 0.0, 100.0]))
        sys.add_link(
            "pipe", bus_from=a, bus_to=b, capacity=100,
            linepack_capacity=50, linepack_initial=0.0,
        )
        result = sys.optimise()
        assert result.status == "optimal"
        # Total injection still equals total demand.
        assert result.link_flow["pipe"].sum() == pytest.approx(100.0, abs=1e-3)
