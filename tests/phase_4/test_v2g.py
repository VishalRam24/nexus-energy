"""Phase 4 — V2G mobile storage with availability pattern.

EVs are storage that's only connected to the grid part of the day. We
model this as a per-timestep availability fraction in [0, 1] that
multiplies the (dis)charge capacity. Without it, the LP would let the
fleet feed the grid at 3 a.m. when half of it is at the office.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


class TestV2GAvailability:
    def test_unavailable_window_clamps_dispatch(self):
        sys = ne.EnergySystem("v2g")
        sys.set_timesteps(4, dt=1.0)
        bus = sys.add_bus("e")
        # Gen large enough to serve the load alone (so the LP is feasible
        # even when the EV fleet is parked) but expensive — the LP would
        # rather discharge the EV at every step. Availability=0 must still
        # clamp the EV legs to zero in those steps.
        sys.add_generator("g", bus=bus, capacity=50, marginal_cost=100)
        sys.add_load("ld", bus=bus, amount=[20.0, 20.0, 20.0, 20.0])
        # EV fleet 50 MW power. Available 100% at t=0 / t=3, 0% at t=1 / t=2.
        avail = np.array([1.0, 0.0, 0.0, 1.0])
        sys.add_storage(
            "ev", bus=bus, power_capacity=50, energy_capacity=200,
            soc_initial=0.5, cyclic=False, availability=avail,
        )
        result = sys.optimise()
        assert result.status == "optimal"
        ch = result.storage_charge["ev"]
        dis = result.storage_discharge["ev"]
        # Both legs must be ≈ 0 when availability=0.
        assert ch[1] < 1e-6 and ch[2] < 1e-6
        assert dis[1] < 1e-6 and dis[2] < 1e-6
        # At an available step the EV should actually be discharging
        # (gen at $100, EV at $0) — proves the clamp isn't trivially zero.
        assert dis[0] > 1e-3

    def test_full_availability_matches_default_storage(self):
        # Setting availability=1.0 everywhere should leave behaviour
        # identical to a Storage with no availability series.
        T = 4
        avail = np.ones(T)

        def make(use_avail: bool):
            sys = ne.EnergySystem("eq")
            sys.set_timesteps(T, dt=1.0)
            bus = sys.add_bus("e")
            sys.add_generator("g", bus=bus, capacity=200, marginal_cost=10)
            sys.add_load("ld", bus=bus, amount=[10.0, 10.0, 10.0, 10.0])
            kw = {"availability": avail} if use_avail else {}
            sys.add_storage("s", bus=bus, power_capacity=50,
                            energy_capacity=100, soc_initial=0.5,
                            cyclic=False, **kw)
            return sys.optimise()

        r_default = make(False)
        r_avail   = make(True)
        assert r_default.status == "optimal"
        assert r_avail.status == "optimal"
        assert r_avail.total_cost == pytest.approx(r_default.total_cost, rel=1e-6)
