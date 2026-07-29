"""Phase 4 — pumped hydro: independent pump and turbine capacities.

Real PSH plants have asymmetric installed sizes — typically the
turbine is sized to peaker duty (large) and the pump to off-peak fill
duty (smaller). The default Storage uses ``power_capacity`` for both
charge and discharge, which under-models this class of asset.
"""

from __future__ import annotations

import pytest

import nexus_energy as ne


class TestPumpedHydro:
    def test_separate_caps_bind_charge_and_discharge(self):
        sys = ne.EnergySystem("psh")
        sys.set_timesteps(2, dt=1.0)
        bus = sys.add_bus("e")
        # Off-peak generator (cheap) at t=0; on-peak load at t=1.
        sys.add_generator("g", bus=bus, capacity=200, marginal_cost=10)
        sys.add_load("ld", bus=bus, amount=[10.0, 10.0])
        # PSH: turbine 100 MW (to discharge fast at peak), pump 30 MW
        # (slower fill). With pump_capacity=30, charge in t=0 is bounded
        # at 30 MW even though power_capacity says 100.
        sys.add_storage(
            "psh", bus=bus, power_capacity=100, energy_capacity=500,
            soc_initial=0.0, cyclic=False,
            pump_capacity=30, turbine_capacity=100,
        )
        result = sys.optimise()
        assert result.status == "optimal"
        # Charge in any timestep ≤ 30 MW.
        assert result.storage_charge["psh"].max() <= 30.0 + 1e-6
        # Discharge could go up to 100 MW, but with only 10 MW load it
        # won't be tested unless we add a peaker scenario.

    def test_default_storage_has_symmetric_caps(self):
        # Sanity: without pump_capacity / turbine_capacity, behaviour is
        # unchanged from previous phases.
        sys = ne.EnergySystem("sym")
        sys.set_timesteps(2, dt=1.0)
        bus = sys.add_bus("e")
        sys.add_generator("g", bus=bus, capacity=200, marginal_cost=10)
        sys.add_load("ld", bus=bus, amount=[100.0, 100.0])
        sys.add_storage("bat", bus=bus, power_capacity=50, energy_capacity=100,
                        soc_initial=0.5, cyclic=False)
        result = sys.optimise()
        assert result.status == "optimal"
        # Both charge and discharge bounded by the same 50 MW cap.
        assert result.storage_charge["bat"].max() <= 50.0 + 1e-6
        assert result.storage_discharge["bat"].max() <= 50.0 + 1e-6
