"""Phase 4 — heat carrier + thermal storage with self-discharge.

Heat networks need three things on top of the electricity model:

1. A separate carrier (so a heat bus is not satisfied by an electric MW).
2. Conversion links (CHP, heat pump, electric boiler) between
   electricity and heat at well-defined COP / heat ratios.
3. Thermal storage that bleeds energy over time (self-discharge from
   ambient losses) — distinguishes a hot-water tank from a battery.

The first two already work via the carrier system + Link.efficiency
shipped earlier; this file demonstrates the path end-to-end and adds
explicit coverage of the self-discharge term in the SOC equation.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


class TestHeatNetwork:
    def test_heat_pump_serves_heat_load_via_electricity(self):
        # Heat pump COP = 3 → 1 MW electricity → 3 MW heat.
        sys = ne.EnergySystem("hp")
        sys.set_timesteps(2, dt=1.0)
        elec = sys.add_bus("e", carrier="electricity")
        heat = sys.add_bus("h", carrier="heat")
        sys.add_generator("grid", bus=elec, capacity=100, marginal_cost=10)
        sys.add_load("hd", bus=heat, amount=[30.0, 30.0])
        sys.add_link("hp", bus_from=elec, bus_to=heat, capacity=50,
                     efficiency=3.0)
        result = sys.optimise()
        assert result.status == "optimal"
        # Heat delivered = 30 MW × 2 h = 60 MWh; needs 20 MWh electricity.
        # Cost = 20 × $10/MWh = $200.
        assert result.total_cost == pytest.approx(200.0, rel=1e-4)
        # Verify the elec bus saw the 10 MW draw, not 30.
        gen = result.generator_dispatch["grid"]
        assert gen.sum() == pytest.approx(20.0, rel=1e-4)

    def test_carrier_isolation_is_enforced(self):
        # An elec generator on the elec bus should NOT be allowed to serve
        # a heat load directly — without a converting Link the heat bus is
        # infeasible.
        sys = ne.EnergySystem("isolated")
        sys.set_timesteps(1, dt=1.0)
        elec = sys.add_bus("e", carrier="electricity")
        heat = sys.add_bus("h", carrier="heat")
        sys.add_generator("g", bus=elec, capacity=100, marginal_cost=10)
        sys.add_load("hd", bus=heat, amount=10.0)
        result = sys.optimise()
        assert result.status == "infeasible"


class TestThermalStorage:
    def test_self_discharge_bleeds_soc(self):
        # Hot-water tank with 5% loss per timestep, no charging — SOC
        # must decay geometrically.
        sys = ne.EnergySystem("tank")
        sys.set_timesteps(3, dt=1.0)
        heat = sys.add_bus("h", carrier="heat")
        # Need at least one balanced load to keep the LP non-trivial.
        sys.add_generator("aux", bus=heat, capacity=10, marginal_cost=1000)
        sys.add_load("hd", bus=heat, amount=[0.0, 0.0, 0.0])
        sys.add_storage(
            "tank", bus=heat, power_capacity=10, energy_capacity=100,
            soc_initial=1.0, cyclic=False,
            self_discharge=0.05,  # 5% / step
            efficiency_charge=1.0, efficiency_discharge=1.0,
        )
        result = sys.optimise()
        assert result.status == "optimal"
        soc = result.storage_soc["tank"]
        # No load → no discharge optimal. SOC[0] = 100 - 0.05*100 = 95.
        # SOC[1] = 95 - 0.05*95 = 90.25. SOC[2] = 90.25 - 0.05*90.25 = 85.7375.
        assert soc[0] == pytest.approx(95.0, rel=1e-4)
        assert soc[1] == pytest.approx(90.25, rel=1e-4)
        assert soc[2] == pytest.approx(85.7375, rel=1e-4)

    def test_zero_self_discharge_preserves_soc(self):
        # With no self-discharge the tank holds 100 MWh forever.
        sys = ne.EnergySystem("perfect_tank")
        sys.set_timesteps(3, dt=1.0)
        heat = sys.add_bus("h", carrier="heat")
        sys.add_generator("aux", bus=heat, capacity=10, marginal_cost=1000)
        sys.add_load("hd", bus=heat, amount=[0.0, 0.0, 0.0])
        sys.add_storage(
            "tank", bus=heat, power_capacity=10, energy_capacity=100,
            soc_initial=1.0, cyclic=False,
            self_discharge=0.0,
            # Round-trip η=1 avoids the LP-degeneracy artifact where the
            # bus-balance trivially admits ch=dis=X for any X, costing
            # (1−η_c·η_d)·X of SOC. PyPSA hits the same shape.
            efficiency_charge=1.0, efficiency_discharge=1.0,
        )
        result = sys.optimise()
        assert result.status == "optimal"
        assert result.storage_soc["tank"][-1] == pytest.approx(100.0, rel=1e-4)
