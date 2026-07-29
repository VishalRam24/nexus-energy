"""
Tests for nexus-energy core: data model, dispatch, temporal engine.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Phase 0: Data Model
# ---------------------------------------------------------------------------

class TestDataModel:
    """Test that the data model can be constructed correctly."""

    def test_create_empty_system(self):
        from nexus_energy import EnergySystem
        sys = EnergySystem("test")
        assert sys.name == "test"
        assert sys.n_buses == 0
        assert sys.n_components == 0

    def test_add_bus(self):
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec", carrier="electricity")
        assert elec.name == "elec"
        assert elec.carrier.name == "electricity"
        assert sys.n_buses == 1

    def test_add_custom_carrier(self):
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        sys.add_carrier("ammonia", unit="kg")
        bus = sys.add_bus("nh3_bus", carrier="ammonia")
        assert bus.carrier.name == "ammonia"
        assert bus.carrier.unit == "kg"

    def test_add_generator(self):
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        gen = sys.add_generator("pv", bus=elec, capacity=100, marginal_cost=0)
        assert gen.name == "pv"
        assert gen.capacity == 100
        assert sys.n_components == 1

    def test_add_storage(self):
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sto = sys.add_storage("bat", bus=elec,
                              power_capacity=50, energy_capacity=200)
        assert sto.power_capacity == 50
        assert sto.energy_capacity == 200

    def test_add_load(self):
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        load = sys.add_load("demand", bus=elec, amount=80)
        assert load.amount == 80

    def test_add_link(self):
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        heat = sys.add_bus("heat", carrier="heat")
        link = sys.add_link("hp", bus_from=elec, bus_to=heat,
                            capacity=10, efficiency=3.0)
        assert link.efficiency == 3.0
        assert sys.n_components == 1

    def test_summary(self):
        from nexus_energy import EnergySystem
        sys = EnergySystem("test_summary")
        elec = sys.add_bus("elec")
        sys.add_generator("gen", bus=elec, capacity=100)
        sys.add_load("load", bus=elec, amount=50)
        s = sys.summary()
        assert "test_summary" in s
        assert "Generators: 1" in s
        assert "Loads: 1" in s

    def test_unknown_carrier_raises(self):
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        with pytest.raises(ValueError, match="Unknown carrier"):
            sys.add_bus("bad", carrier="unobtanium")


# ---------------------------------------------------------------------------
# Phase 1: Single-Timestep Dispatch
# ---------------------------------------------------------------------------

class TestSingleTimestepDispatch:
    """Test single-timestep (static) economic dispatch."""

    def test_simple_dispatch(self):
        """Two generators, one load. Cheapest should serve demand."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sys.add_generator("cheap", bus=elec, capacity=100, marginal_cost=10)
        sys.add_generator("expensive", bus=elec, capacity=100, marginal_cost=50)
        sys.add_load("demand", bus=elec, amount=80)

        result = sys.optimise()
        assert result.status == "optimal"
        assert abs(result.generator_dispatch["cheap"][0] - 80) < 1e-4
        assert abs(result.generator_dispatch["expensive"][0]) < 1e-4
        assert abs(result.total_cost - 80 * 10) < 1e-2

    def test_capacity_limited_dispatch(self):
        """Cheap generator can't meet full demand, expensive fills the gap."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sys.add_generator("cheap", bus=elec, capacity=60, marginal_cost=10)
        sys.add_generator("expensive", bus=elec, capacity=100, marginal_cost=50)
        sys.add_load("demand", bus=elec, amount=80)

        result = sys.optimise()
        assert result.status == "optimal"
        assert abs(result.generator_dispatch["cheap"][0] - 60) < 1e-4
        assert abs(result.generator_dispatch["expensive"][0] - 20) < 1e-4
        assert abs(result.total_cost - (60 * 10 + 20 * 50)) < 1e-2

    def test_multi_bus_with_link(self):
        """Two buses connected by a link. Power flows from cheap to expensive side."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        bus_a = sys.add_bus("bus_a")
        bus_b = sys.add_bus("bus_b")
        sys.add_generator("gen_a", bus=bus_a, capacity=100, marginal_cost=10)
        sys.add_generator("gen_b", bus=bus_b, capacity=100, marginal_cost=50)
        sys.add_link("line", bus_from=bus_a, bus_to=bus_b,
                     capacity=50, efficiency=1.0)
        sys.add_load("demand_b", bus=bus_b, amount=80)

        result = sys.optimise()
        assert result.status == "optimal"
        # Line should be at capacity (50 MW), gen_b fills the rest (30 MW)
        assert abs(result.link_flow["line"][0] - 50) < 1e-4
        assert abs(result.generator_dispatch["gen_b"][0] - 30) < 1e-4

    def test_link_with_efficiency(self):
        """Link with losses: 100 MW in → 90 MW out (η=0.9)."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        bus_a = sys.add_bus("bus_a")
        bus_b = sys.add_bus("bus_b")
        sys.add_generator("gen", bus=bus_a, capacity=200, marginal_cost=10)
        sys.add_link("lossy_line", bus_from=bus_a, bus_to=bus_b,
                     capacity=200, efficiency=0.9)
        sys.add_load("demand", bus=bus_b, amount=90)

        result = sys.optimise()
        assert result.status == "optimal"
        # Need 100 MW flow to deliver 90 MW (100 * 0.9 = 90)
        assert abs(result.link_flow["lossy_line"][0] - 100) < 1e-3

    def test_heat_pump_sector_coupling(self):
        """Electricity bus → heat bus via heat pump (COP=3 → efficiency=3.0)."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec", carrier="electricity")
        heat = sys.add_bus("heat", carrier="heat")
        sys.add_generator("grid", bus=elec, capacity=100, marginal_cost=40)
        sys.add_link("hp", bus_from=elec, bus_to=heat,
                     capacity=50, efficiency=3.0)
        sys.add_load("heat_demand", bus=heat, amount=30)

        result = sys.optimise()
        assert result.status == "optimal"
        # 10 MW electricity × COP 3 = 30 MW heat
        assert abs(result.link_flow["hp"][0] - 10) < 1e-3


# ---------------------------------------------------------------------------
# Phase 2: Multi-Timestep Dispatch
# ---------------------------------------------------------------------------

class TestMultiTimestepDispatch:
    """Test temporal dispatch with time-varying demand and generation."""

    def test_timeseries_demand(self):
        """Dispatch follows varying demand over 4 timesteps."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sys.add_generator("gen", bus=elec, capacity=200, marginal_cost=10)
        demand = np.array([50.0, 100.0, 150.0, 80.0])
        sys.add_load("demand", bus=elec, amount=demand)

        result = sys.optimise()
        assert result.status == "optimal"
        dispatch = result.generator_dispatch["gen"]
        assert len(dispatch) == 4
        np.testing.assert_allclose(dispatch, demand, atol=1e-4)

    def test_solar_with_capacity_factor(self):
        """Solar generator with time-varying capacity factor."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        cf = np.array([0.0, 0.3, 0.8, 0.5, 0.0])  # night-day-night
        sys.add_generator("solar", bus=elec, capacity=100,
                          marginal_cost=0, carrier_factor=cf)
        sys.add_generator("gas", bus=elec, capacity=100, marginal_cost=50)
        demand = np.array([40.0, 40.0, 40.0, 40.0, 40.0])
        sys.add_load("demand", bus=elec, amount=demand)

        result = sys.optimise()
        assert result.status == "optimal"
        solar = result.generator_dispatch["solar"]
        gas = result.generator_dispatch["gas"]
        # t=0: solar=0, gas=40
        assert abs(solar[0]) < 1e-4
        assert abs(gas[0] - 40) < 1e-4
        # t=2: solar=40 (capped by demand, not CF), gas=0
        assert abs(solar[2] - 40) < 1e-4
        assert abs(gas[2]) < 1e-4

    def test_storage_arbitrage(self):
        """Storage charges when cheap, discharges when expensive."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")

        # Cheap at t=0,1 and expensive at t=2,3
        sys.add_generator("cheap", bus=elec, capacity=100, marginal_cost=10)
        demand = np.array([50.0, 50.0, 100.0, 100.0])
        sys.add_load("demand", bus=elec, amount=demand)

        # Add expensive backup
        sys.add_generator("backup", bus=elec, capacity=100, marginal_cost=80)

        # Storage should charge during low-demand (t=0,1) and discharge during high (t=2,3)
        sys.add_storage("battery", bus=elec,
                        power_capacity=25, energy_capacity=50,
                        efficiency_charge=1.0, efficiency_discharge=1.0,
                        soc_initial=0.0, cyclic=False, soc_min=0.0)

        result = sys.optimise()
        assert result.status == "optimal"
        charge = result.storage_charge["battery"]
        discharge = result.storage_discharge["battery"]
        # Should charge in early timesteps
        assert charge[0] > 1.0 or charge[1] > 1.0
        # Should discharge in later timesteps
        assert discharge[2] > 1.0 or discharge[3] > 1.0
        # Total cost should be less than without storage
        # (without storage: 50*10+50*10+100*10+100*10 = impossible, needs backup)
        # With storage: charges from cheap, avoids expensive backup

    def test_ramp_constraints(self):
        """Generator with ramp limits can't change output too fast."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sys.add_generator("ramped", bus=elec, capacity=100,
                          marginal_cost=10, ramp_up=30, ramp_down=30)
        sys.add_generator("backup", bus=elec, capacity=100, marginal_cost=100)
        demand = np.array([10.0, 80.0, 20.0])
        sys.add_load("demand", bus=elec, amount=demand)

        result = sys.optimise()
        assert result.status == "optimal"
        p = result.generator_dispatch["ramped"]
        # Ramp from t=0 to t=1: max increase = 30 MW
        assert p[1] - p[0] <= 30 + 1e-4

    def test_24h_dispatch(self):
        """Realistic 24-hour dispatch with solar + gas + demand profile."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")

        # Solar: peaks at noon
        hours = np.arange(24)
        solar_cf = np.maximum(0, np.sin((hours - 6) * np.pi / 12))
        solar_cf[:6] = 0
        solar_cf[18:] = 0
        sys.add_generator("solar", bus=elec, capacity=500,
                          marginal_cost=0, carrier_factor=solar_cf)
        sys.add_generator("gas", bus=elec, capacity=300, marginal_cost=50)

        # Demand: morning and evening peaks
        demand = 100 + 80 * np.sin((hours - 3) * np.pi / 12)
        demand = np.maximum(demand, 50)
        sys.add_load("demand", bus=elec, amount=demand)

        result = sys.optimise()
        assert result.status == "optimal"
        assert result.total_cost > 0
        solar_dispatch = result.generator_dispatch["solar"]
        gas_dispatch = result.generator_dispatch["gas"]
        # Solar should be zero at night
        assert abs(solar_dispatch[0]) < 1e-4
        assert abs(solar_dispatch[23]) < 1e-4
        # Gas fills the gap
        assert gas_dispatch[0] > 10  # night: gas serves demand


class TestStorageCyclic:
    """Test cyclic storage constraint (SOC end == SOC start)."""

    def test_cyclic_enforcement(self):
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sys.add_generator("gen", bus=elec, capacity=200, marginal_cost=10)
        demand = np.array([50.0, 50.0, 50.0, 50.0])
        sys.add_load("demand", bus=elec, amount=demand)
        sys.add_storage("bat", bus=elec,
                        power_capacity=50, energy_capacity=100,
                        soc_initial=0.5, cyclic=True)

        result = sys.optimise()
        assert result.status == "optimal"
        soc = result.storage_soc["bat"]
        # Final SOC should equal initial SOC (0.5 * 100 = 50 MWh)
        assert abs(soc[-1] - 50) < 1e-2


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    """Basic performance sanity checks."""

    def test_100_bus_construction(self):
        """100 buses with generators and loads — should build fast."""
        import time
        from nexus_energy import EnergySystem

        sys = EnergySystem()
        buses = [sys.add_bus(f"bus_{i}") for i in range(100)]
        for i, bus in enumerate(buses):
            sys.add_generator(f"gen_{i}", bus=bus, capacity=100,
                              marginal_cost=10 + i)
            sys.add_load(f"load_{i}", bus=bus, amount=50)

        t0 = time.perf_counter()
        result = sys.optimise()
        elapsed = time.perf_counter() - t0

        assert result.status == "optimal"
        assert elapsed < 5.0, f"100-bus single-timestep took {elapsed:.2f}s"
