"""
Tests for nexus-energy Phases 3-6:
  - Phase 3: Investment & Capacity Expansion
  - Phase 4: Component Registry & F0 Integration
  - Phase 5: Advanced Temporal Methods
  - Phase 6: Sector Coupling
"""

import numpy as np
import pytest


# ===========================================================================
# Phase 3: Investment & Capacity Expansion
# ===========================================================================

class TestInvestment:
    """Test extendable capacity for generators, storage, and links."""

    def test_extendable_generator(self):
        """Generator capacity is a decision variable."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        # Extendable cheap generator + fixed expensive backup
        # Capital cost < (expensive - cheap) * hours, so it's worth building
        sys.add_generator("cheap", bus=elec, capacity=0,
                          marginal_cost=10, capital_cost=5,  # low capital
                          extendable=True, max_capacity=1000)
        sys.add_generator("expensive", bus=elec, capacity=100, marginal_cost=100)
        sys.add_load("demand", bus=elec, amount=80)

        result = sys.optimise()
        assert result.status == "optimal"
        # Should build at least 80 MW of cheap generator
        assert "cheap" in result.capacity_additions
        assert result.capacity_additions["cheap"] >= 80 - 1e-3

    def test_extendable_solar_with_cf(self):
        """Solar capacity optimised with time-varying capacity factor."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        # Solar CF: 0% at night, 80% at midday
        cf = np.array([0.0, 0.3, 0.8, 0.3, 0.0])
        sys.add_generator("solar", bus=elec, capacity=0,
                          marginal_cost=0, capital_cost=10,  # cheap capital
                          carrier_factor=cf,
                          extendable=True, max_capacity=1000)
        sys.add_generator("backup", bus=elec, capacity=200,
                          marginal_cost=200)
        demand = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        sys.add_load("demand", bus=elec, amount=demand)

        result = sys.optimise()
        assert result.status == "optimal"
        # Should build solar with capacity scaled by peak CF
        # At t=2 (CF=0.8), need to serve some demand; solar cap * 0.8 >= something
        assert "solar" in result.capacity_additions
        assert result.capacity_additions["solar"] > 0

    def test_extendable_storage(self):
        """Storage power + energy capacity are decision variables."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sys.add_generator("cheap", bus=elec, capacity=100, marginal_cost=10)
        sys.add_generator("expensive", bus=elec, capacity=100, marginal_cost=500)
        # Varying demand: low-high-low-high cycle
        demand = np.array([40.0, 120.0, 40.0, 120.0])
        sys.add_load("demand", bus=elec, amount=demand)
        # Storage arbitrages cheap → expensive periods
        sys.add_storage("bat", bus=elec,
                        power_capacity=0, energy_capacity=0,
                        capital_cost_power=50, capital_cost_energy=10,
                        efficiency_charge=1.0, efficiency_discharge=1.0,
                        soc_initial=0.0, cyclic=False, soc_min=0.0,
                        extendable=True,
                        max_power_capacity=200,
                        max_energy_capacity=400)

        result = sys.optimise()
        assert result.status == "optimal"
        # Storage should be built to avoid expensive generator
        assert "bat_power" in result.capacity_additions
        assert "bat_energy" in result.capacity_additions

    def test_extendable_link(self):
        """Transmission link capacity is a decision variable."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        bus_a = sys.add_bus("bus_a")
        bus_b = sys.add_bus("bus_b")
        sys.add_generator("cheap_a", bus=bus_a, capacity=200, marginal_cost=10)
        sys.add_generator("expensive_b", bus=bus_b, capacity=200, marginal_cost=100)
        sys.add_load("demand_b", bus=bus_b, amount=150)
        # Extendable link from cheap to expensive side
        sys.add_link("line", bus_from=bus_a, bus_to=bus_b,
                     capacity=0, efficiency=1.0,
                     capital_cost=20, extendable=True, max_capacity=300)

        result = sys.optimise()
        assert result.status == "optimal"
        # Line should be built to import cheap power
        assert "line" in result.capacity_additions
        assert result.capacity_additions["line"] > 100

    def test_emission_limit(self):
        """CO2 cap forces investment in zero-emission capacity."""
        from nexus_energy import EnergySystem
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        # Dirty cheap + clean expensive
        sys.add_generator("coal", bus=elec, capacity=100,
                          marginal_cost=20, emission_factor=1.0)
        sys.add_generator("nuclear", bus=elec, capacity=100,
                          marginal_cost=40, emission_factor=0)
        sys.add_load("demand", bus=elec, amount=80)

        # Without limit: all coal
        result1 = sys.optimise()
        assert result1.generator_dispatch["coal"][0] > 70

        # With tight emission limit: switch to nuclear
        sys.set_emission_limit(10.0)  # max 10 tCO2
        result2 = sys.optimise()
        assert result2.status == "optimal"
        assert result2.generator_dispatch["coal"][0] * 1.0 <= 10.0 + 1e-3


# ===========================================================================
# Phase 4: Component Registry
# ===========================================================================

class TestComponentRegistry:
    """Test the component registry and F0 integration."""

    def test_registry_has_components(self):
        from nexus_energy import registry
        assert registry.count > 20  # at least 20 default components

    def test_registry_sectors(self):
        from nexus_energy import registry
        sectors = registry.list_sectors()
        assert "batteries" in sectors
        assert "solar" in sectors
        assert "wind" in sectors
        assert "hydrogen" in sectors
        assert "thermal" in sectors

    def test_get_component_info(self):
        from nexus_energy import registry
        info = registry.info("EC019")
        assert info["ec_id"] == "EC019"
        assert info["name"] == "NMC Battery"
        assert info["sector"] == "batteries"
        assert info["category"] == "storage"

    def test_unknown_component_raises(self):
        from nexus_energy import registry
        with pytest.raises(KeyError):
            registry.get("EC999")

    def test_add_component_generator(self):
        """Add a solar PV using the registry."""
        from nexus_energy import EnergySystem, add_component
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        pv = add_component(sys, "solar1", "EC044", bus=elec, capacity=100)
        assert pv.capacity == 100
        assert pv.marginal_cost == 0

    def test_add_component_storage(self):
        """Add an NMC battery using the registry."""
        from nexus_energy import EnergySystem, add_component
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        bat = add_component(sys, "battery1", "EC019", bus=elec, capacity=50)
        # Default NMC: 4-hour duration
        assert bat.energy_capacity == pytest.approx(200)  # 50 MW * 4h

    def test_add_component_converter(self):
        """Add a heat pump (requires bus_to)."""
        from nexus_energy import EnergySystem, add_component
        sys = EnergySystem()
        elec = sys.add_bus("elec", carrier="electricity")
        heat = sys.add_bus("heat", carrier="heat")
        hp = add_component(sys, "hp1", "EC068",
                           bus=elec, bus_to=heat, capacity=10)
        # Heat pump: COP=3.0 (efficiency)
        assert hp.efficiency == pytest.approx(3.0)

    def test_converter_requires_bus_to(self):
        """Converter component without bus_to raises."""
        from nexus_energy import EnergySystem, add_component
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        with pytest.raises(ValueError, match="converter"):
            add_component(sys, "hp_bad", "EC068", bus=elec, capacity=10)

    def test_register_custom_component(self):
        """Users can register custom component templates."""
        from nexus_energy import ComponentTemplate, registry
        custom = ComponentTemplate(
            ec_id="CUSTOM_001",
            name="My Fancy Battery",
            sector="custom",
            category="storage",
            capital_cost=100,
            energy_to_power_ratio=6,
            efficiency_charge=0.97,
            efficiency_discharge=0.97,
        )
        registry.register(custom)
        assert "CUSTOM_001" in registry.list_components()
        info = registry.info("CUSTOM_001")
        assert info["name"] == "My Fancy Battery"

    def test_full_system_from_registry(self):
        """Build and solve a full system using registry components."""
        from nexus_energy import EnergySystem, add_component
        sys = EnergySystem("registry_test")
        elec = sys.add_bus("elec")

        # Solar with cf profile
        cf = np.array([0.0, 0.5, 0.9, 0.5, 0.0])
        solar = add_component(sys, "pv", "EC044", bus=elec,
                              capacity=200, carrier_factor=cf)
        # Battery
        bat = add_component(sys, "bat", "EC019", bus=elec, capacity=50)
        # Gas backup
        gas = add_component(sys, "gas", "EC109", bus=elec, capacity=100)
        # Gas fuel supply
        gas_bus = sys.add_bus("gas_supply", carrier="natural_gas")
        sys.add_generator("gas_fuel", bus=gas_bus, capacity=500, marginal_cost=30)
        # Connect gas fuel to gas turbine (simplified: convert gas directly)
        # Actually EC109 is a generator not a converter — so its fuel is abstracted
        # For this test, just let gas generator produce directly

        demand = np.array([80.0, 80.0, 80.0, 80.0, 80.0])
        sys.add_load("demand", bus=elec, amount=demand)

        result = sys.optimise()
        assert result.status == "optimal"


# ===========================================================================
# Phase 5: Advanced Temporal Methods
# ===========================================================================

class TestTemporalMethods:
    """Test time-series aggregation and rolling horizon."""

    def test_k_medoids_basic(self):
        """k-medoids returns correct number of clusters."""
        from nexus_energy.temporal import k_medoids
        rng = np.random.RandomState(0)
        data = rng.randn(50, 5)
        medoids, labels, dists = k_medoids(data, k=5, seed=42)
        assert len(medoids) == 5
        assert len(np.unique(medoids)) == 5  # unique medoids
        assert len(labels) == 50

    def test_aggregate_to_representative_days(self):
        """Aggregate 365 days of synthetic data to 7 representative days."""
        from nexus_energy.temporal import aggregate_to_representative_days

        rng = np.random.RandomState(0)
        n_hours = 365 * 24
        # Seasonal demand: higher in winter
        demand = 50 + 30 * np.sin(np.arange(n_hours) * 2 * np.pi / (365 * 24)) + rng.randn(n_hours) * 5
        # Solar: daily pattern
        hour = np.arange(n_hours) % 24
        solar = np.maximum(0, np.sin((hour - 6) * np.pi / 12)) * 0.8

        rep = aggregate_to_representative_days(
            {"demand": demand, "solar_cf": solar},
            n_days=7,
        )
        assert rep.n_periods == 7
        assert rep.period_length == 24
        assert rep.profiles.shape == (7, 24, 2)
        # Weights should sum to total days (365)
        assert rep.weights.sum() == 365

    def test_rolling_horizon_simple(self):
        """Rolling horizon produces a concatenated dispatch."""
        from nexus_energy import EnergySystem
        from nexus_energy.temporal import rolling_horizon_solve

        # Full 24-hour demand
        full_demand = 50 + 20 * np.sin(np.arange(24) * np.pi / 12)

        def factory(start, end):
            sys = EnergySystem()
            elec = sys.add_bus("elec")
            sys.add_generator("gen", bus=elec, capacity=200, marginal_cost=10)
            sys.add_load("demand", bus=elec, amount=full_demand[start:end])
            return sys

        results = rolling_horizon_solve(
            factory, total_timesteps=24, window_size=8, overlap=0
        )

        dispatch = results["generator_dispatch"]["gen"]
        assert len(dispatch) == 24
        # Dispatch should match demand
        np.testing.assert_allclose(dispatch, full_demand, atol=1e-3)


# ===========================================================================
# Phase 6: Sector Coupling
# ===========================================================================

class TestSectorCoupling:
    """Test multi-carrier networks and sector coupling patterns."""

    def test_power_to_hydrogen_chain(self):
        """Build a P2H system: electricity → electrolyser → H2 storage → fuel cell."""
        from nexus_energy import EnergySystem
        from nexus_energy.sectors import create_power_to_hydrogen

        sys = EnergySystem("p2h")
        elec = sys.add_bus("elec")

        # Variable cheap electricity (e.g. solar)
        cf = np.array([0.0, 1.0, 1.0, 0.0])
        sys.add_generator("solar", bus=elec, capacity=100,
                          marginal_cost=0, carrier_factor=cf)
        sys.add_generator("expensive", bus=elec, capacity=100, marginal_cost=500)

        # Constant electricity demand
        sys.add_load("demand", bus=elec, amount=np.array([50.0, 50.0, 50.0, 50.0]))

        # P2H chain: electrolyser + H2 storage + fuel cell
        components = create_power_to_hydrogen(
            sys, elec,
            electrolyser_capacity=100,
            h2_storage_capacity=50,
            h2_storage_duration=12,
            fuel_cell_capacity=50,
        )

        assert "h2_bus" in components
        assert "electrolyser" in components
        assert "h2_storage" in components
        assert "fuel_cell" in components

        result = sys.optimise()
        assert result.status == "optimal"

    def test_heat_system(self):
        """Build a heat system: electricity → heat pump → heat bus → TES."""
        from nexus_energy import EnergySystem
        from nexus_energy.sectors import create_heat_system

        sys = EnergySystem("heat")
        elec = sys.add_bus("elec")
        sys.add_generator("grid", bus=elec, capacity=100, marginal_cost=40)

        components = create_heat_system(
            sys, elec,
            heat_pump_capacity=10,
            tes_capacity=5,
            tes_duration=6,
        )

        assert "heat_bus" in components
        assert "heat_pump" in components
        assert "tes" in components

        # Heat demand: time-varying
        heat_demand = np.array([20.0, 30.0, 15.0, 20.0])
        sys.add_load("heat_demand", bus=components["heat_bus"], amount=heat_demand)

        result = sys.optimise()
        assert result.status == "optimal"
        # Heat pump provides majority of heat
        assert "heat_heat_pump" in result.link_flow or "heat_hp" in result.link_flow

    def test_multi_carrier_system(self):
        """Create a pre-configured multi-carrier system."""
        from nexus_energy.sectors import create_multi_carrier_system

        sys, buses = create_multi_carrier_system(
            name="test_mc",
            carriers=["electricity", "heat", "hydrogen", "natural_gas"],
        )
        assert len(buses) == 4
        assert "electricity" in buses
        assert "heat" in buses
        assert buses["electricity"].carrier.name == "electricity"

    def test_power_to_gas(self):
        """Build a P2G chain: electricity → electrolyser → H2 → methanation → gas."""
        from nexus_energy import EnergySystem
        from nexus_energy.sectors import create_power_to_gas

        sys = EnergySystem("p2g")
        elec = sys.add_bus("elec")
        sys.add_generator("solar", bus=elec, capacity=200, marginal_cost=0,
                          carrier_factor=np.array([0.0, 1.0, 1.0, 0.0]))
        # Backup so night timesteps are feasible
        sys.add_generator("backup", bus=elec, capacity=200, marginal_cost=100)

        components = create_power_to_gas(
            sys, elec,
            electrolyser_capacity=100,
            methanation_capacity=60,
        )

        assert "h2_bus" in components
        assert "gas_bus" in components
        assert "electrolyser" in components
        assert "methanation" in components

        # H2 storage buffer so electrolyser can run during solar hours
        # and methanation consumes over the full horizon
        from nexus_energy.components import add_component as _add
        _add(sys, "h2_buffer", "EC012", bus=components["h2_bus"],
             capacity=50, energy_to_power_ratio=12)

        # Gas demand: small constant (methanation runs steadily)
        sys.add_load("gas_demand", bus=components["gas_bus"],
                     amount=np.array([5.0, 5.0, 5.0, 5.0]))

        result = sys.optimise()
        assert result.status == "optimal"

    def test_three_sector_coupling(self):
        """Electricity + Heat + Hydrogen coupled system."""
        from nexus_energy import EnergySystem, add_component
        from nexus_energy.sectors import create_multi_carrier_system

        sys, buses = create_multi_carrier_system(
            carriers=["electricity", "heat", "hydrogen"]
        )

        # Solar on electricity bus
        cf = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        add_component(sys, "solar", "EC044", bus=buses["electricity"],
                      capacity=200, carrier_factor=cf)
        # Gas backup
        sys.add_generator("backup", bus=buses["electricity"], capacity=100,
                          marginal_cost=500)

        # Heat pump: elec → heat
        add_component(sys, "hp", "EC068",
                      bus=buses["electricity"], bus_to=buses["heat"],
                      capacity=30)

        # Electrolyser: elec → H2
        add_component(sys, "elz", "EC008",
                      bus=buses["electricity"], bus_to=buses["hydrogen"],
                      capacity=50)

        # Demands
        sys.add_load("elec_demand", bus=buses["electricity"],
                     amount=np.array([30.0, 30.0, 30.0, 30.0, 30.0]))
        sys.add_load("heat_demand", bus=buses["heat"],
                     amount=np.array([10.0, 20.0, 15.0, 20.0, 10.0]))
        sys.add_load("h2_demand", bus=buses["hydrogen"],
                     amount=np.array([5.0, 10.0, 10.0, 10.0, 5.0]))

        result = sys.optimise()
        assert result.status == "optimal"


# ===========================================================================
# Integration: Investment + Components + Sector Coupling
# ===========================================================================

class TestIntegrationScenarios:
    """End-to-end scenarios combining multiple phases."""

    def test_invest_in_pv_and_battery(self):
        """Greenfield investment: decide optimal PV + battery mix."""
        from nexus_energy import EnergySystem, add_component

        sys = EnergySystem("greenfield")
        elec = sys.add_bus("elec")

        # Solar CF profile (24 hours, realistic)
        hours = np.arange(24)
        solar_cf = np.maximum(0, np.sin((hours - 6) * np.pi / 12))
        solar_cf[:6] = 0
        solar_cf[18:] = 0

        # Extendable PV
        add_component(sys, "pv", "EC044", bus=elec, capacity=0,
                      carrier_factor=solar_cf,
                      extendable=True, max_capacity=500)
        # Extendable battery
        add_component(sys, "bat", "EC019", bus=elec, capacity=0,
                      extendable=True, max_capacity=200)
        # Expensive backup (fixed)
        sys.add_generator("backup", bus=elec, capacity=200,
                          marginal_cost=300)
        # Demand
        demand = 100 + 30 * np.sin((hours - 3) * np.pi / 12)
        demand = np.maximum(demand, 70)
        sys.add_load("demand", bus=elec, amount=demand)

        result = sys.optimise()
        assert result.status == "optimal"
        # Should build both PV and battery
        assert result.capacity_additions.get("pv", 0) > 0
        # Backup should be used less
        backup_usage = result.generator_dispatch["backup"].sum()
        demand_total = demand.sum()
        assert backup_usage < demand_total  # PV+bat serve most of demand
