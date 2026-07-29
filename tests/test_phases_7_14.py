"""
Tests for nexus-energy Phases 7-14:
  - Phase 7: Decomposition
  - Phase 8: Stochastic & Robust
  - Phase 10: Simulation Mode
  - Phase 11: Diagnostics & Reporting
  - Phase 12: Benchmarking
  - Phase 13: PyPSA Compatibility
  - Phase 14: MPC / Real-Time
"""

import numpy as np
import pytest


# ===========================================================================
# Phase 11: Diagnostics
# ===========================================================================

class TestDiagnostics:
    """Post-processing diagnostics."""

    def test_summary(self):
        from nexus_energy import EnergySystem, add_component
        from nexus_energy.diagnostics import Diagnostics

        sys = EnergySystem("diag_test")
        elec = sys.add_bus("elec")
        cf = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        add_component(sys, "solar", "EC044", bus=elec, capacity=100,
                      carrier_factor=cf)
        sys.add_generator("gas", bus=elec, capacity=100, marginal_cost=50)
        sys.add_load("demand", bus=elec, amount=np.array([40.0]*5))

        result = sys.optimise()
        assert result.status == "optimal"

        diag = Diagnostics(sys, result)
        summary = diag.summary()
        assert "diag_test" in summary
        assert "optimal" in summary

    def test_curtailment_report(self):
        from nexus_energy import EnergySystem, add_component
        from nexus_energy.diagnostics import Diagnostics

        sys = EnergySystem()
        elec = sys.add_bus("elec")
        # Oversized solar → will curtail when demand is low
        cf = np.array([0.5, 1.0, 1.0, 0.5])
        add_component(sys, "solar", "EC044", bus=elec, capacity=1000,
                      carrier_factor=cf)
        # Backup so the system is feasible at low-CF hours
        sys.add_generator("backup", bus=elec, capacity=100, marginal_cost=500)
        sys.add_load("demand", bus=elec, amount=np.array([20.0]*4))

        result = sys.optimise()
        assert result.status == "optimal"
        diag = Diagnostics(sys, result)
        curt = diag.curtailment_report()
        # Solar potential = sum(cf) * 1000 = 3000 MWh, used = ~80 MWh, curtailed huge
        assert "solar" in curt.total_curtailed
        assert curt.total_curtailed["solar"] > 1000

    def test_bottleneck_report(self):
        from nexus_energy import EnergySystem
        from nexus_energy.diagnostics import Diagnostics

        sys = EnergySystem()
        elec = sys.add_bus("elec")
        # Cheap generator at capacity, expensive has slack
        sys.add_generator("cheap", bus=elec, capacity=80, marginal_cost=10)
        sys.add_generator("expensive", bus=elec, capacity=100, marginal_cost=100)
        sys.add_load("demand", bus=elec, amount=100)

        result = sys.optimise()
        diag = Diagnostics(sys, result)
        bn = diag.bottleneck_report()
        # cheap is at capacity
        assert bn.utilisation.get("cheap", 0) >= 0.99

    def test_energy_balance_verification(self):
        from nexus_energy import EnergySystem
        from nexus_energy.diagnostics import Diagnostics

        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sys.add_generator("gen", bus=elec, capacity=100, marginal_cost=10)
        sys.add_load("demand", bus=elec, amount=80)

        result = sys.optimise()
        diag = Diagnostics(sys, result)
        imbalances = diag.verify_energy_balance()
        # Balance should hold at optimum
        assert imbalances["elec"] < 1e-3

    def test_why_infeasible(self):
        from nexus_energy import EnergySystem
        from nexus_energy.diagnostics import Diagnostics

        # Infeasible: demand exceeds capacity
        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sys.add_generator("small", bus=elec, capacity=10, marginal_cost=10)
        sys.add_load("demand", bus=elec, amount=100)

        result = sys.optimise()
        diag = Diagnostics(sys, result)
        explanation = diag.why_infeasible()
        assert "Capacity" in explanation or "infeasible" in explanation.lower()


# ===========================================================================
# Phase 10: Simulation Mode
# ===========================================================================

class TestSimulation:
    """Forward simulation (no optimisation)."""

    def test_merit_order_simulation(self):
        from nexus_energy import EnergySystem
        from nexus_energy.simulation import simulate

        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sys.add_generator("cheap", bus=elec, capacity=50, marginal_cost=10)
        sys.add_generator("expensive", bus=elec, capacity=100, marginal_cost=100)
        sys.add_load("demand", bus=elec, amount=np.array([30.0, 70.0, 100.0, 40.0]))

        result = sys.optimise()
        sim = simulate(sys)

        # Merit order should give same or worse result (never better than optimum)
        assert sim["total_cost"] >= result.total_cost - 1e-6

    def test_simulation_with_storage(self):
        from nexus_energy import EnergySystem, add_component
        from nexus_energy.simulation import simulate

        sys = EnergySystem()
        elec = sys.add_bus("elec")
        cf = np.array([0.0, 1.0, 1.0, 0.0])
        add_component(sys, "solar", "EC044", bus=elec, capacity=100,
                      carrier_factor=cf)
        add_component(sys, "bat", "EC019", bus=elec, capacity=20)
        sys.add_generator("backup", bus=elec, capacity=100, marginal_cost=500)
        sys.add_load("demand", bus=elec, amount=np.array([50.0]*4))

        sim = simulate(sys)
        # Check outputs exist
        assert "solar" in sim["generator_dispatch"]
        assert "bat" in sim["storage_charge"]


# ===========================================================================
# Phase 7: Decomposition
# ===========================================================================

class TestDecomposition:
    """Decomposition & scaling."""

    def test_temporal_decomposition(self):
        from nexus_energy import EnergySystem, add_component
        from nexus_energy.decomposition import temporal_decomposition

        sys = EnergySystem()
        elec = sys.add_bus("elec")
        T = 48
        hours = np.arange(T)
        cf = np.maximum(0, np.sin((hours % 24 - 6) * np.pi / 12))
        add_component(sys, "solar", "EC044", bus=elec, capacity=100,
                      carrier_factor=cf)
        sys.add_generator("gas", bus=elec, capacity=200, marginal_cost=50)
        add_component(sys, "bat", "EC019", bus=elec, capacity=20)
        sys.add_load("demand", bus=elec, amount=50 + 30 * np.sin(hours * np.pi / 12))

        dec_result = temporal_decomposition(sys, window_size=24, overlap=6)
        assert dec_result["status"] == "optimal"
        assert "gas" in dec_result["generator_dispatch"]
        # Should cover all 48 timesteps
        assert len(dec_result["generator_dispatch"]["gas"]) == T

    def test_recommend_decomposition(self):
        from nexus_energy import EnergySystem
        from nexus_energy.decomposition import recommend_decomposition

        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sys.add_generator("g", bus=elec, capacity=100)
        sys.add_load("l", bus=elec, amount=80)

        rec = recommend_decomposition(sys)
        assert isinstance(rec, str)
        assert "Recommendation" in rec


# ===========================================================================
# Phase 8: Stochastic
# ===========================================================================

class TestStochastic:
    """Stochastic & robust optimisation."""

    def test_scenario_generation(self):
        from nexus_energy.stochastic import generate_demand_scenarios

        scenarios = generate_demand_scenarios(100.0, n_scenarios=5, std=0.2)
        assert len(scenarios) == 5
        # Probabilities sum to 1
        assert abs(sum(s.probability for s in scenarios) - 1.0) < 1e-6

    def test_stochastic_expected(self):
        from nexus_energy import EnergySystem
        from nexus_energy.stochastic import solve_stochastic, Scenario

        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sys.add_generator("cheap", bus=elec, capacity=50, marginal_cost=10)
        sys.add_generator("expensive", bus=elec, capacity=200, marginal_cost=100)
        sys.add_load("demand", bus=elec, amount=80)

        scenarios = [
            Scenario("low", 0.3, demand_factor=0.8),
            Scenario("mid", 0.5, demand_factor=1.0),
            Scenario("high", 0.2, demand_factor=1.3),
        ]

        res = solve_stochastic(sys, scenarios, risk_measure="expected")
        assert res.status == "optimal"
        assert "low" in res.scenario_costs
        assert "mid" in res.scenario_costs
        assert "high" in res.scenario_costs
        # Expected cost should be weighted average
        exp = (0.3 * res.scenario_costs["low"]
               + 0.5 * res.scenario_costs["mid"]
               + 0.2 * res.scenario_costs["high"])
        assert abs(res.expected_cost - exp) < 1e-1

    def test_robust_worst_case(self):
        from nexus_energy import EnergySystem
        from nexus_energy.stochastic import solve_robust

        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sys.add_generator("cheap", bus=elec, capacity=150, marginal_cost=10)
        sys.add_load("demand", bus=elec, amount=80)

        res = solve_robust(sys, demand_deviation=0.2, cf_deviation=0.0)
        assert res.status == "optimal"
        # Worst case: demand is 80 * 1.2 = 96
        # cheap is 150 >= 96, feasible
        assert res.worst_case_cost > 0


# ===========================================================================
# Phase 12: Benchmarking
# ===========================================================================

class TestBenchmarking:
    """Benchmark harness."""

    def test_run_small_benchmark(self):
        from nexus_energy.benchmarks import benchmark, build_3bus_island
        res = benchmark("small", "3-bus 24h", lambda: build_3bus_island(T=24))
        assert res.status == "optimal"
        assert res.construction_time_s > 0
        assert res.solve_time_s >= 0

    def test_benchmark_report_format(self):
        from nexus_energy.benchmarks import benchmark, build_3bus_island, print_benchmark_report
        results = [benchmark("b1", "3-bus 24h", lambda: build_3bus_island(T=24))]
        report = print_benchmark_report(results)
        assert "Benchmark" in report
        assert "b1" in report


# ===========================================================================
# Phase 13: PyPSA Compatibility
# ===========================================================================

class TestPypsaCompat:
    """PyPSA import/export. Requires pypsa installed."""

    def test_import_skipped_without_pypsa(self):
        # This is just to ensure the module loads even if pypsa isn't installed.
        from nexus_energy import pypsa_compat
        assert hasattr(pypsa_compat, "from_pypsa")
        assert hasattr(pypsa_compat, "to_pypsa")

    def test_roundtrip_if_pypsa_available(self):
        """Full roundtrip test if pypsa is installed."""
        try:
            import pypsa
        except ImportError:
            pytest.skip("pypsa not installed")

        import pandas as pd
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=4, freq="h"))
        n.add("Bus", "bus_a")
        n.add("Generator", "gen_a", bus="bus_a", p_nom=100, marginal_cost=10)
        n.add("Load", "load_a", bus="bus_a", p_set=50)

        from nexus_energy.pypsa_compat import from_pypsa
        sys = from_pypsa(n)
        assert sys.n_buses == 1
        assert sys.n_components >= 1

        result = sys.optimise()
        assert result.status == "optimal"

    def test_advanced_storage_features_roundtrip(self):
        """Test standing loss, inflow, state of charge pinning mapping and roundtrip."""
        try:
            import pypsa
        except ImportError:
            pytest.skip("pypsa not installed")

        import pandas as pd
        import numpy as np
        
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2024-01-01", periods=4, freq="h"))
        n.add("Bus", "bus_a")
        
        # Add StorageUnit with standing loss and inflow and state_of_charge_set
        n.add("StorageUnit", "sto_unit", bus="bus_a", p_nom=10, max_hours=4, standing_loss=0.01, spill_cost=0.02)
        n.storage_units_t.inflow["sto_unit"] = [1.0, 2.0, 1.5, 0.5]
        n.storage_units_t.state_of_charge_set["sto_unit"] = [np.nan, 5.0, np.nan, 2.0]
        
        # Add Store with standing loss and e_set
        n.add("Store", "store_unit", bus="bus_a", e_nom=50, standing_loss=0.005)
        n.stores_t.e_set["store_unit"] = [10.0, np.nan, 15.0, np.nan]
        
        # Convert to Nexus
        from nexus_energy.pypsa_compat import from_pypsa, to_pypsa
        sys = from_pypsa(n)
        
        # Verify StorageUnit translation
        sto = next(s for s in sys._storages if s.name == "sto_unit")
        assert abs(sto.self_discharge - 0.01) < 1e-6
        assert sto.spill_cost == 0.02
        assert sto.inflow is not None
        assert np.allclose(sto.inflow, [1.0, 2.0, 1.5, 0.5])
        assert sto.soc_fixed == {1: 5.0, 3: 2.0}
        
        # Verify Store translation
        store = next(s for s in sys._storages if s.name == "store_unit")
        assert abs(store.self_discharge - 0.005) < 1e-6
        assert store.storage_model == "store"
        assert store.soc_fixed == {0: 10.0, 2: 15.0}
        
        # Roundtrip to PyPSA
        n_back = to_pypsa(sys)
        assert "sto_unit" in n_back.storage_units.index
        assert "store_unit" in n_back.stores.index


# ===========================================================================
# Phase 14: MPC / Real-Time
# ===========================================================================

class TestMPC:
    """Model Predictive Control."""

    def test_mpc_small(self):
        from nexus_energy import EnergySystem, add_component
        from nexus_energy.mpc import MPCController

        total_hours = 12
        demand_full = np.array([30.0, 35.0, 40.0, 50.0, 60.0, 70.0,
                                 65.0, 55.0, 45.0, 40.0, 35.0, 30.0])

        def factory(start, horizon):
            sys = EnergySystem(f"mpc_t{start}")
            elec = sys.add_bus("elec")
            sys.add_generator("cheap", bus=elec, capacity=50, marginal_cost=10)
            sys.add_generator("expensive", bus=elec, capacity=100, marginal_cost=100)
            add_component(sys, "bat", "EC019", bus=elec, capacity=20)
            end = min(start + horizon, total_hours)
            sys.add_load("demand", bus=elec, amount=demand_full[start:end])
            return sys

        mpc = MPCController(
            system_factory=factory,
            total_steps=total_hours,
            control_horizon=4,
            apply_steps=1,
            verbose=False,
        )
        result = mpc.run()
        assert result["status"] == "optimal"
        assert result["n_resolves"] > 0
        # Should have applied dispatch for all 12 hours
        assert len(result["applied_dispatch"]["cheap"]) == total_hours

    def test_warm_start_resolve(self):
        from nexus_energy import EnergySystem
        from nexus_energy.mpc import warm_start_resolve

        sys = EnergySystem()
        elec = sys.add_bus("elec")
        sys.add_generator("gen", bus=elec, capacity=100, marginal_cost=10)
        sys.add_load("demand", bus=elec, amount=50)

        res1 = sys.optimise()
        # Update demand
        res2 = warm_start_resolve(sys, res1, updates={"demand:demand": 80.0})
        assert res2.status == "optimal"
        # New dispatch should serve 80
        assert abs(res2.generator_dispatch["gen"][0] - 80) < 1e-3
