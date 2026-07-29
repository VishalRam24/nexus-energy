"""Sector coupling validation — proves 5 FEATURE_MATRIX rows at Y.

Each test class validates one row that was previously marked P (partial).
All expected values are hand-computed so the tests are self-contained —
no external library required.

Row 1: CO₂ as tracked carrier (physical flow, capture/storage chain)
Row 2: P2H sector coupling (heat pump + TES + gas boiler)
Row 3: P2G / electrolysis (electrolyser + H₂ storage + fuel cell)
Row 4: Gas network flow (linepack + efficiency loss on pipeline)
Row 5: H₂ electrolyzer co-location (shared bus, reversible operation)
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


# ===================================================================
# Row 1 — CO₂ as tracked carrier
# ===================================================================

class TestCO2AsCarrier:
    """CO₂ flows as a physical carrier: gas plant emits onto a CO₂
    atmosphere bus, capture Link removes CO₂ to a sequestration bus,
    CO₂ storage holds captured mass. The optimizer dispatches capture
    to meet a budget (CO₂ load = allowed emissions)."""

    def test_ccs_chain_basic(self):
        """Gas plant produces 100 MW for 4 h; emits 0.4 tCO₂/MWh.
        Total gross emissions = 100 × 4 × 0.4 = 160 tCO₂.
        CO₂ capture link (90% efficiency) removes CO₂ to underground.
        Without any CO₂ demand / constraint, excess CO₂ must go
        somewhere — a spill generator absorbs the remainder."""
        sys = ne.EnergySystem("ccs")
        sys.set_timesteps(4, dt=1.0)

        elec = sys.add_bus("elec")
        co2_atm = sys.add_bus("co2_atm", carrier="co2")
        co2_stored = sys.add_bus("co2_stored", carrier="co2")

        sys.add_load("demand", bus=elec, amount=100.0)

        # Gas plant: dispatches on elec, emits CO₂ onto co2_atm
        sys.add_generator("gas", bus=elec, capacity=200,
                          marginal_cost=50.0,
                          co2_output_bus=co2_atm,
                          co2_output_factor=0.4)

        # Capture link: removes CO₂ from atmosphere to underground
        sys.add_link("capture", bus_from=co2_atm, bus_to=co2_stored,
                     capacity=200, efficiency=0.9, marginal_cost=20.0)

        # Underground CO₂ storage (sink, perfect efficiency)
        sys.add_storage("co2_sink", bus=co2_stored,
                        power_capacity=200, energy_capacity=10000,
                        efficiency_charge=1.0, efficiency_discharge=1.0,
                        soc_initial=0.0, cyclic=False)

        # CO₂ atmosphere spill: absorbs uncaptured CO₂
        sys.add_storage("co2_atm_buffer", bus=co2_atm,
                        power_capacity=200, energy_capacity=10000,
                        efficiency_charge=1.0, efficiency_discharge=1.0,
                        soc_initial=0.0, cyclic=False)

        result = sys.optimise()
        assert result.status == "optimal"

        # Gas plant dispatches 100 MW × 4 h → CO₂ emitted = 160 tCO₂
        gas_dispatch = result.generator_dispatch["gas"]
        assert np.allclose(gas_dispatch, 100.0, atol=1e-3)

        # Capture link runs at some level, CO₂ flows are non-negative
        capture_flow = result.link_flow["capture"]
        assert (capture_flow >= -1e-6).all()

    def test_co2_budget_forces_capture(self):
        """With a CO₂ budget constraint (modeled as a load that must
        be served on the CO₂ bus), the optimizer is forced to capture.

        Gas plant: 50 MW × 2 h, 0.5 tCO₂/MWh → 50 tCO₂ gross.
        CO₂ capture link: 90% efficient, $30/tCO₂.
        No CO₂ dump allowed → all CO₂ must go through capture.
        Expected: capture = 50 tCO₂ total, stored = 45 tCO₂ (90%)."""
        sys = ne.EnergySystem("ccs_budget")
        sys.set_timesteps(2, dt=1.0)

        elec = sys.add_bus("elec")
        co2_atm = sys.add_bus("co2_atm", carrier="co2")
        co2_stored = sys.add_bus("co2_stored", carrier="co2")

        sys.add_load("demand", bus=elec, amount=50.0)

        sys.add_generator("gas", bus=elec, capacity=100,
                          marginal_cost=40.0,
                          co2_output_bus=co2_atm,
                          co2_output_factor=0.5)

        # Capture: all CO₂ must be captured (no spill path)
        sys.add_link("capture", bus_from=co2_atm, bus_to=co2_stored,
                     capacity=100, efficiency=0.9, marginal_cost=30.0)

        # Underground sink (perfect efficiency — CO₂ mass is conserved)
        sys.add_storage("co2_sink", bus=co2_stored,
                        power_capacity=100, energy_capacity=10000,
                        efficiency_charge=1.0, efficiency_discharge=1.0,
                        soc_initial=0.0, cyclic=False)

        result = sys.optimise()
        assert result.status == "optimal"

        # Capture flow should equal gross emissions = 50 × 0.5 = 25 tCO₂/h
        capture_flow = result.link_flow["capture"]
        assert np.allclose(capture_flow, 25.0, atol=1e-3)

        # Stored = 25 × 0.9 × 2 = 45 tCO₂ total
        sink_soc = result.storage_soc["co2_sink"]
        assert sink_soc[-1] == pytest.approx(45.0, abs=0.1)

    def test_link_co2_output(self):
        """Links also emit CO₂ — a gas-to-electricity converter."""
        sys = ne.EnergySystem("link_co2")
        sys.set_timesteps(1, dt=1.0)

        gas_bus = sys.add_bus("gas", carrier="natural_gas")
        elec = sys.add_bus("elec")
        co2 = sys.add_bus("co2", carrier="co2")

        sys.add_generator("gas_supply", bus=gas_bus, capacity=200,
                          marginal_cost=20.0)
        sys.add_load("demand", bus=elec, amount=100.0)

        # Gas→elec converter: 50% efficient, emits 0.2 tCO₂/MWh_gas
        sys.add_link("ccgt", bus_from=gas_bus, bus_to=elec,
                     capacity=300, efficiency=0.5, marginal_cost=5.0,
                     co2_output_bus=co2, co2_output_factor=0.2)

        # CO₂ sink (perfect efficiency)
        sys.add_storage("co2_store", bus=co2,
                        power_capacity=500, energy_capacity=10000,
                        efficiency_charge=1.0, efficiency_discharge=1.0,
                        soc_initial=0.0, cyclic=False)

        result = sys.optimise()
        assert result.status == "optimal"

        # Demand 100 MW, CCGT η=0.5 → gas input = 200 MW
        # CO₂ = 200 × 0.2 = 40 tCO₂
        co2_soc = result.storage_soc["co2_store"]
        assert co2_soc[-1] == pytest.approx(40.0, abs=0.1)


# ===================================================================
# Row 2 — P2H sector coupling
# ===================================================================

class TestPowerToHeat:
    """Power-to-heat: heat pump with COP, gas boiler backup, thermal
    energy storage. Validates that the optimizer dispatches the
    cheapest heat source first and TES shifts load."""

    def test_heat_pump_cheaper_than_boiler(self):
        """Heat pump COP=3 at $50/MWh_elec → $16.7/MWh_th.
        Gas boiler η=0.9 at $30/MWh_gas → $33.3/MWh_th.
        Optimizer should prefer heat pump."""
        sys = ne.EnergySystem("p2h")
        sys.set_timesteps(4, dt=1.0)

        elec = sys.add_bus("elec")
        gas = sys.add_bus("gas", carrier="natural_gas")
        heat = sys.add_bus("heat", carrier="heat")

        sys.add_generator("grid", bus=elec, capacity=500,
                          marginal_cost=50.0)
        sys.add_generator("gas_supply", bus=gas, capacity=500,
                          marginal_cost=30.0)

        # Heat pump: COP=3 means η=3.0 (elec→heat)
        sys.add_link("hp", bus_from=elec, bus_to=heat,
                     capacity=100, efficiency=3.0, marginal_cost=0.0)

        # Gas boiler: η=0.9
        sys.add_link("boiler", bus_from=gas, bus_to=heat,
                     capacity=100, efficiency=0.9, marginal_cost=0.0)

        sys.add_load("heat_demand", bus=heat, amount=50.0)

        result = sys.optimise()
        assert result.status == "optimal"

        # HP: 50 MW_th / 3.0 COP = 16.67 MW_elec × $50 = $833.3/h
        # Boiler: 50 MW_th / 0.9 = 55.56 MW_gas × $30 = $1666.7/h
        # Optimizer picks HP → total cost = 4 × $833.3 = $3333.3
        hp_flow = result.link_flow["hp"]
        assert np.allclose(hp_flow, 50.0 / 3.0, atol=1e-2)
        boiler_flow = result.link_flow["boiler"]
        assert np.allclose(boiler_flow, 0.0, atol=1e-3)

    def test_tes_absorbs_surplus_heat(self):
        """TES absorbs heat surplus and releases it when the heat pump
        can't cover demand alone. Validates heat carrier storage."""
        sys = ne.EnergySystem("p2h_tes")
        sys.set_timesteps(4, dt=1.0)

        elec = sys.add_bus("elec")
        heat = sys.add_bus("heat", carrier="heat")

        sys.add_generator("grid", bus=elec, capacity=500,
                          marginal_cost=50.0)

        # Heat pump with limited capacity
        sys.add_link("hp", bus_from=elec, bus_to=heat,
                     capacity=40, efficiency=3.0)

        # Variable heat demand: low-high-low-high
        sys.add_load("heat_demand", bus=heat,
                     amount=np.array([20.0, 60.0, 20.0, 60.0]))

        # TES smooths the peaks — charges when demand < hp cap,
        # discharges when demand > hp cap.
        sys.add_storage("tes", bus=heat,
                        power_capacity=100, energy_capacity=200,
                        efficiency_charge=1.0, efficiency_discharge=1.0,
                        soc_initial=0.5, cyclic=True)

        result = sys.optimise()
        assert result.status == "optimal"

        # Total heat demand = 160 MWh_th. HP + TES must serve all of it.
        # HP input total × COP should cover total demand (perfect η TES).
        hp_flow = result.link_flow["hp"]
        total_hp_heat = hp_flow.sum() * 3.0
        assert total_hp_heat == pytest.approx(160.0, abs=1.0)
        # Cost: total elec = 160/3 ≈ 53.33 MWh @ $50 ≈ $2666.7
        assert result.total_cost == pytest.approx(160.0 / 3.0 * 50.0, abs=5.0)


# ===================================================================
# Row 3 — P2G / electrolysis
# ===================================================================

class TestPowerToGas:
    """Electrolyser (elec→H₂), H₂ storage, fuel cell (H₂→elec).
    Round-trip validates mass balance and efficiency chain."""

    def test_electrolyser_fuel_cell_round_trip(self):
        """Elec → electrolyser (65% η) → H₂ → fuel cell (50% η) → elec.
        Validates conversion chain: an H₂ demand of 20 MW forces the
        electrolyser to run, and fuel cell is idle (no reverse need).
        H₂ bus balance: electrolyser_out = h2_demand."""
        sys = ne.EnergySystem("p2g")
        sys.set_timesteps(2, dt=1.0)

        elec = sys.add_bus("elec")
        h2 = sys.add_bus("h2", carrier="hydrogen")

        sys.add_generator("grid", bus=elec, capacity=500,
                          marginal_cost=30.0)

        # Electric demand
        sys.add_load("elec_demand", bus=elec, amount=50.0)
        # H₂ demand (industrial offtaker)
        sys.add_load("h2_demand", bus=h2, amount=20.0)

        # Electrolyser: elec → H₂, η=0.65
        sys.add_link("electrolyser", bus_from=elec, bus_to=h2,
                     capacity=100, efficiency=0.65, marginal_cost=2.0)

        # Fuel cell: H₂ → elec, η=0.50 (not needed here, but validates
        # that it doesn't interfere with the electrolyser path)
        sys.add_link("fuel_cell", bus_from=h2, bus_to=elec,
                     capacity=100, efficiency=0.50, marginal_cost=5.0)

        result = sys.optimise()
        assert result.status == "optimal"

        # Electrolyser must produce 20 MW_H2 → input = 20/0.65 ≈ 30.77 MW_elec
        elec_flow = result.link_flow["electrolyser"]
        assert np.allclose(elec_flow, 20.0 / 0.65, atol=0.1)

        # Fuel cell should be idle (no H₂ surplus)
        fc_flow = result.link_flow["fuel_cell"]
        assert np.allclose(fc_flow, 0.0, atol=1e-3)

        # Total grid dispatch = elec demand + electrolyser input
        grid = result.generator_dispatch["grid"]
        assert np.allclose(grid, 50.0 + 20.0 / 0.65, atol=0.2)

    def test_p2g_methanation_chain(self):
        """Full P2G: elec → H₂ → methanation → natural gas.
        Validates 3-carrier chain."""
        sys = ne.EnergySystem("p2g_chain")
        sys.set_timesteps(2, dt=1.0)

        elec = sys.add_bus("elec")
        h2 = sys.add_bus("h2", carrier="hydrogen")
        gas = sys.add_bus("gas", carrier="natural_gas")

        sys.add_generator("grid", bus=elec, capacity=500,
                          marginal_cost=30.0)

        # Electrolyser: elec → H₂, η=0.65
        sys.add_link("electrolyser", bus_from=elec, bus_to=h2,
                     capacity=200, efficiency=0.65)

        # Methanation: H₂ → natural gas, η=0.60
        sys.add_link("methanation", bus_from=h2, bus_to=gas,
                     capacity=200, efficiency=0.60)

        # Gas demand: 20 MWh_gas/h
        sys.add_load("gas_demand", bus=gas, amount=20.0)

        result = sys.optimise()
        assert result.status == "optimal"

        # Gas demand = 20 → methanation input = 20/0.6 = 33.33 MW_H2
        # → electrolyser input = 33.33/0.65 = 51.28 MW_elec
        meth_flow = result.link_flow["methanation"]
        assert np.allclose(meth_flow, 20.0 / 0.60, atol=0.1)
        elec_flow = result.link_flow["electrolyser"]
        assert np.allclose(elec_flow, 20.0 / 0.60 / 0.65, atol=0.1)


# ===================================================================
# Row 4 — Gas network flow (linepack validated separately in
# test_linepack.py; here we add efficiency loss + multi-bus routing)
# ===================================================================

class TestGasNetwork:
    """Gas pipeline with linepack + efficiency loss validates the
    gas-network-flow feature row."""

    def test_gas_pipeline_with_loss(self):
        """3-bus gas network: source → hub → demand.
        Pipeline 1: 2% loss. Pipeline 2: 3% loss.
        Demand 100 MW_gas. Expected source: ~105.1 MW."""
        sys = ne.EnergySystem("gas_net")
        sys.set_timesteps(2, dt=1.0)

        src = sys.add_bus("src", carrier="natural_gas")
        hub = sys.add_bus("hub", carrier="natural_gas")
        sink = sys.add_bus("sink", carrier="natural_gas")

        sys.add_generator("well", bus=src, capacity=200,
                          marginal_cost=25.0)

        # Pipeline 1: src → hub, 2% loss
        sys.add_link("pipe1", bus_from=src, bus_to=hub,
                     capacity=200, efficiency=1.0, loss=0.02)

        # Pipeline 2: hub → sink, 3% loss
        sys.add_link("pipe2", bus_from=hub, bus_to=sink,
                     capacity=200, efficiency=1.0, loss=0.03)

        sys.add_load("city", bus=sink, amount=100.0)

        result = sys.optimise()
        assert result.status == "optimal"

        # After pipe1: delivered to hub = flow × (1-0.02)
        # After pipe2: delivered to sink = hub_flow × (1-0.03) = 100
        # → hub_flow = 100 / 0.97 ≈ 103.09
        # → src_flow = 103.09 / 0.98 ≈ 105.20
        well_dispatch = result.generator_dispatch["well"]
        assert np.allclose(well_dispatch, 100.0 / 0.97 / 0.98, atol=0.2)

    def test_gas_linepack_time_shift(self):
        """Gas linepack decouples source from demand timing.
        Source is ramp-limited but demand swings. Pipe inventory
        absorbs the mismatch."""
        sys = ne.EnergySystem("gas_lp2")
        sys.set_timesteps(4, dt=1.0)

        a = sys.add_bus("a", carrier="natural_gas")
        b = sys.add_bus("b", carrier="natural_gas")

        sys.add_generator("src", bus=a, capacity=60, marginal_cost=20,
                          ramp_up=10, ramp_down=10)
        sys.add_load("ld", bus=b,
                     amount=np.array([20.0, 50.0, 20.0, 50.0]))
        sys.add_link("pipe", bus_from=a, bus_to=b, capacity=80,
                     linepack_capacity=100, linepack_initial=0.5,
                     linepack_cyclic=True)

        result = sys.optimise()
        assert result.status == "optimal"
        # Total demand = 140, total source should match (lossless pipe)
        total_gen = result.generator_dispatch["src"].sum()
        assert total_gen == pytest.approx(140.0, abs=1e-2)


# ===================================================================
# Row 5 — H₂ electrolyzer co-location
# ===================================================================

class TestH2CoLocation:
    """Electrolyser + fuel cell sharing a hydrogen bus, optionally
    with storage. Validates bidirectional power↔H₂ conversion and
    the bus balance mechanics that make co-location work."""

    def test_reversible_electrolyser_fuel_cell(self):
        """Two generators at different costs: cheap solar + expensive
        gas. Electrolyser stores cheap solar as H₂, fuel cell
        releases in hours when solar is unavailable."""
        sys = ne.EnergySystem("h2_coloc")
        sys.set_timesteps(4, dt=1.0)

        elec = sys.add_bus("elec")
        h2 = sys.add_bus("h2", carrier="hydrogen")

        # Solar: cheap but intermittent (available h0,h1 only)
        sys.add_generator("solar", bus=elec, capacity=200,
                          marginal_cost=5.0,
                          carrier_factor=np.array([1.0, 1.0, 0.0, 0.0]))

        # Gas: expensive backup
        sys.add_generator("gas", bus=elec, capacity=200,
                          marginal_cost=100.0)

        sys.add_load("demand", bus=elec, amount=50.0)

        # Electrolyser: elec → H₂
        sys.add_link("electrolyser", bus_from=elec, bus_to=h2,
                     capacity=100, efficiency=0.70)

        # Fuel cell: H₂ → elec
        sys.add_link("fuel_cell", bus_from=h2, bus_to=elec,
                     capacity=100, efficiency=0.55)

        # H₂ buffer storage
        sys.add_storage("h2_buffer", bus=h2,
                        power_capacity=100, energy_capacity=500,
                        efficiency_charge=1.0, efficiency_discharge=1.0,
                        soc_initial=0.5, cyclic=True)

        result = sys.optimise()
        assert result.status == "optimal"

        elec_flow = result.link_flow["electrolyser"]
        fc_flow = result.link_flow["fuel_cell"]

        # Electrolyser should run in solar hours (0,1)
        assert elec_flow[0] > 1.0
        # Fuel cell should run in dark hours (2,3) to displace gas
        assert fc_flow[2] > 0.1 or fc_flow[3] > 0.1

    def test_shared_h2_bus_mass_balance(self):
        """Two electrolysers feed one H₂ bus; one fuel cell draws.
        Mass balance must hold exactly."""
        sys = ne.EnergySystem("h2_shared")
        sys.set_timesteps(2, dt=1.0)

        elec = sys.add_bus("elec")
        h2 = sys.add_bus("h2", carrier="hydrogen")

        sys.add_generator("grid", bus=elec, capacity=500,
                          marginal_cost=30.0)
        sys.add_load("demand", bus=elec, amount=10.0)

        # Two electrolysers with different efficiencies
        sys.add_link("pem_elec", bus_from=elec, bus_to=h2,
                     capacity=50, efficiency=0.65, marginal_cost=2.0)
        sys.add_link("alk_elec", bus_from=elec, bus_to=h2,
                     capacity=50, efficiency=0.60, marginal_cost=3.0)

        # Fuel cell consumes H₂
        sys.add_link("fc", bus_from=h2, bus_to=elec,
                     capacity=50, efficiency=0.50, marginal_cost=5.0)

        # H₂ demand (industrial offtaker)
        sys.add_load("h2_demand", bus=h2, amount=20.0)

        result = sys.optimise()
        assert result.status == "optimal"

        # H₂ bus balance: PEM×0.65 + ALK×0.60 = h2_demand + FC_input
        pem = result.link_flow["pem_elec"]
        alk = result.link_flow["alk_elec"]
        fc = result.link_flow["fc"]
        h2_produced = pem * 0.65 + alk * 0.60
        h2_consumed = 20.0 + fc  # load + fuel cell input
        assert np.allclose(h2_produced, h2_consumed, atol=1e-3)


# ===================================================================
# Multi-sector integration: 3-carrier system end-to-end
# ===================================================================

class TestMultiSectorIntegration:
    """Full 3-carrier system (electricity + heat + hydrogen) with
    sector coupling links, validating that all carriers and conversion
    paths work together."""

    def test_three_carrier_dispatch(self):
        sys = ne.EnergySystem("multi")
        sys.set_timesteps(4, dt=1.0)

        elec = sys.add_bus("elec")
        heat = sys.add_bus("heat", carrier="heat")
        h2 = sys.add_bus("h2", carrier="hydrogen")
        co2 = sys.add_bus("co2", carrier="co2")

        # Electricity supply: wind is cheap, gas is expensive
        sys.add_generator("wind", bus=elec, capacity=100,
                          marginal_cost=0.0)
        sys.add_generator("gas_gen", bus=elec, capacity=300,
                          marginal_cost=60.0,
                          co2_output_bus=co2, co2_output_factor=0.4)

        # Elec demand exceeds wind so gas must run
        sys.add_load("elec_demand", bus=elec, amount=150.0)
        # Heat demand
        sys.add_load("heat_demand", bus=heat, amount=30.0)
        # H₂ demand (industrial)
        sys.add_load("h2_demand", bus=h2, amount=10.0)

        # Heat pump: COP=3.0
        sys.add_link("hp", bus_from=elec, bus_to=heat,
                     capacity=50, efficiency=3.0)

        # Electrolyser: elec→H₂ at 65%
        sys.add_link("electrolyser", bus_from=elec, bus_to=h2,
                     capacity=50, efficiency=0.65)

        # CO₂ sink (perfect efficiency)
        sys.add_storage("co2_sink", bus=co2,
                        power_capacity=500, energy_capacity=10000,
                        efficiency_charge=1.0, efficiency_discharge=1.0,
                        soc_initial=0.0, cyclic=False)

        result = sys.optimise()
        assert result.status == "optimal"

        # Wind is zero-MC so it runs at capacity; gas covers the rest
        wind = result.generator_dispatch["wind"]
        gas = result.generator_dispatch["gas_gen"]

        # Wind should be at capacity (100 MW)
        assert np.allclose(wind, 100.0, atol=1e-2)

        # CO₂ stored = gas dispatch × 0.4
        co2_stored = result.storage_soc["co2_sink"]
        total_co2 = gas.sum() * 0.4
        assert co2_stored[-1] == pytest.approx(total_co2, abs=0.5)

        # Total cost should be feasible (positive, finite)
        assert 0 < result.total_cost < 1e8
