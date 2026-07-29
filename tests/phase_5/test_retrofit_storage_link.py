"""Phase 5.1 — retrofit / repower for Storage and Link in MultiStageSystem."""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


def test_storage_retrofit_capped_by_retiring_host():
    """A pumped-hydro 'revamp' storage is a retrofit of a retiring battery.
    Its stage-2040 new-build power is capped by the host's retiring power."""
    mss = ne.MultiStageSystem("st_retro")

    def build(year: int) -> ne.EnergySystem:
        sys = ne.EnergySystem(f"s{year}")
        sys.set_timesteps(2, dt=1.0)
        b = sys.add_bus("e")
        sys.add_load("d", bus=b, amount=np.array([10.0, 0.0]))
        sys.add_generator("g", bus=b, capacity=100.0, marginal_cost=1.0)
        # Host battery: 50 MW / 100 MWh brownfield retiring in 2040.
        sys.add_storage("battery", bus=b, power_capacity=50.0,
                        energy_capacity=100.0, retire_at_year=2040)
        # Revamp storage: cheap retrofit of the battery.
        sys.add_storage("revamp", bus=b, power_capacity=0.0, energy_capacity=0.0,
                        extendable=True, max_power_capacity=500.0,
                        max_energy_capacity=1000.0, capital_cost_power=1.0,
                        capital_cost_energy=1.0,
                        lifetime_years=25, retrofit_of="battery")
        return sys

    mss.add_stage(year=2030, system=build(2030))
    mss.add_stage(year=2040, system=build(2040))
    r = mss.optimise()
    assert r.status == "optimal"
    # Stage 0: nothing retiring → revamp power new-build is 0.
    assert r.storage_new_power["revamp"][0] == pytest.approx(0.0, abs=1e-3)
    # Stage 1: battery's 50 MW retires → revamp power ≤ 50 MW.
    assert r.storage_new_power["revamp"][1] <= 50.0 + 1e-3


def test_link_retrofit_capped_by_retiring_host():
    """Hydrogen pipeline as a retrofit of a retiring gas pipeline."""
    mss = ne.MultiStageSystem("lk_retro")

    def build(year: int) -> ne.EnergySystem:
        sys = ne.EnergySystem(f"s{year}")
        sys.set_timesteps(1, dt=1.0)
        a = sys.add_bus("a")
        b = sys.add_bus("b")
        sys.add_generator("g", bus=a, capacity=100.0, marginal_cost=1.0)
        sys.add_load("d", bus=b, amount=20.0)
        # Host gas pipeline: 40 MW brownfield retiring in 2040.
        sys.add_link("gas_pipe", bus_from=a, bus_to=b, capacity=40.0,
                     retire_at_year=2040)
        # H2 pipeline retrofit.
        sys.add_link("h2_pipe", bus_from=a, bus_to=b, capacity=0.0,
                     extendable=True, max_capacity=500.0, capital_cost=1.0,
                     lifetime_years=40, retrofit_of="gas_pipe")
        return sys

    mss.add_stage(year=2030, system=build(2030))
    mss.add_stage(year=2040, system=build(2040))
    r = mss.optimise()
    assert r.status == "optimal"
    assert r.link_new_builds["h2_pipe"][0] == pytest.approx(0.0, abs=1e-3)
    assert r.link_new_builds["h2_pipe"][1] <= 40.0 + 1e-3


def test_electrolyzer_partload_efficiency_concave():
    """Part-load efficiency curve: delivered H2 follows the concave envelope.
    At low load efficiency is higher; the LP gets less-than-flat output at
    high load."""
    sys = ne.EnergySystem("elz")
    sys.set_timesteps(1)
    e = sys.add_bus("elec")
    h = sys.add_bus("h2", carrier="hydrogen")
    sys.add_generator("grid", bus=e, capacity=100.0, marginal_cost=1.0)
    # electrolyzer 100 MW: 80% eff at 50% load, 70% eff at full load (concave).
    sys.add_link("elz", bus_from=e, bus_to=h, capacity=100.0,
                 efficiency=0.7, efficiency_segments=[(0.5, 0.8), (1.0, 0.7)])
    sys.add_load("h2d", bus=h, amount=40.0)  # need 40 MW H2
    r = sys.optimise()
    assert r.status == "optimal"
    # Delivered output must satisfy 40 MW demand; the envelope binds so the
    # electrolyzer draws enough power. At 50% load (50 MW in) output = 40 MW
    # (eff 0.8), so it should run near 50 MW input, not 57 (40/0.7).
    flow = np.asarray(r.link_flow["elz"])
    assert flow[0] <= 50.0 + 1e-3, f"input {flow[0]} exceeds high-eff point"


def test_shared_capacity_converter_mutex():
    """Electrolyzer + fuel cell sharing one converter cannot both run at full
    power: combined throughput ≤ shared rating."""
    sys = ne.EnergySystem("share")
    sys.set_timesteps(1)
    e = sys.add_bus("elec")
    h = sys.add_bus("h2", carrier="hydrogen")
    sys.add_generator("grid", bus=e, capacity=200.0, marginal_cost=1.0)
    sys.add_generator("h2src", bus=h, capacity=200.0, marginal_cost=1.0)
    sys.add_link("elz", bus_from=e, bus_to=h, capacity=50.0, efficiency=0.7)
    sys.add_link("fc", bus_from=h, bus_to=e, capacity=50.0, efficiency=0.5)
    sys.add_load("hd", bus=h, amount=30.0)
    sys.add_load("ed", bus=e, amount=10.0)
    sys.set_shared_capacity(["elz", "fc"], mutex=True)
    r = sys.optimise()
    assert r.status == "optimal"
    f_elz = np.asarray(r.link_flow["elz"])[0]
    f_fc = np.asarray(r.link_flow["fc"])[0]
    assert f_elz + f_fc <= 50.0 + 1e-4, f"shared-cap mutex violated: {f_elz}+{f_fc}"


def test_temperature_heat_network_builder():
    """create_temperature_heat_network wires hot/cold buses + HX (+booster)."""
    from nexus_energy.sectors import create_temperature_heat_network
    sys = ne.EnergySystem("dh")
    sys.set_timesteps(2)
    e = sys.add_bus("elec")
    sys.add_generator("grid", bus=e, capacity=100.0, marginal_cost=1.0)
    comps = create_temperature_heat_network(
        sys, elec_bus=e, hx_capacity=50.0, hx_efficiency=0.95,
        booster_cop=3.0, booster_capacity=30.0)
    assert "hot_bus" in comps and "cold_bus" in comps
    assert "heat_exchanger" in comps and "booster" in comps
    # High-grade source + low-temp demand → flows through the exchanger.
    sys.add_generator("chp", bus=comps["hot_bus"], capacity=100.0, marginal_cost=2.0)
    sys.add_load("low_demand", bus=comps["cold_bus"], amount=np.array([20.0, 20.0]))
    r = sys.optimise()
    assert r.status == "optimal"
    hx = np.asarray(r.link_flow[comps["heat_exchanger"].name])
    # 20 MW low-temp demand / 0.95 efficiency ≈ 21.05 MW through the HX.
    assert hx[0] > 20.0
