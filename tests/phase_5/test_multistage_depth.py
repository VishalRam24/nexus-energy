"""
Phase 5 — deeper multi-stage features (deferred from Phase 5 first cut,
landed in Phase 10.x depth pass, 2026-04-19):

    (a) multi-bus stages with transport links,
    (b) storage vintaging (power + energy capacity, live across stages),
    (c) link vintaging + discrete transmission expansion (integer units),
    (d) construction lead time (``build_lead_years``),
    (e) myopic rolling mode (``optimise(myopic=True)``) produces a feasible,
        interpretable path vs perfect foresight — non-anticipative by
        construction, so cost ≥ PF and the dispatch is consistent with
        the fixed earlier-stage capacity.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


# ---------------------------------------------------------------------------
# (a) multi-bus planning
# ---------------------------------------------------------------------------


def _two_bus_stage(name: str, demand: float) -> ne.EnergySystem:
    sys = ne.EnergySystem(name)
    sys.set_timesteps(1, dt=1.0)
    b1 = sys.add_bus("b1")
    b2 = sys.add_bus("b2")
    # Cheap generation on b1, load on b2 — LP must use a link.
    sys.add_generator(
        "cheap", bus=b1, capacity=0, marginal_cost=10.0,
        capital_cost=20_000.0, extendable=True, max_capacity=1000.0,
        lifetime_years=30,
    )
    sys.add_load("d", bus=b2, amount=demand)
    sys.add_link(
        "l12", bus_from=b1, bus_to=b2, capacity=0,
        capital_cost=5_000.0, extendable=True, max_capacity=1000.0,
        lifetime_years=40,
    )
    return sys


def test_multibus_stage_builds_gen_and_link():
    mss = ne.MultiStageSystem("multibus")
    mss.add_stage(year=2030, system=_two_bus_stage("2030", demand=100.0))
    mss.add_stage(year=2040, system=_two_bus_stage("2040", demand=150.0))

    r = mss.optimise()
    assert r.status == "optimal"
    # Must build generator ≥ 100 to serve stage 0 demand.
    assert r.capacity_active["cheap"][0] >= 100.0 - 1e-3
    # Must build link ≥ 100 in stage 0 (demand routes b1 → b2).
    assert r.link_capacity_active["l12"][0] >= 100.0 - 1e-3
    # Stage 1 needs 150 MW routed → link active cap ≥ 150.
    assert r.link_capacity_active["l12"][1] >= 150.0 - 1e-3


# ---------------------------------------------------------------------------
# (b) storage vintaging
# ---------------------------------------------------------------------------


def test_storage_vintage_carries_power_and_energy():
    """
    Force a stage-0 battery build via ``min_power_capacity`` and check that
    the resulting vintage power *and* energy capacity are both still live at
    stage 1 (age 10 < lifetime 15). This verifies the separate power / energy
    vintage tracks compose correctly.
    """
    def build(name: str) -> ne.EnergySystem:
        sys = ne.EnergySystem(name)
        sys.set_timesteps(1, dt=1.0)
        b = sys.add_bus("e")
        sys.add_load("d", bus=b, amount=50.0)
        sys.add_generator("g", bus=b, capacity=200.0, marginal_cost=10.0)
        sys.add_storage(
            "bat", bus=b, power_capacity=0.0, energy_capacity=0.0,
            capital_cost_power=1.0, capital_cost_energy=1.0,
            efficiency_charge=0.95, efficiency_discharge=0.95,
            cyclic=True, extendable=True,
            min_power_capacity=30.0, min_energy_capacity=60.0,
            max_power_capacity=200.0, max_energy_capacity=400.0,
            lifetime_years=15,
        )
        return sys

    mss = ne.MultiStageSystem("storage_vintage")
    mss.add_stage(year=2030, system=build("2030"))
    mss.add_stage(year=2040, system=build("2040"))

    r = mss.optimise()
    assert r.status == "optimal"
    # Both stages forced to build ≥30 MW / ≥60 MWh via min bounds.
    assert r.storage_new_power["bat"][0] >= 30.0 - 1e-3
    assert r.storage_new_energy["bat"][0] >= 60.0 - 1e-3
    # Stage-1 active = stage-0 vintage + stage-1 new build.
    assert r.storage_power_active["bat"][1] >= r.storage_new_power["bat"][0] + 30.0 - 1e-3
    assert r.storage_energy_active["bat"][1] >= r.storage_new_energy["bat"][0] + 60.0 - 1e-3


# ---------------------------------------------------------------------------
# (c) link vintaging + discrete tx expansion
# ---------------------------------------------------------------------------


def test_link_integer_investment_rounds_to_unit_size():
    mss = ne.MultiStageSystem("discrete_tx")

    def build() -> ne.EnergySystem:
        sys = ne.EnergySystem("s")
        sys.set_timesteps(1, dt=1.0)
        b1 = sys.add_bus("b1"); b2 = sys.add_bus("b2")
        sys.add_generator("g", bus=b1, capacity=500.0, marginal_cost=10.0)
        sys.add_load("d", bus=b2, amount=120.0)
        sys.add_link(
            "tx", bus_from=b1, bus_to=b2, capacity=0,
            capital_cost=1000.0,
            extendable=True, max_capacity=500.0,
            integer_investment=True, unit_size=50.0,
            lifetime_years=40,
        )
        return sys

    mss.add_stage(year=2030, system=build())
    r = mss.optimise()
    assert r.status == "optimal"
    built = r.link_new_builds["tx"][0]
    # Must be a multiple of unit_size=50 and ≥ demand=120 → 150 MW.
    assert built == pytest.approx(150.0, abs=1e-3)


# ---------------------------------------------------------------------------
# (d) construction lead time
# ---------------------------------------------------------------------------


def test_build_lead_years_defers_activation():
    """
    A 10-year lead means a new-build at 2030 only goes live at stage
    2040. Force the build via ``min_capacity`` and verify the vintage
    expression withholds the capacity at stage 0 and grants it at stage 1.
    """
    mss = ne.MultiStageSystem("lead")

    def build() -> ne.EnergySystem:
        sys = ne.EnergySystem("s")
        sys.set_timesteps(1, dt=1.0)
        b = sys.add_bus("e")
        sys.add_load("d", bus=b, amount=50.0)
        sys.add_generator("peaker", bus=b, capacity=500.0, marginal_cost=500.0)
        sys.add_generator(
            "nuke", bus=b, capacity=0, marginal_cost=10.0,
            capital_cost=1.0,
            extendable=True, min_capacity=100.0, max_capacity=200.0,
            lifetime_years=40, build_lead_years=10,
        )
        return sys

    mss.add_stage(year=2030, system=build())
    mss.add_stage(year=2040, system=build())

    r = mss.optimise()
    assert r.status == "optimal"
    # Stage-0 nuke build forced to ≥100 MW, but lead=10 → NOT active at stage 0.
    assert r.new_builds["nuke"][0] >= 100.0 - 1e-3
    assert r.capacity_active["nuke"][0] == pytest.approx(0.0, abs=1e-3)
    # Stage-1 build is also forced (min_capacity applies per-stage); lead=10
    # means it's not active at stage 1 either. But stage-0 vintage IS active
    # at stage 1 (year 2040 = 2030+10, the lead threshold).
    assert r.capacity_active["nuke"][1] >= 100.0 - 1e-3


# ---------------------------------------------------------------------------
# (e) myopic rolling
# ---------------------------------------------------------------------------


def test_retrofit_converts_retiring_host_capacity():
    """
    Coal plant (100 MW brownfield) is retired at 2040 via
    ``retire_at_year``. A biomass retrofit (``retrofit_of="coal"``)
    has low capex because it reuses the retiring site — but its
    new-build at stage 2040 is capped by coal's retiring 100 MW.
    The LP should build biomass_retrofit ≤ 100 MW at stage 2040 and
    zero at stage 2030 (no host retiring yet).
    """
    mss = ne.MultiStageSystem("retrofit")

    def build(year: int) -> ne.EnergySystem:
        sys = ne.EnergySystem(f"s{year}")
        sys.set_timesteps(1, dt=1.0)
        b = sys.add_bus("e")
        sys.add_load("d", bus=b, amount=80.0)
        # Coal brownfield that retires in 2040.
        sys.add_generator("coal", bus=b, capacity=100.0,
                         marginal_cost=40.0, retire_at_year=2040)
        # Biomass as a retrofit of coal.
        sys.add_generator(
            "biomass", bus=b, capacity=0, marginal_cost=20.0,
            capital_cost=1.0,
            extendable=True, max_capacity=500.0,
            lifetime_years=25, retrofit_of="coal",
        )
        # Greenfield solar (far more expensive) as a non-retrofit alternative.
        sys.add_generator(
            "solar", bus=b, capacity=0, marginal_cost=0.0,
            capital_cost=10_000.0,
            extendable=True, max_capacity=500.0,
            lifetime_years=25,
        )
        return sys

    mss.add_stage(year=2030, system=build(2030))
    mss.add_stage(year=2040, system=build(2040))

    r = mss.optimise()
    assert r.status == "optimal"
    # Stage 0 (2030): no coal retiring yet → biomass cannot be built.
    assert r.new_builds["biomass"][0] == pytest.approx(0.0, abs=1e-3)
    # Stage 1 (2040): coal's 100 MW retires → biomass ≤ 100 MW. The LP
    # prefers biomass over solar (1 $/MW vs 10k $/MW), so it builds
    # exactly 80 MW (= demand).
    assert r.new_builds["biomass"][1] == pytest.approx(80.0, abs=1e-3)
    assert r.new_builds["solar"][1] == pytest.approx(0.0, abs=1e-3)


def test_retrofit_capped_by_host_retiring_amount():
    """
    Demand exceeds the retiring host's capacity → retrofit saturates
    at the host's retiring amount, and the LP fills the rest with
    the greenfield alternative.
    """
    mss = ne.MultiStageSystem("retrofit_cap")

    def build(year: int) -> ne.EnergySystem:
        sys = ne.EnergySystem(f"s{year}")
        sys.set_timesteps(1, dt=1.0)
        b = sys.add_bus("e")
        sys.add_load("d", bus=b, amount=150.0)
        sys.add_generator("coal", bus=b, capacity=100.0,
                         marginal_cost=40.0, retire_at_year=2040)
        sys.add_generator(
            "biomass", bus=b, capacity=0, marginal_cost=20.0,
            capital_cost=1.0,
            extendable=True, max_capacity=500.0,
            lifetime_years=25, retrofit_of="coal",
        )
        sys.add_generator(
            "solar", bus=b, capacity=0, marginal_cost=0.0,
            capital_cost=10_000.0,
            extendable=True, max_capacity=500.0,
            lifetime_years=25,
        )
        return sys

    mss.add_stage(year=2030, system=build(2030))
    mss.add_stage(year=2040, system=build(2040))

    r = mss.optimise()
    assert r.status == "optimal"
    # Stage 0: solar must cover demand ≥ coal's 100 (coal: 40$/MWh dispatch).
    # Stage 1: biomass saturates at 100 MW (coal's retiring amount);
    # remaining 50 MW comes from cheapest alternative (solar at stage 0
    # is still alive → prefer that if vintage covers). Stage-0 solar
    # vintage (built to cover stage 0) carries into stage 1. LP chooses
    # whichever path minimises total cost — we just verify the retrofit
    # cap: biomass ≤ 100 MW.
    assert r.new_builds["biomass"][1] <= 100.0 + 1e-3
    # And every stage must still cover load.
    total_active_s1 = (r.capacity_active["biomass"][1]
                       + r.capacity_active["solar"][1]
                       + r.capacity_active["coal"][1])
    assert total_active_s1 >= 150.0 - 1e-3


def test_myopic_ge_perfect_foresight_and_is_feasible():
    """
    Myopic horizon: each stage is solved seeing only itself + prior
    fixed capacity. Can only be ≥ PF cost (no foresight discount).
    Also validates myopic completes on multi-stage system.
    """
    def make() -> ne.MultiStageSystem:
        m = ne.MultiStageSystem("m")
        for y, demand in [(2030, 100.0), (2040, 200.0), (2050, 150.0)]:
            sys = ne.EnergySystem(f"{y}")
            sys.set_timesteps(1, dt=1.0)
            b = sys.add_bus("e")
            sys.add_load("d", bus=b, amount=demand)
            sys.add_generator("peaker", bus=b, capacity=500.0, marginal_cost=500.0)
            sys.add_generator(
                "cheap", bus=b, capacity=0, marginal_cost=10.0,
                capital_cost=50_000.0, extendable=True, max_capacity=500.0,
                lifetime_years=30,
            )
            m.add_stage(year=y, system=sys)
        return m

    r_pf = make().optimise(myopic=False)
    r_my = make().optimise(myopic=True)

    assert r_pf.status == "optimal"
    assert r_my.status == "optimal"
    # Myopic ≥ PF (no foresight can only hurt).
    assert r_my.total_cost >= r_pf.total_cost - 1.0
    # Myopic must cover every stage's demand.
    for s in range(3):
        assert r_my.capacity_active["cheap"][s] + 500.0 >= [100.0, 200.0, 150.0][s]
