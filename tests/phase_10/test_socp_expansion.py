"""N_En_Phase 10.2 — capacity-expansion AC-OPF (SOCP relaxation).

TINY Clarabel smoke tests for :func:`solve_socp_opf_expansion`: investment +
dispatch co-optimisation on the Jabr SOCP lift. Generators / lines flagged
``extendable=True`` get their rating promoted to a decision variable bounded by
``[min_capacity, max_capacity]`` with a linear ``capital_cost·cap`` investment
term layered on the operating objective.

All instances are deliberately tiny (≤ 3 buses).
"""

from __future__ import annotations

import pytest

import nexus_energy as ne
from nexus_energy._conic import is_available as clarabel_available
from nexus_energy.network_socp import (
    solve_socp_opf_expansion,
    SOCPExpansionResult,
)


pytestmark = pytest.mark.skipif(
    not clarabel_available(), reason="clarabel not installed")


def _radial(*, extendable: bool, cap: float, capital_cost: float = 0.0):
    """2-bus radial: extendable gen on b1 feeding a 0.5 pu load on b2."""
    sys = ne.EnergySystem("socp_exp")
    b1 = sys.add_bus("b1", carrier="electricity")
    b2 = sys.add_bus("b2", carrier="electricity")
    g = sys.add_generator("g", bus=b1, capacity=cap, marginal_cost=10.0)
    g.extendable = extendable
    g.max_capacity = 5.0
    g.min_capacity = 0.0
    g.capital_cost = capital_cost
    sys.add_load("d", bus=b2, amount=0.5)
    link = sys.add_link("l", bus_from=b1, bus_to=b2, capacity=2.0)
    link.resistance = 0.01
    link.reactance = 0.10
    link.s_max = 2.0
    link.model_type = "socp_opf"
    return sys, g


def test_expansion_off_matches_fixed_dispatch():
    """No extendable component ⇒ expansion result equals plain dispatch.

    With nothing flagged ``extendable`` the build is the same fixed-capacity
    program, so the expansion entry-point must reproduce the dispatch-only
    optimum (and report no chosen capacities).
    """
    sys_fixed, _ = _radial(extendable=False, cap=2.0)
    sys_exp, _ = _radial(extendable=False, cap=2.0)

    r_fixed = ne.solve_socp_opf(sys_fixed)
    r_exp = solve_socp_opf_expansion(sys_exp)

    assert r_fixed.status == "optimal"
    assert r_exp.status == "optimal"
    assert isinstance(r_exp, SOCPExpansionResult)
    assert r_exp.total_cost == pytest.approx(r_fixed.total_cost, rel=1e-6)
    assert r_exp.gen_capacity == {}
    assert r_exp.line_capacity == {}


def test_expansion_lowers_cost_vs_starved_fixed():
    """Capacity-starved fixed case is infeasible / dearer; investment fixes it.

    Fixed gen capacity 0.3 pu cannot serve a 0.5 pu load (+losses) → the
    SOCP has no feasible dispatch. Allowing the gen to be extendable (with a
    modest capital cost) lets the planner buy ~0.5 pu of capacity and serve
    the load. The expansion total cost must be finite/optimal, the bought
    capacity must respect [min_capacity, max_capacity], and it must be at
    least the served load.
    """
    # (a) starved fixed case — gen too small to cover the load.
    sys_starved, _ = _radial(extendable=False, cap=0.3)
    r_starved = ne.solve_socp_opf(sys_starved)
    assert r_starved.status != "optimal"  # no feasible dispatch

    # (b) expansion case — same network, gen now extendable.
    sys_exp, g = _radial(extendable=True, cap=0.3, capital_cost=100.0)
    r_exp = solve_socp_opf_expansion(sys_exp)
    assert r_exp.status == "optimal"

    cap = r_exp.gen_capacity["g"]
    assert g.min_capacity - 1e-6 <= cap <= g.max_capacity + 1e-6
    # Must build enough to carry load + losses.
    assert cap >= 0.5 - 1e-4
    assert r_exp.gen_p["g"] >= 0.5 - 1e-4
    # Built capacity tracks dispatch (no over-build — investment is costed).
    assert cap == pytest.approx(r_exp.gen_p["g"], abs=5e-3)

    # Total = operating (mc·p) + investment (capital_cost·cap), both > 0.
    op = 10.0 * r_exp.gen_p["g"]
    inv = 100.0 * cap
    assert r_exp.total_cost == pytest.approx(op + inv, rel=1e-3)


def test_expansion_cheaper_than_oversized_fixed():
    """Investment co-opt beats paying for unused fixed capacity.

    Compare a fixed gen sized at the 5 pu max (paying capital_cost on all
    5 pu via an equivalent accounting) against the extendable case which
    buys only what it dispatches. The extendable plan's total cost must be
    strictly lower than financing the oversized fixed rating.
    """
    capital_cost = 100.0

    # Extendable: buys ~load.
    sys_exp, _ = _radial(extendable=True, cap=0.3, capital_cost=capital_cost)
    r_exp = solve_socp_opf_expansion(sys_exp)
    assert r_exp.status == "optimal"
    cap_built = r_exp.gen_capacity["g"]

    # Oversized fixed reference: dispatch the same load but financed at the
    # 5 pu rating (capital on the full fixed plate).
    sys_fixed, _ = _radial(extendable=False, cap=5.0)
    r_fixed = ne.solve_socp_opf(sys_fixed)
    assert r_fixed.status == "optimal"
    oversized_total = r_fixed.total_cost + capital_cost * 5.0

    assert r_exp.total_cost < oversized_total
    assert cap_built < 5.0  # did not over-build


def test_expansion_line_extendable():
    """Extendable line rating is a decision variable bounded by max_capacity."""
    sys = ne.EnergySystem("socp_exp_line")
    b1 = sys.add_bus("b1", carrier="electricity")
    b2 = sys.add_bus("b2", carrier="electricity")
    sys.add_generator("g", bus=b1, capacity=5.0, marginal_cost=10.0)
    sys.add_load("d", bus=b2, amount=0.5)
    link = sys.add_link("l", bus_from=b1, bus_to=b2, capacity=2.0)
    link.resistance = 0.01
    link.reactance = 0.10
    link.extendable = True
    link.min_capacity = 0.0
    link.max_capacity = 2.0
    link.capital_cost = 50.0
    link.model_type = "socp_opf"

    r = solve_socp_opf_expansion(sys)
    assert r.status == "optimal"
    s_cap = r.line_capacity["l"]
    assert 0.0 - 1e-6 <= s_cap <= 2.0 + 1e-6
    # Must carry the sending-end apparent power.
    import math
    s_flow = math.hypot(r.branch_p["l"], r.branch_q["l"])
    assert s_cap >= s_flow - 1e-4
