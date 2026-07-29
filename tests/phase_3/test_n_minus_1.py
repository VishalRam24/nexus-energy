"""Phase 3 — preventive N-1 security tests.

A flow-based N-1 constraint adds a replica DC-OPF state per contingency
line. With dispatch held fixed (preventive), losing the cheapest path
forces the LP to either route flow on a longer parallel path (often
binding the longer path's capacity) or pre-position generation on the
load side.
"""

from __future__ import annotations

import pytest

import nexus_energy as ne


def _two_path_system(secure_l01: bool):
    """b0 (cheap gen) → b1 (load), via two parallel lines l01a / l01b.

    Each line cap = 60 MW. Demand = 100 MW. Without security: dispatch
    splits 50/50 (KVL with equal reactance). With N-1 against l01a: the
    base-case dispatch must still satisfy load if l01a trips, i.e. l01b
    alone (60 MW cap) cannot carry 100 MW → infeasible unless we add a
    local gen on b1.
    """
    sys = ne.EnergySystem("n_minus_1")
    b0 = sys.add_bus("b0"); b1 = sys.add_bus("b1")
    sys.add_generator("g_cheap", bus=b0, capacity=200, marginal_cost=10)
    sys.add_generator("g_local", bus=b1, capacity=200, marginal_cost=80)
    sys.add_load("ld", bus=b1, amount=100)
    sys.add_link("l01a", bus_from=b0, bus_to=b1, capacity=60,
                 reactance=0.1, model_type="dc_opf")
    sys.add_link("l01b", bus_from=b0, bus_to=b1, capacity=60,
                 reactance=0.1, model_type="dc_opf")
    if secure_l01:
        sys.set_n_minus_1(["l01a"])
    return sys


class TestNMinus1:
    def test_secure_dispatch_costs_more(self):
        r_base = _two_path_system(secure_l01=False).optimise()
        r_sec  = _two_path_system(secure_l01=True).optimise()
        assert r_base.status == "optimal"
        assert r_sec.status  == "optimal"
        # Security costs money: pre-positioning g_local raises the bill.
        assert r_sec.total_cost >= r_base.total_cost - 1e-3

    def test_secure_dispatch_uses_local_gen(self):
        r_sec = _two_path_system(secure_l01=True).optimise()
        # Local (expensive) gen must dispatch enough so that, on l01a outage,
        # the surviving line (60 MW) can still cover the residual cheap
        # injection (which then = 60 MW). i.e. local gen ≥ 40 MW.
        local = r_sec.generator_dispatch["g_local"].sum()
        assert local >= 40.0 - 1e-3

    def test_no_contingency_lines_is_noop(self):
        # set_n_minus_1([]) leaves the model identical to the base case.
        sys = _two_path_system(secure_l01=False)
        sys.set_n_minus_1([])
        r = sys.optimise()
        assert r.status == "optimal"
        # Base-case dispatch should be all-cheap (load 100 split across
        # the two 60 MW lines).
        assert r.total_cost == pytest.approx(1000.0, abs=1.0)
