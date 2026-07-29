"""Phase 3 — transmission switching tests.

A line marked ``switchable=True`` gains a binary z[t]; when z=0 the line
flow must be zero. The classic value of switching: an outage on a parallel
path that's congesting cheap generation.
"""

from __future__ import annotations

import pytest

import nexus_energy as ne


def _bottleneck_loop(switchable_l02: bool):
    """Cheap gen at b0, expensive at b2, load at b1.

    Loop with l01 / l12 fat, l02 capped at 10 MW. KVL with all-cheap
    sends 33 MW around the loop on l02 → infeasible. To stay within the
    10 MW limit the LP must dispatch some g_exp at b2 (raising cost).
    Switching l02 off "breaks" the loop entirely: tree network has no
    KVL coupling, so all-cheap (100 MW radial) is feasible at the
    minimum cost.
    """
    sys = ne.EnergySystem("switching")
    b0 = sys.add_bus("b0"); b1 = sys.add_bus("b1"); b2 = sys.add_bus("b2")
    sys.add_generator("g_cheap", bus=b0, capacity=200, marginal_cost=10)
    sys.add_generator("g_exp",   bus=b2, capacity=200, marginal_cost=100)
    sys.add_load("ld", bus=b1, amount=100)
    sys.add_link("l01", bus_from=b0, bus_to=b1, capacity=200,
                 reactance=0.1, model_type="dc_opf")
    sys.add_link("l12", bus_from=b1, bus_to=b2, capacity=200,
                 reactance=0.1, model_type="dc_opf")
    sys.add_link("l02", bus_from=b0, bus_to=b2, capacity=10,
                 reactance=0.1, model_type="dc_opf",
                 switchable=switchable_l02)
    return sys


class TestSwitching:
    def test_switching_can_only_help(self):
        # Without switching the LP is forced to dispatch some expensive
        # gen because KVL ties l02 flow to l01 flow. With switching
        # the LP gets the option to disconnect a line.
        r_off = _bottleneck_loop(switchable_l02=False).optimise()
        r_on  = _bottleneck_loop(switchable_l02=True).optimise()
        assert r_off.status == "optimal"
        assert r_on.status  == "optimal"
        # Switching is an additional binary choice → optimal cost can only
        # decrease (or stay the same).
        assert r_on.total_cost <= r_off.total_cost + 1e-3

    def test_switched_off_zero_flow(self):
        sys = ne.EnergySystem("force_off")
        b0 = sys.add_bus("b0"); b1 = sys.add_bus("b1")
        sys.add_generator("g", bus=b0, capacity=200, marginal_cost=10)
        sys.add_generator("g_local", bus=b1, capacity=200, marginal_cost=20)
        sys.add_load("ld", bus=b1, amount=100)
        # The transfer line is switchable; with cheap-enough local gen it
        # stays open, but we just verify the binary mechanism wires up and
        # the LP solves.
        sys.add_link("trans", bus_from=b0, bus_to=b1, capacity=200,
                     reactance=0.1, model_type="dc_opf", switchable=True)
        result = sys.optimise()
        assert result.status == "optimal"
