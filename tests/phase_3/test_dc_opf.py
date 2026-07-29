"""Phase 3 — DC-OPF (linearised KVL) tests.

A 3-bus loop network is the smallest case where transport-model dispatch
and DC-OPF dispatch diverge: the loop pins flow distribution to the
reactance ratios, so the LP can no longer freely route around expensive
lines.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne


def _three_bus_loop(reactances=(0.1, 0.1, 0.1), capacities=(200, 200, 200)):
    """3-bus loop: cheap gen at b0, expensive at b2, load at b1.

    Lines: l01 (b0→b1), l12 (b1→b2), l02 (b0→b2). Returns (system, links).
    """
    sys = ne.EnergySystem("loop3")
    b0 = sys.add_bus("b0")
    b1 = sys.add_bus("b1")
    b2 = sys.add_bus("b2")
    sys.add_generator("g_cheap", bus=b0, capacity=200, marginal_cost=10)
    sys.add_generator("g_exp",   bus=b2, capacity=200, marginal_cost=100)
    sys.add_load("ld", bus=b1, amount=100)
    l01 = sys.add_link("l01", bus_from=b0, bus_to=b1,
                       capacity=capacities[0], reactance=reactances[0],
                       model_type="dc_opf")
    l12 = sys.add_link("l12", bus_from=b1, bus_to=b2,
                       capacity=capacities[1], reactance=reactances[1],
                       model_type="dc_opf")
    l02 = sys.add_link("l02", bus_from=b0, bus_to=b2,
                       capacity=capacities[2], reactance=reactances[2],
                       model_type="dc_opf")
    return sys, (l01, l12, l02)


class TestDCOPFKVLIdentity:
    def test_equal_reactance_loop_splits_two_thirds_one_third(self):
        sys, _ = _three_bus_loop()
        result = sys.optimise()
        assert result.status == "optimal"
        # Cheap gen alone serves 100 MW, KVL splits: l01=2/3*100, l02=1/3*100,
        # l12=-1/3*100 (b2→b1, "negative" because line is oriented b1→b2).
        assert result.link_flow["l01"][0] == pytest.approx(66.6667, abs=0.01)
        assert result.link_flow["l02"][0] == pytest.approx(33.3333, abs=0.01)
        assert result.link_flow["l12"][0] == pytest.approx(-33.3333, abs=0.01)

    def test_asymmetric_reactance_redistributes_flow(self):
        # Make l02 "longer" (high reactance) — flow should prefer l01.
        sys, _ = _three_bus_loop(reactances=(0.1, 0.1, 1.0))
        result = sys.optimise()
        assert result.status == "optimal"
        # With x02 >> x01,x12, the parallel path b0→b2 sees ~10x impedance
        # of b0→b1→b2, so it carries far less flow.
        assert abs(result.link_flow["l01"][0]) > abs(result.link_flow["l02"][0])

    def test_dc_opf_cheaper_than_isolated_pure_cheap(self):
        # Pure-LP transport could dispatch all-cheap (100 MW) freely; DC-OPF
        # forces the same answer on this instance because cap(200) on every
        # line is non-binding.
        sys, _ = _three_bus_loop()
        result = sys.optimise()
        assert result.total_cost == pytest.approx(1000.0, abs=1.0)


class TestDCOPFCapacityBinding:
    def test_tight_line_forces_expensive_generation(self):
        # Cap l02 at 10 MW. KVL with all-cheap dispatch sends 33 MW around
        # the loop on l02 — can't fit. The LP must dispatch some g_exp at b2
        # to relieve the KVL-induced flow on l02.
        sys, _ = _three_bus_loop(capacities=(200, 200, 10))
        result = sys.optimise()
        assert result.status == "optimal"
        # Expensive gen is used; cost should exceed pure-cheap baseline.
        assert result.total_cost > 1000.0
        assert result.generator_dispatch["g_exp"].sum() > 1.0

    def test_unloaded_loop_has_zero_flow(self):
        sys = ne.EnergySystem("idle")
        b0 = sys.add_bus("b0"); b1 = sys.add_bus("b1"); b2 = sys.add_bus("b2")
        sys.add_generator("g", bus=b0, capacity=100, marginal_cost=10)
        sys.add_load("ld", bus=b0, amount=50)  # load at slack — no transmission
        for a, c, name in [(b0, b1, "l01"), (b1, b2, "l12"), (b0, b2, "l02")]:
            sys.add_link(name, bus_from=a, bus_to=c, capacity=200,
                         reactance=0.1, model_type="dc_opf")
        result = sys.optimise()
        assert result.status == "optimal"
        for name in ("l01", "l12", "l02"):
            assert abs(result.link_flow[name][0]) < 1e-6
