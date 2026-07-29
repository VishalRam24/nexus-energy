"""Phase 3 — PTDF formulation should be observationally equivalent to DC-OPF.

PTDF is a derived map from injections to flows; given the same injections
the LP should pick the same flows (within numerical tolerance) as the
explicit theta-based formulation. This test runs both formulations on
the same instance and compares.
"""

from __future__ import annotations

import pytest

import nexus_energy as ne


def _three_bus_loop(model_type="dc_opf"):
    sys = ne.EnergySystem(f"loop3_{model_type}")
    b0 = sys.add_bus("b0"); b1 = sys.add_bus("b1"); b2 = sys.add_bus("b2")
    sys.add_generator("g_cheap", bus=b0, capacity=200, marginal_cost=10)
    sys.add_generator("g_exp",   bus=b2, capacity=200, marginal_cost=100)
    sys.add_load("ld", bus=b1, amount=100)
    sys.add_link("l01", bus_from=b0, bus_to=b1, capacity=200,
                 reactance=0.1, model_type=model_type)
    sys.add_link("l12", bus_from=b1, bus_to=b2, capacity=200,
                 reactance=0.1, model_type=model_type)
    sys.add_link("l02", bus_from=b0, bus_to=b2, capacity=200,
                 reactance=0.1, model_type=model_type)
    return sys


class TestPTDFEquivalence:
    def test_ptdf_matches_dc_opf_objective(self):
        r_dc = _three_bus_loop("dc_opf").optimise()
        r_pt = _three_bus_loop("ptdf").optimise()
        assert r_dc.status == "optimal"
        assert r_pt.status == "optimal"
        assert r_pt.total_cost == pytest.approx(r_dc.total_cost, rel=1e-4)

    def test_ptdf_matches_dc_opf_flows(self):
        r_dc = _three_bus_loop("dc_opf").optimise()
        r_pt = _three_bus_loop("ptdf").optimise()
        for name in ("l01", "l12", "l02"):
            assert r_pt.link_flow[name][0] == pytest.approx(
                r_dc.link_flow[name][0], abs=1e-2
            )


class TestPTDFMatrixShape:
    def test_ptdf_matrix_dimensions(self):
        from nexus_energy.network import build_ptdf_matrix, _ptdf_lines
        sys = _three_bus_loop("ptdf")
        # Trigger _id assignment by running optimise() quickly.
        sys.optimise()
        lines = _ptdf_lines(sys)
        ptdf = build_ptdf_matrix(sys, lines)
        assert ptdf.shape == (3, 3)  # 3 lines × 3 buses

    def test_ptdf_slack_column_is_zero(self):
        # By construction the slack-bus column of PTDF is zero (we never
        # invert it; it's reinserted as a zero row/col).
        from nexus_energy.network import build_ptdf_matrix, _ptdf_lines
        sys = _three_bus_loop("ptdf")
        sys.optimise()
        lines = _ptdf_lines(sys)
        ptdf = build_ptdf_matrix(sys, lines)
        # Bus 0 is slack — column 0 should be all zero.
        assert (abs(ptdf[:, 0]) < 1e-10).all()
