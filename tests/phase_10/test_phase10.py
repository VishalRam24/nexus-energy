"""
Phase 10 — multi-solver + conic backend tests.

Coverage:
  (a) ``solver="highs"`` routes through end-to-end and returns an optimal
      solution matching the default HiGHS path;
  (b) Clarabel direct adapter solves a textbook SOC problem to optimality;
  (c) SOCP AC-OPF relaxation solves a 3-bus radial case with |V| in bounds
      and power balance satisfied;
  (d) PWL quadratic transmission loss strictly raises total cost vs the
      lossless baseline and the loss variable is tight-bounded above by
      ``loss_quadratic · f²`` at the optimum;
  (e) ``rolling_horizon_solve(warm_start=True)`` is a no-op for pure-LP
      windows (identical dispatch vs ``warm_start=False``) — documents the
      LP-basis deferral while keeping the API stable.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy._conic import ConicProblem, is_available as clarabel_available


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _tiny_lp(T: int = 4) -> ne.EnergySystem:
    sys = ne.EnergySystem("tiny")
    elec = sys.add_bus("elec", carrier="electricity")
    load = np.full(T, 80.0)
    sys.add_load("d", bus=elec, amount=load)
    sys.add_generator("cheap", bus=elec, capacity=60, marginal_cost=10)
    sys.add_generator("peak", bus=elec, capacity=200, marginal_cost=120)
    return sys


# ---------------------------------------------------------------------------
# (a) solver= kwarg routing
# ---------------------------------------------------------------------------


def test_solver_kwarg_highs_matches_default():
    sys_default = _tiny_lp()
    sys_forced = _tiny_lp()

    r_default = sys_default.optimise()
    r_forced = sys_forced.optimise(solver="highs")

    assert r_default.status == "optimal"
    assert r_forced.status == "optimal"
    assert r_forced.total_cost == pytest.approx(r_default.total_cost, rel=1e-6)


# ---------------------------------------------------------------------------
# (b) Clarabel direct adapter — SOC feasibility
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not clarabel_available(), reason="clarabel not installed")
def test_clarabel_socp_unit_disk():
    """
    min  -x - y   s.t.  ‖(x, y)‖₂ ≤ 1
    Optimum: x = y = 1/√2, obj = -√2.

    Adapter-level: uses ``add_soc(t_var, [u_vars])``; t_var pinned to 1 by an
    equality so the cone reads ``1 ≥ ‖(x, y)‖``.
    """
    prob = ConicProblem(n=3)  # vars: [x, y, t]
    X, Y, T = 0, 1, 2
    prob.add_linear_obj({X: -1.0, Y: -1.0})
    prob.add_eq({T: 1.0}, 1.0)              # pin t = 1
    prob.add_soc(t_var=T, u_vars=[X, Y])    # 1 ≥ ‖(x, y)‖

    res = prob.solve(eps=1e-9)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(-np.sqrt(2.0), abs=1e-5)
    assert res.x[X] == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-4)
    assert res.x[Y] == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-4)


# ---------------------------------------------------------------------------
# (c) SOCP AC-OPF relaxation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not clarabel_available(), reason="clarabel not installed")
def test_socp_opf_three_bus_radial():
    """
    Three buses in a line: gen → mid → load. Two branches with r, x > 0.
    Radial topology → relaxation is exact. Values are in per-unit on the
    system MVA base (the SOCP builder treats user amounts as pu; unit
    scaling is deferred to Phase 10.x).

    Check: status optimal; 0.95 ≤ |V| ≤ 1.05 everywhere; generator covers
    load plus branch losses; losses are strictly positive on a resistive
    network.
    """
    sys = ne.EnergySystem("socp_radial")
    b1 = sys.add_bus("b1", carrier="electricity")
    b2 = sys.add_bus("b2", carrier="electricity")
    b3 = sys.add_bus("b3", carrier="electricity")

    sys.add_generator("slack", bus=b1, capacity=5.0, marginal_cost=30.0)
    sys.add_load("d3", bus=b3, amount=0.5)

    l12 = sys.add_link("l12", bus_from=b1, bus_to=b2, capacity=2.0)
    l23 = sys.add_link("l23", bus_from=b2, bus_to=b3, capacity=2.0)
    for link in (l12, l23):
        link.resistance = 0.01
        link.reactance = 0.10
        link.model_type = "socp_opf"

    res = ne.solve_socp_opf(sys)

    assert res.status == "optimal"
    for name, v in res.voltage_mag.items():
        assert 0.949 <= v <= 1.051, f"|V[{name}]|={v} out of bounds"
    p_gen = res.gen_p["slack"]
    p_load = 0.5
    loss = sum(res.branch_loss.values())
    # Slack gen covers load + network losses (SOC-relaxation gives
    # loss ≥ r·I² which is strictly > 0 on a resistive radial).
    assert p_gen == pytest.approx(p_load + loss, abs=1e-4)
    assert loss > 0.0


def test_socp_opf_line_and_bus_shunts():
    """
    π-line shunt + bus shunt support (N_En_Phase 17.3 extension).

    Two-bus radial with a capacitive line shunt (b_fr = b_to > 0) and a
    capacitive bus shunt at the load end. The shunt injects reactive
    power at the load bus, which reduces the reactive dispatch required
    from the upstream generator. Compared to the no-shunt baseline:
      - both runs must be optimal,
      - generator real-power dispatch is within numerical tolerance
        (real losses dominated by r·|I|² are barely affected by shunt Q),
      - generator reactive dispatch on the shunt run is *strictly* less
        than on the no-shunt run (the shunts source Q locally).
    """
    def _build(b_line: float, b_bus: float) -> ne.EnergySystem:
        sys = ne.EnergySystem("socp_shunt")
        b1 = sys.add_bus("b1", carrier="electricity")
        b2 = sys.add_bus("b2", carrier="electricity")
        b2.b_shunt = b_bus
        sys.add_generator("g", bus=b1, capacity=5.0, marginal_cost=10.0)
        ld = sys.add_load("d", bus=b2, amount=0.5)
        ld.q_amount = 0.2
        link = sys.add_link("l", bus_from=b1, bus_to=b2, capacity=2.0)
        link.resistance = 0.01
        link.reactance = 0.10
        link.b_fr = b_line
        link.b_to = b_line
        link.model_type = "socp_opf"
        return sys

    baseline = ne.solve_socp_opf(_build(b_line=0.0, b_bus=0.0))
    with_shunt = ne.solve_socp_opf(_build(b_line=0.05, b_bus=0.1))

    assert baseline.status == "optimal"
    assert with_shunt.status == "optimal"

    # Real-power dispatch barely changes (shunts source Q, not P)
    assert with_shunt.gen_p["g"] == pytest.approx(baseline.gen_p["g"], rel=5e-3)

    # Reactive-power dispatch drops (shunts supplied some of it locally)
    assert with_shunt.gen_q["g"] < baseline.gen_q["g"] - 0.01


def test_socp_opf_transformer_tap_and_shift():
    """
    Transformer tap + phase shift (N_En_Phase 10.x extension, 2026-04-20).

    Three checks on a 2-bus radial:

      (a) ``tap=1, shift=0`` must reproduce the no-tap-attribute baseline
          bit-exactly — defaults have to be harmless (regression).
      (b) ``tap=1.10`` solves to optimal, balances power + feasible, and
          changes total cost relative to the baseline (proves the tap is
          wired into the Y-bus).
      (c) ``shift=0.05 rad`` solves to optimal and balances power — proves
          the phase-shift limb is wired into Y_ft / Y_tf.
    """
    def _build(**kwargs) -> ne.EnergySystem:
        sys = ne.EnergySystem("socp_tap")
        b1 = sys.add_bus("b1", carrier="electricity")
        b2 = sys.add_bus("b2", carrier="electricity")
        sys.add_generator("g", bus=b1, capacity=5.0, marginal_cost=10.0)
        sys.add_load("d", bus=b2, amount=0.3)
        link = sys.add_link("l", bus_from=b1, bus_to=b2, capacity=2.0)
        link.resistance = 0.01
        link.reactance = 0.10
        link.model_type = "socp_opf"
        for k, v in kwargs.items():
            setattr(link, k, v)
        return sys

    # (a) no-op regression — tap=1, shift=0 must be identical to "attribute absent"
    r_baseline = ne.solve_socp_opf(_build())
    r_noop = ne.solve_socp_opf(_build(tap=1.0, shift=0.0))
    assert r_baseline.status == "optimal"
    assert r_noop.status == "optimal"
    assert r_noop.total_cost == pytest.approx(r_baseline.total_cost, rel=1e-10)
    for bus in ("b1", "b2"):
        assert r_noop.voltage_mag[bus] == pytest.approx(
            r_baseline.voltage_mag[bus], abs=1e-9)

    # (b) tap != 1 — must solve, be feasible, and move the answer
    r_tap = ne.solve_socp_opf(_build(tap=1.10))
    assert r_tap.status == "optimal"
    assert 0.949 <= r_tap.voltage_mag["b1"] <= 1.051
    assert 0.949 <= r_tap.voltage_mag["b2"] <= 1.051
    loss_tap = sum(r_tap.branch_loss.values())
    assert r_tap.gen_p["g"] == pytest.approx(0.3 + loss_tap, abs=1e-4)
    # Stepping the tap changes the solution by more than solver tolerance.
    assert abs(r_tap.total_cost - r_baseline.total_cost) > 1e-4

    # (c) phase shift != 0 — must solve + balance
    r_shift = ne.solve_socp_opf(_build(shift=0.05))
    assert r_shift.status == "optimal"
    loss_shift = sum(r_shift.branch_loss.values())
    assert r_shift.gen_p["g"] == pytest.approx(0.3 + loss_shift, abs=1e-4)


def test_socp_opf_angle_diff_envelope():
    """
    Arctangent envelope (N_En_Phase 10.10, Kocuk et al. 2016 "SOCP+AT").

    Checks on a resistive 2-bus radial (Jabr is exact on radial, so the
    envelope is either inactive or drives to infeasibility — clean bar):

      (a) baseline — no kwarg vs ``angle_diff_max=None`` match bit-exactly
          (pre-10.10 behaviour preserved).
      (b) permissive envelope (π/3 ≈ 60°) is inactive on a 2-bus radial
          with realistic load — solution matches baseline.
      (c) per-link override — ``link.angle_diff_max`` beats the
          function-level default (the link's own wider bound wins).
      (d) pathologically tight envelope (0.005 rad ≈ 0.29°) is strictly
          below the feasible sending-angle for the given load and must
          flip the solver to non-optimal status (infeasible / dual_infeasible).
      (e) invalid ``angle_diff_max ≥ π/2`` raises at build time.
    """
    import math as _math
    def _build() -> ne.EnergySystem:
        sys = ne.EnergySystem("socp_atan")
        b1 = sys.add_bus("b1", carrier="electricity")
        b2 = sys.add_bus("b2", carrier="electricity")
        sys.add_generator("g", bus=b1, capacity=5.0, marginal_cost=10.0)
        sys.add_load("d", bus=b2, amount=0.5)
        link = sys.add_link("l", bus_from=b1, bus_to=b2, capacity=2.0)
        link.resistance = 0.05
        link.reactance = 0.20
        link.model_type = "socp_opf"
        return sys, link

    # (a) baseline — no argument vs explicit None must be identical.
    sys_a, _ = _build()
    sys_b, _ = _build()
    r_nokw = ne.solve_socp_opf(sys_a)
    r_none = ne.solve_socp_opf(sys_b, angle_diff_max=None)
    assert r_nokw.total_cost == pytest.approx(r_none.total_cost, rel=1e-12)

    # (b) permissive envelope — loose enough that the unconstrained
    #     optimum already satisfies it, so the answer is identical.
    sys_c, _ = _build()
    r_wide = ne.solve_socp_opf(sys_c, angle_diff_max=_math.pi / 3)
    assert r_wide.total_cost == pytest.approx(r_nokw.total_cost, rel=1e-6)

    # (c) per-link override — set a tight default but a wide per-link bound.
    sys_d, link_d = _build()
    link_d.angle_diff_max = _math.pi / 3
    link_d.angle_diff_min = -_math.pi / 3
    r_override = ne.solve_socp_opf(sys_d, angle_diff_max=0.001)
    # If the override took, we still solve optimally and match baseline.
    # If the tight function-level default leaked through, the solver would
    # fail (infeasibility at 0.001 rad on a 0.5-pu resistive line).
    assert r_override.status == "optimal"
    assert r_override.total_cost == pytest.approx(r_nokw.total_cost, rel=1e-6)

    # (d) tight envelope below the feasible sending-angle: no primal point.
    sys_e, _ = _build()
    r_tight = ne.solve_socp_opf(sys_e, angle_diff_max=0.005)
    assert r_tight.status != "optimal"

    # (e) invalid bound rejected at build time.
    sys_f, _ = _build()
    with pytest.raises(ValueError, match="angle_diff_max"):
        ne.solve_socp_opf(sys_f, angle_diff_max=_math.pi / 2)


@pytest.mark.skipif(not clarabel_available(), reason="clarabel not installed")
def test_socp_opf_cycle_closure():
    """
    QC-lite cycle closure (N_En_Phase 10.11).

    Layers per-branch ``θ_ij`` + sin-Taylor coupling + Σ±θ_ij=0 loop-closure
    on top of the Jabr SOCP. Coverage:

      (a) radial backward-compat — on a 2-bus radial the fundamental cycle
          basis is empty, so ``enforce_cycle_closure=True`` must not
          perturb the optimum.
      (b) triangle activation — on a 3-bus meshed case the cycle constraint
          is live (exactly one fundamental cycle) and the solver still
          returns an optimal dispatch with balanced power flow.
      (c) missing bound error — ``enforce_cycle_closure=True`` without any
          ``angle_diff_max`` raises a ValueError.
      (d) helper correctness — ``_fundamental_cycle_basis`` returns
          ``E - V + C`` cycles, handles parallel branches, and gives
          linearly consistent orientation signs.
    """
    import math as _math
    from nexus_energy.network_socp import _fundamental_cycle_basis

    # (a) radial backward-compat.
    def _build_radial():
        sys = ne.EnergySystem("qc_radial")
        b1 = sys.add_bus("b1", carrier="electricity")
        b2 = sys.add_bus("b2", carrier="electricity")
        sys.add_generator("g", bus=b1, capacity=5.0, marginal_cost=10.0)
        sys.add_load("d", bus=b2, amount=0.5)
        link = sys.add_link("l", bus_from=b1, bus_to=b2, capacity=2.0)
        link.resistance = 0.05
        link.reactance = 0.20
        link.model_type = "socp_opf"
        return sys

    r_base = ne.solve_socp_opf(_build_radial())
    r_qc = ne.solve_socp_opf(_build_radial(),
                             angle_diff_max=_math.pi / 6,
                             enforce_cycle_closure=True)
    assert r_qc.status == "optimal"
    assert r_qc.total_cost == pytest.approx(r_base.total_cost, rel=1e-6)

    # (b) triangle — exactly one fundamental cycle; solver stays optimal.
    def _build_triangle():
        sys = ne.EnergySystem("qc_tri")
        a = sys.add_bus("a", carrier="electricity")
        b = sys.add_bus("b", carrier="electricity")
        c = sys.add_bus("c", carrier="electricity")
        sys.add_generator("ga", bus=a, capacity=5.0, marginal_cost=10.0)
        sys.add_load("db", bus=b, amount=0.4)
        sys.add_load("dc", bus=c, amount=0.3)
        for (name, bf, bt) in [("ab", a, b), ("bc", b, c), ("ac", a, c)]:
            l = sys.add_link(name, bus_from=bf, bus_to=bt, capacity=2.0)
            l.resistance = 0.04
            l.reactance = 0.15
            l.model_type = "socp_opf"
        return sys

    r_tri = ne.solve_socp_opf(_build_triangle(),
                              angle_diff_max=_math.pi / 6,
                              enforce_cycle_closure=True)
    assert r_tri.status == "optimal"
    total_load = 0.4 + 0.3
    assert r_tri.gen_p["ga"] >= total_load - 1e-6  # must cover load + losses

    # (c) missing bound error.
    sys_err = _build_triangle()
    with pytest.raises(ValueError, match="angle_diff_max"):
        ne.solve_socp_opf(sys_err, enforce_cycle_closure=True)

    # (d1) helper on triangle — 3 links, 3 buses, 1 component ⇒ 1 cycle.
    sys_tri = _build_triangle()
    lines_tri = [l for l in sys_tri._links if l.model_type == "socp_opf"]
    cycles_tri = _fundamental_cycle_basis(lines_tri, sys_tri._buses)
    assert len(cycles_tri) == 1
    assert len(cycles_tri[0]) == 3
    # Every link appears exactly once in the single cycle.
    names_in_cycle = {link.name for (link, _) in cycles_tri[0]}
    assert names_in_cycle == {"ab", "bc", "ac"}
    # Signs sum to zero around the physical loop — traverse ab(+) → θ_A-θ_B,
    # bc(+) → θ_B-θ_C, ac(-) → θ_C-θ_A; the basis may choose different
    # orientation but must remain consistent (exactly one sign flip away).
    signs = {link.name: s for (link, s) in cycles_tri[0]}
    assert signs["ab"] * signs["bc"] * signs["ac"] == -1

    # (d2) helper on radial — empty basis.
    sys_rad = _build_radial()
    lines_rad = [l for l in sys_rad._links if l.model_type == "socp_opf"]
    assert _fundamental_cycle_basis(lines_rad, sys_rad._buses) == []

    # (d3) helper on parallel branches — 2 buses, 2 links ⇒ 1 cycle of size 2.
    sys_par = ne.EnergySystem("qc_par")
    b1 = sys_par.add_bus("b1", carrier="electricity")
    b2 = sys_par.add_bus("b2", carrier="electricity")
    sys_par.add_generator("g", bus=b1, capacity=5.0, marginal_cost=10.0)
    sys_par.add_load("d", bus=b2, amount=0.5)
    for name in ("l1", "l2"):
        l = sys_par.add_link(name, bus_from=b1, bus_to=b2, capacity=2.0)
        l.resistance = 0.05
        l.reactance = 0.20
        l.model_type = "socp_opf"
    lines_par = [l for l in sys_par._links if l.model_type == "socp_opf"]
    cycles_par = _fundamental_cycle_basis(lines_par, sys_par._buses)
    assert len(cycles_par) == 1
    assert len(cycles_par[0]) == 2
    s_par = {link.name: s for (link, s) in cycles_par[0]}
    # Same bus_from/bus_to on both parallel links ⇒ one traversed forward,
    # the other backward relative to the cycle direction.
    assert s_par["l1"] * s_par["l2"] == -1


@pytest.mark.skipif(not clarabel_available(), reason="clarabel not installed")
def test_socp_opf_tight_qc():
    """
    Tight QC with McCormick bilinears (N_En_Phase 10.12, Coffrin & Hijazi 2015).

    Upgrades the 10.11 QC-lite sin-Taylor coupling to the full
    Coffrin-Hijazi QC: per-bus ``|V|`` lifted with secant+tangent cuts on
    ``c_ii = |V|²``, per-branch ``w = |V_i|·|V_j|``, ``cos(θ_ij)`` and
    ``sin(θ_ij)`` lifted with trig envelopes, and the bilinears ``c_ij =
    w·cos θ``, ``s_ij = w·sin θ``, ``w = |V_i|·|V_j|`` bracketed by their
    McCormick envelopes. Coverage:

      (a) implication — ``enforce_tight_qc=True`` turns on cycle closure
          transparently (users shouldn't have to ask for both).
      (b) radial backward-compat — empty cycle basis on a 2-bus radial
          means the McCormick lift alone must not perturb the Jabr
          optimum beyond solver tolerance.
      (c) triangle activation — meshed 3-bus network stays optimal with
          the full QC lift live (11 extra variables + ~30 extra rows per
          branch must all be feasible at the solution).
      (d) missing bound error — ``enforce_tight_qc=True`` without any
          ``angle_diff_max`` raises a ValueError (same contract as 10.11).
      (e) tightness — cost with tight QC is ≥ cost with Jabr-only
          (relaxation monotonicity: more cuts can only raise a lower
          bound on the NLP optimum).
    """
    import math as _math

    def _build_radial():
        sys = ne.EnergySystem("tqc_radial")
        b1 = sys.add_bus("b1", carrier="electricity")
        b2 = sys.add_bus("b2", carrier="electricity")
        sys.add_generator("g", bus=b1, capacity=5.0, marginal_cost=10.0)
        sys.add_load("d", bus=b2, amount=0.5)
        link = sys.add_link("l", bus_from=b1, bus_to=b2, capacity=2.0)
        link.resistance = 0.05
        link.reactance = 0.20
        link.model_type = "socp_opf"
        return sys

    def _build_triangle():
        sys = ne.EnergySystem("tqc_tri")
        a = sys.add_bus("a", carrier="electricity")
        b = sys.add_bus("b", carrier="electricity")
        c = sys.add_bus("c", carrier="electricity")
        sys.add_generator("ga", bus=a, capacity=5.0, marginal_cost=10.0)
        sys.add_load("db", bus=b, amount=0.4)
        sys.add_load("dc", bus=c, amount=0.3)
        for (name, bf, bt) in [("ab", a, b), ("bc", b, c), ("ac", a, c)]:
            l = sys.add_link(name, bus_from=bf, bus_to=bt, capacity=2.0)
            l.resistance = 0.04
            l.reactance = 0.15
            l.model_type = "socp_opf"
        return sys

    # (a) + (b) radial backward-compat: tight QC on a radial must match Jabr.
    r_jabr = ne.solve_socp_opf(_build_radial())
    r_tight = ne.solve_socp_opf(_build_radial(),
                                angle_diff_max=_math.pi / 6,
                                enforce_tight_qc=True)
    assert r_tight.status == "optimal"
    assert r_tight.total_cost == pytest.approx(r_jabr.total_cost, rel=1e-4)

    # (c) triangle — full McCormick lift must stay feasible.
    r_tri_jabr = ne.solve_socp_opf(_build_triangle())
    r_tri_tight = ne.solve_socp_opf(_build_triangle(),
                                    angle_diff_max=_math.pi / 6,
                                    enforce_tight_qc=True)
    assert r_tri_tight.status == "optimal"
    assert r_tri_tight.gen_p["ga"] >= 0.7 - 1e-6  # covers load + losses

    # (e) monotonicity — more cuts ⇒ cost does not decrease.
    assert r_tri_tight.total_cost >= r_tri_jabr.total_cost - 1e-6

    # (d) missing bound error.
    with pytest.raises(ValueError, match="angle_diff_max"):
        ne.solve_socp_opf(_build_triangle(), enforce_tight_qc=True)


@pytest.mark.skipif(not clarabel_available(), reason="clarabel not installed")
def test_socp_opf_obbt():
    """OBBT pre-solve tightens v_mag / θ_ij boxes (N_En_Phase 10.14).

    Coverage:
      (a) Radial no-op — single-branch tree has no cycles and slack
          dispatch; OBBT should return quickly and should not widen
          any box (never-loosen invariant). Cost remains unchanged.
      (b) Triangle activation — meshed 3-bus with tight QC: at least
          one bus gets a v_mag shrink or one branch gets a θ_ij
          shrink; cost is monotone non-decreasing (box tightening
          can only raise relaxation cost).
      (c) Mutation semantics — bus.v_min / bus.v_max /
          link.angle_diff_min / link.angle_diff_max are updated in
          place on the passed-in system.
      (d) Missing-bound error — obbt_tighten with
          enforce_tight_qc=True on a triangle with no angle_diff_max
          raises (via the inner builder call).
      (e) End-to-end: `solve_socp_opf(..., enable_obbt=True)` produces
          an optimal result with the same `.total_cost ≥ baseline` shape.
    """
    import math as _math

    def _build_radial():
        sys = ne.EnergySystem("obbt_radial")
        b1 = sys.add_bus("b1", carrier="electricity")
        b2 = sys.add_bus("b2", carrier="electricity")
        b1.v_min, b1.v_max = 0.95, 1.05
        b2.v_min, b2.v_max = 0.95, 1.05
        sys.add_generator("g1", bus=b1, capacity=1.0, marginal_cost=5.0)
        sys.add_generator("g2", bus=b2, capacity=1.0, marginal_cost=20.0)
        sys.add_load("d2", bus=b2, amount=0.3)
        l12 = sys.add_link("l12", bus_from=b1, bus_to=b2, capacity=1.0)
        l12.resistance = 0.01
        l12.reactance = 0.1
        l12.s_max = 1.0
        l12.model_type = "socp_opf"
        return sys

    def _build_triangle():
        sys = ne.EnergySystem("obbt_tri")
        b1 = sys.add_bus("b1", carrier="electricity")
        b2 = sys.add_bus("b2", carrier="electricity")
        b3 = sys.add_bus("b3", carrier="electricity")
        for b in (b1, b2, b3):
            b.v_min, b.v_max = 0.95, 1.05
        sys.add_generator("ga", bus=b1, capacity=1.0, marginal_cost=5.0)
        sys.add_generator("gb", bus=b2, capacity=1.0, marginal_cost=30.0)
        sys.add_load("d3", bus=b3, amount=0.7)
        for (f, t, name) in ((b1, b2, "l12"), (b2, b3, "l23"), (b1, b3, "l13")):
            link = sys.add_link(name, bus_from=f, bus_to=t, capacity=1.0)
            link.resistance = 0.01
            link.reactance = 0.1
            link.s_max = 1.0
            link.model_type = "socp_opf"
        return sys

    # (a) Radial no-op: OBBT runs but never widens boxes.
    sys_r = _build_radial()
    bus_widths_before = {b.name: (b.v_min, b.v_max) for b in sys_r._buses}
    stats_r = ne.obbt_tighten(sys_r, max_iter=2, tol=1e-4,
                              angle_diff_max=_math.pi / 6,
                              enforce_tight_qc=True)
    assert stats_r.iters >= 1
    for b in sys_r._buses:
        v_min0, v_max0 = bus_widths_before[b.name]
        assert b.v_min >= v_min0 - 1e-9
        assert b.v_max <= v_max0 + 1e-9
    r_r = ne.solve_socp_opf(sys_r, angle_diff_max=_math.pi / 6,
                            enforce_tight_qc=True)
    assert r_r.status == "optimal"

    # (b) Triangle activation + (c) mutation semantics.
    sys_t = _build_triangle()
    r_baseline = ne.solve_socp_opf(_build_triangle(),
                                   angle_diff_max=_math.pi / 6,
                                   enforce_tight_qc=True)
    assert r_baseline.status == "optimal"

    bus_widths_before = {b.name: b.v_max - b.v_min for b in sys_t._buses}
    link_widths_before: dict[str, float] = {}
    for l in sys_t._links:
        t_max = getattr(l, "angle_diff_max", _math.pi / 6)
        t_min = getattr(l, "angle_diff_min", -t_max)
        link_widths_before[l.name] = t_max - t_min

    stats_t = ne.obbt_tighten(sys_t, max_iter=3, tol=1e-5,
                              angle_diff_max=_math.pi / 6,
                              enforce_tight_qc=True)
    assert stats_t.iters >= 1

    any_shrink = False
    for b in sys_t._buses:
        new_width = b.v_max - b.v_min
        if new_width < bus_widths_before[b.name] - 1e-6:
            any_shrink = True
    for l in sys_t._links:
        t_max = float(l.angle_diff_max)
        t_min = float(l.angle_diff_min)
        new_width = t_max - t_min
        if new_width < link_widths_before[l.name] - 1e-6:
            any_shrink = True
    assert any_shrink, "OBBT on triangle must shrink at least one box"

    r_tight_after = ne.solve_socp_opf(sys_t,
                                      angle_diff_max=_math.pi / 6,
                                      enforce_tight_qc=True)
    assert r_tight_after.status == "optimal"
    assert r_tight_after.total_cost >= r_baseline.total_cost - 1e-6

    # (d) Missing-bound error.
    sys_missing = _build_triangle()
    with pytest.raises(ValueError, match="angle_diff_max"):
        ne.obbt_tighten(sys_missing, enforce_tight_qc=True)

    # (e) End-to-end via solve_socp_opf(enable_obbt=True).
    sys_e2e = _build_triangle()
    r_e2e = ne.solve_socp_opf(sys_e2e, angle_diff_max=_math.pi / 6,
                              enforce_tight_qc=True,
                              enable_obbt=True, obbt_iters=2)
    assert r_e2e.status == "optimal"
    assert r_e2e.total_cost >= r_baseline.total_cost - 1e-6


def test_socp_opf_pwl_cos_envelope():
    """Piecewise-linear cos upper envelope (N_En_Phase 10.14.1).

    cos is concave on (-π/2, π/2), so every tangent line is a global
    upper bound. More tangents ⇒ tighter envelope ⇒ smaller feasible
    region ⇒ monotone non-decreasing SOCP relaxation cost. The chord
    lower bound is a strict improvement over the flat
    `cos_θ ≥ cos_lo` box bound used in 10.12.

    Coverage:
      (a) Invalid K raises (K=1 — must be ≥ 2).
      (b) K=2 recovers (approximately) the 10.12 endpoint-tangent
          behaviour; solve succeeds.
      (c) Triangle monotonicity: cost with K=4 ≥ K=2, K=8 ≥ K=4,
          K=16 ≥ K=8 (all within 1e-6 slack for solver noise).
      (d) End-to-end with OBBT + PWL cos: asymmetric post-OBBT
          intervals still produce an optimal solve at K=8.
    """
    import math as _math

    def _build_triangle():
        sys = ne.EnergySystem("pwl_cos_tri")
        b1 = sys.add_bus("b1", carrier="electricity")
        b2 = sys.add_bus("b2", carrier="electricity")
        b3 = sys.add_bus("b3", carrier="electricity")
        for b in (b1, b2, b3):
            b.v_min, b.v_max = 0.95, 1.05
        sys.add_generator("ga", bus=b1, capacity=1.0, marginal_cost=5.0)
        sys.add_generator("gb", bus=b2, capacity=1.0, marginal_cost=30.0)
        sys.add_load("d3", bus=b3, amount=0.7)
        for (f, t, name) in ((b1, b2, "l12"), (b2, b3, "l23"), (b1, b3, "l13")):
            link = sys.add_link(name, bus_from=f, bus_to=t, capacity=1.0)
            link.resistance = 0.01
            link.reactance = 0.1
            link.s_max = 1.0
            link.model_type = "socp_opf"
        return sys

    # (a) Invalid K raises.
    with pytest.raises(ValueError, match="cos_envelope_pieces"):
        ne.solve_socp_opf(_build_triangle(),
                          angle_diff_max=_math.pi / 6,
                          enforce_tight_qc=True,
                          cos_envelope_pieces=1)

    # (b/c) Monotone non-decreasing cost as K grows.
    costs: dict[int, float] = {}
    for K in (2, 4, 8, 16):
        r = ne.solve_socp_opf(_build_triangle(),
                              angle_diff_max=_math.pi / 6,
                              enforce_tight_qc=True,
                              cos_envelope_pieces=K)
        assert r.status == "optimal", f"K={K} failed: {r.status}"
        costs[K] = r.total_cost
    assert costs[4] >= costs[2] - 1e-6, f"K=4 {costs[4]} < K=2 {costs[2]}"
    assert costs[8] >= costs[4] - 1e-6, f"K=8 {costs[8]} < K=4 {costs[4]}"
    assert costs[16] >= costs[8] - 1e-6, f"K=16 {costs[16]} < K=8 {costs[8]}"

    # (d) OBBT + PWL cos end-to-end (K=8 default).
    sys_e2e = _build_triangle()
    r_e2e = ne.solve_socp_opf(sys_e2e, angle_diff_max=_math.pi / 6,
                              enforce_tight_qc=True,
                              cos_envelope_pieces=8,
                              enable_obbt=True, obbt_iters=2)
    assert r_e2e.status == "optimal"


def test_socp_opf_quadratic_cost():
    """Quadratic generator cost in the SOCP objective (N_En_Phase 10.4).

    `generator.quadratic_cost` adds a `q2·p²` term to the SOCP
    objective via `ConicProblem.add_quadratic_obj`. Coverage:
      (a) Opt-out default — no `quadratic_cost` attribute behaves
          bit-exactly like pre-10.4 (no regression).
      (b) Convexity guard — negative q2 raises.
      (c) Pure-quadratic dispatch split — two identical-linear gens
          with different q2 split load inverse to their q2, per the
          KKT condition `2·q2_i·p_i = 2·q2_j·p_j` (same dual).
      (d) Linear + quadratic parity with a hand-computed optimum on
          a one-bus, one-gen fixed-demand case.
    """
    # (a) Opt-out default — no quadratic_cost attribute, baseline holds.
    sys_a = ne.EnergySystem("qc_optout")
    b = sys_a.add_bus("b", carrier="electricity")
    b.v_min, b.v_max = 0.95, 1.05
    sys_a.add_generator("g", bus=b, capacity=10.0, marginal_cost=5.0)
    sys_a.add_load("d", bus=b, amount=0.5)
    # Radial 1-bus has no SOCP branch — add a tiny dummy branch to a
    # second bus with matched supply so the solver has a full SOCP.
    b2 = sys_a.add_bus("b2", carrier="electricity")
    b2.v_min, b2.v_max = 0.95, 1.05
    sys_a.add_generator("g2", bus=b2, capacity=10.0, marginal_cost=8.0)
    l12 = sys_a.add_link("l12", bus_from=b, bus_to=b2, capacity=10.0)
    l12.resistance = 0.01
    l12.reactance = 0.1
    l12.s_max = 10.0
    l12.model_type = "socp_opf"
    r_a = ne.solve_socp_opf(sys_a)
    assert r_a.status == "optimal"
    # Merit order: cheap gen "g" (mc=5) serves the 0.5 pu load.
    assert r_a.gen_p["g"] == pytest.approx(0.5, abs=1e-4)
    assert r_a.gen_p["g2"] == pytest.approx(0.0, abs=1e-4)

    # (b) Convexity guard.
    sys_b = ne.EnergySystem("qc_convex")
    bb = sys_b.add_bus("b", carrier="electricity")
    bb.v_min, bb.v_max = 0.95, 1.05
    bb2 = sys_b.add_bus("b2", carrier="electricity")
    bb2.v_min, bb2.v_max = 0.95, 1.05
    g_bad = sys_b.add_generator("g", bus=bb, capacity=10.0, marginal_cost=5.0)
    g_bad.quadratic_cost = -1.0
    sys_b.add_load("d", bus=bb, amount=0.1)
    sys_b.add_generator("g2", bus=bb2, capacity=10.0, marginal_cost=8.0)
    lb = sys_b.add_link("l", bus_from=bb, bus_to=bb2, capacity=10.0)
    lb.resistance = 0.01
    lb.reactance = 0.1
    lb.s_max = 10.0
    lb.model_type = "socp_opf"
    with pytest.raises(ValueError, match="quadratic_cost"):
        ne.solve_socp_opf(sys_b)

    # (c) Pure-quadratic split — same linear cost, different q2.
    # Hand calc: min q2a·pa² + q2b·pb² s.t. pa + pb = D, losses ~ 0.
    # Lagrangian ∂/∂pa = 2 q2a pa + λ = 0 → pa = -λ/(2 q2a); same for pb.
    # pa/pb = q2b/q2a. With q2a=1, q2b=3, D=0.4 → pa=0.3, pb=0.1.
    sys_c = ne.EnergySystem("qc_split")
    bc1 = sys_c.add_bus("b1", carrier="electricity")
    bc2 = sys_c.add_bus("b2", carrier="electricity")
    for bb_ in (bc1, bc2):
        bb_.v_min, bb_.v_max = 0.98, 1.02
    ga = sys_c.add_generator("ga", bus=bc1, capacity=5.0, marginal_cost=0.0)
    ga.quadratic_cost = 1.0
    gb = sys_c.add_generator("gb", bus=bc2, capacity=5.0, marginal_cost=0.0)
    gb.quadratic_cost = 3.0
    sys_c.add_load("da", bus=bc1, amount=0.2)
    sys_c.add_load("db", bus=bc2, amount=0.2)
    lc = sys_c.add_link("lc", bus_from=bc1, bus_to=bc2, capacity=5.0)
    lc.resistance = 1e-5  # tiny R → losses ≈ 0 so cost split stays clean
    lc.reactance = 0.05
    lc.s_max = 5.0
    lc.model_type = "socp_opf"
    r_c = ne.solve_socp_opf(sys_c)
    assert r_c.status == "optimal"
    # ga supplies more (cheaper marginal quadratic); ratio pa/pb ≈ q2b/q2a = 3.
    pa = r_c.gen_p["ga"]
    pb = r_c.gen_p["gb"]
    assert pa + pb == pytest.approx(0.4, abs=5e-3)  # total demand
    assert pa == pytest.approx(3.0 * pb, rel=2e-2), f"pa={pa}, pb={pb}"

    # (d) Linear + quadratic parity with hand-computed optimum.
    # Two gens on a 2-bus line serving 1.0 pu load on bus 2.
    # g1 (bus1): mc=10, q2=5. g2 (bus2): mc=20, q2=0.
    # Ignoring the tiny line loss: pick p1, p2 ≥ 0 with p1+p2 = 1.
    # Cost = 10·p1 + 5·p1² + 20·p2 = 10p1 + 5p1² + 20(1-p1)
    #      = 5p1² − 10p1 + 20. d/dp1 = 10p1 − 10 = 0 → p1 = 1.0, p2 = 0.
    # So p1 saturates at 1.0 and cost = 5 − 10 + 20 = 15.
    sys_d = ne.EnergySystem("qc_parity")
    bd1 = sys_d.add_bus("b1", carrier="electricity")
    bd2 = sys_d.add_bus("b2", carrier="electricity")
    for bb_ in (bd1, bd2):
        bb_.v_min, bb_.v_max = 0.98, 1.02
    g1 = sys_d.add_generator("g1", bus=bd1, capacity=2.0, marginal_cost=10.0)
    g1.quadratic_cost = 5.0
    sys_d.add_generator("g2", bus=bd2, capacity=2.0, marginal_cost=20.0)
    sys_d.add_load("d", bus=bd2, amount=1.0)
    ld = sys_d.add_link("ld", bus_from=bd1, bus_to=bd2, capacity=2.0)
    ld.resistance = 1e-5
    ld.reactance = 0.05
    ld.s_max = 2.0
    ld.model_type = "socp_opf"
    r_d = ne.solve_socp_opf(sys_d)
    assert r_d.status == "optimal"
    assert r_d.gen_p["g1"] == pytest.approx(1.0, abs=5e-3)
    assert r_d.gen_p["g2"] == pytest.approx(0.0, abs=5e-3)
    # Cost ≈ 10·1 + 5·1² + 20·0 = 15, ignoring ~1e-5 losses.
    assert r_d.total_cost == pytest.approx(15.0, abs=5e-3)


def test_socp_opf_multi_period_radial():
    """
    Multi-period SOCP AC-OPF (N_En_Phase 10.1).

    Same 3-bus radial as the single-snapshot test but with a 4-step load
    profile [0.3, 0.5, 0.7, 0.4] pu. Each period is fully separable in the
    absence of storage / ramping, so:
      - status optimal in every snapshot,
      - per-snapshot p_gen ≈ p_load + branch_loss,
      - aggregate total_cost equals dt * Σ_t per-snapshot total_cost,
      - higher-load snapshots draw more generator output (monotone in load).
    """
    sys = ne.EnergySystem("socp_radial_multi")
    T = 4
    sys.set_timesteps(T, dt=1.0)
    b1 = sys.add_bus("b1", carrier="electricity")
    b2 = sys.add_bus("b2", carrier="electricity")
    b3 = sys.add_bus("b3", carrier="electricity")

    sys.add_generator("slack", bus=b1, capacity=5.0, marginal_cost=30.0)
    load_profile = np.array([0.3, 0.5, 0.7, 0.4])
    sys.add_load("d3", bus=b3, amount=load_profile)

    l12 = sys.add_link("l12", bus_from=b1, bus_to=b2, capacity=2.0)
    l23 = sys.add_link("l23", bus_from=b2, bus_to=b3, capacity=2.0)
    for link in (l12, l23):
        link.resistance = 0.01
        link.reactance = 0.10
        link.model_type = "socp_opf"

    multi = ne.solve_socp_opf_multi(sys)

    assert multi.status == "optimal"
    assert len(multi.snapshots) == T

    per_snap_cost_sum = 0.0
    p_gen_series = []
    for t, snap in enumerate(multi.snapshots):
        assert snap.status == "optimal"
        assert 0.949 <= snap.voltage_mag["b1"] <= 1.051
        loss = sum(snap.branch_loss.values())
        assert snap.gen_p["slack"] == pytest.approx(load_profile[t] + loss, abs=1e-4)
        per_snap_cost_sum += snap.total_cost
        p_gen_series.append(snap.gen_p["slack"])

    # dt = 1.0 in this test → aggregate equals raw sum.
    assert multi.total_cost == pytest.approx(per_snap_cost_sum, rel=1e-9)
    # Series helpers expose per-snapshot views.
    assert multi.gen_p_series("slack") == p_gen_series
    # Generator output is monotone in load (no other coupling present).
    rank_load = np.argsort(load_profile)
    rank_gen = np.argsort(p_gen_series)
    assert list(rank_load) == list(rank_gen)


# ---------------------------------------------------------------------------
# (d) PWL quadratic transmission loss
# ---------------------------------------------------------------------------


def test_pwl_quadratic_loss_raises_cost_and_is_tight():
    """
    Two-bus transport system with cheap gen on bus_from, expensive slack on
    bus_to. Flow saturates the cheap generator, so loss on the inter-bus
    link is the dominant cost driver.

    With loss_quadratic > 0:
      - total cost strictly increases vs loss_quadratic = 0,
      - at the optimum f ≤ capacity, the PWL-lower-bound loss var equals
        loss_quadratic · f² to within ε (PWL with K breakpoints is exact at
        each breakpoint; cheap gen saturates at breakpoint k = K-1).
    """
    def _build(lq: float) -> ne.EnergySystem:
        sys = ne.EnergySystem(f"pwl_lq_{lq}")
        b1 = sys.add_bus("b1", carrier="electricity")
        b2 = sys.add_bus("b2", carrier="electricity")
        sys.add_generator("cheap", bus=b1, capacity=100.0, marginal_cost=10.0)
        sys.add_generator("slack", bus=b2, capacity=500.0, marginal_cost=200.0)
        sys.add_load("d2", bus=b2, amount=100.0)
        link = sys.add_link("l12", bus_from=b1, bus_to=b2, capacity=100.0)
        link.loss_quadratic = lq
        link.loss_pwl_breakpoints = 5
        return sys, link

    sys_lossless, _ = _build(0.0)
    sys_lossy, link_lossy = _build(5e-4)

    r_lossless = sys_lossless.optimise()
    r_lossy = sys_lossy.optimise()

    assert r_lossless.status == "optimal"
    assert r_lossy.status == "optimal"
    # Loss raises cost: the LP has to make up the loss with expensive slack.
    assert r_lossy.total_cost > r_lossless.total_cost + 1e-3
    # Loss variable at t=0: LP minimises it, and the PWL cuts are tangent
    # lines to y = lq·f² from below, so at the optimum 0 ≤ loss ≤ lq·f²
    # with equality attained whenever f lands on a breakpoint. Here cheap
    # saturates at 100 which is the last PWL breakpoint (K=5, Δ=25), so
    # equality is exact.
    raw = r_lossy._raw
    f0 = raw.value(link_lossy._flow_vars[0])
    loss0 = raw.value(link_lossy._loss_vars[0])
    expected = link_lossy.loss_quadratic * f0 * f0
    # On a breakpoint: PWL lower envelope = quadratic exactly.
    assert loss0 == pytest.approx(expected, abs=1e-6)


def test_pwl_loss_rejects_dc_opf_link():
    """loss_quadratic on a DC-OPF / PTDF link raises at build time — users
    should model AC losses via the SOCP relaxation, not stack approximations.
    """
    sys = ne.EnergySystem("pwl_reject")
    b1 = sys.add_bus("b1", carrier="electricity")
    b2 = sys.add_bus("b2", carrier="electricity")
    sys.add_generator("g", bus=b1, capacity=100.0, marginal_cost=10.0)
    sys.add_load("d", bus=b2, amount=50.0)
    link = sys.add_link("l", bus_from=b1, bus_to=b2, capacity=100.0)
    link.reactance = 0.1
    link.model_type = "dc_opf"
    link.loss_quadratic = 1e-4

    with pytest.raises(ValueError, match="loss_quadratic"):
        sys.optimise()


# ---------------------------------------------------------------------------
# (e) rolling_horizon_solve warm-start hook — LP no-op
# ---------------------------------------------------------------------------


def test_rolling_horizon_warm_start_lp_is_noop():
    """
    Pure-LP windows: ``warm_start=True`` and ``warm_start=False`` must
    produce identical dispatch. The hook is documented as MIP-only; LP basis
    I/O is deferred to Phase 10.x. This test pins that contract so a future
    regression (e.g. someone forwards a partial warm-start that accidentally
    perturbs the LP) fails loudly.
    """
    T_total = 24
    window = 6

    def factory(start, end):
        sys = ne.EnergySystem(f"rh_{start}_{end}")
        elec = sys.add_bus("elec", carrier="electricity")
        load = np.full(end - start, 90.0 + 10.0 * ((start // window) % 2))
        sys.add_load("d", bus=elec, amount=load)
        sys.add_generator("cheap", bus=elec, capacity=80, marginal_cost=10)
        sys.add_generator("peak", bus=elec, capacity=200, marginal_cost=120)
        return sys

    # Force simplex: warm-start IS a dual-simplex basis hot-start (Phase 10.6),
    # a vertex/basis concept the ipm_fast default cannot exercise — and IPM's
    # interior point is not bit-stable across the warm-start kwarg difference.
    cold = ne.rolling_horizon_solve(factory, T_total, window, warm_start=False,
                                    lp_backend="simplex")
    hot = ne.rolling_horizon_solve(factory, T_total, window, warm_start=True,
                                   lp_backend="simplex")

    assert cold["total_cost"] == pytest.approx(hot["total_cost"], rel=1e-9)
    for name in cold["generator_dispatch"]:
        np.testing.assert_allclose(
            cold["generator_dispatch"][name],
            hot["generator_dispatch"][name],
            rtol=1e-9,
            atol=1e-9,
        )
