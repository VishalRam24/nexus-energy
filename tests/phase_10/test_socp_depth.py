"""N_En_Phase 10.7 / 10.8 — conic-backend depth: Weymouth gas + head-dependent hydro.

Tiny Clarabel smoke tests for the two pressure/head-aware builders added on
top of the SOCP backend:

  * :func:`add_weymouth_pipe` — SOC / McCormick outer approximation of the
    Weymouth gas-flow equation ``q·|q| = K·(p₁² − p₂²)``.
  * :func:`add_head_dependent_hydro` — PWL convex-combination surrogate
    coupling discharge efficiency to reservoir SOC.

All instances are deliberately TINY (a single pipe / single reservoir).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nexus_energy._conic import ConicProblem, is_available as clarabel_available
from nexus_energy.network_socp import (
    add_weymouth_pipe,
    add_head_dependent_hydro,
    WeymouthVars,
    HydroHeadVars,
)


pytestmark = pytest.mark.skipif(
    not clarabel_available(), reason="clarabel not installed")


# ---------------------------------------------------------------------------
# Weymouth gas pipe
# ---------------------------------------------------------------------------


def test_weymouth_soc_relaxation_consistent():
    """Single pipe: fix the squared pressures, maximise the carried flow.

    With π_from, π_to pinned the SOC relaxation reads ``q² ≤ K·Δπ``, so the
    largest attainable forward flow is exactly ``q* = sqrt(K·Δπ)``. Maximise
    q (min −q) and check the optimum hits the analytic boundary — i.e. the
    SOC binds (relaxation is exact when pressure is fixed and flow is the
    only free variable).
    """
    K = 2.0
    pi1, pi2 = 9.0, 4.0            # p_from=3, p_to=2 → Δπ = 5
    q_star = math.sqrt(K * (pi1 - pi2))   # = sqrt(10)

    prob = ConicProblem(n=4)       # [q, pi_from, pi_to, w]
    q, pf, pt, w = 0, 1, 2, 3
    prob.add_eq({pf: 1.0}, pi1)
    prob.add_eq({pt: 1.0}, pi2)
    add_weymouth_pipe(prob, q=q, pi_from=pf, pi_to=pt, w=w,
                      weymouth_k=K, q_max=100.0)
    prob.add_linear_obj({q: -1.0})   # maximise q

    res = prob.solve(eps=1e-9)
    assert res.status == "optimal"
    assert res.x[q] == pytest.approx(q_star, abs=1e-4)
    # w lift binds at q² and at K·Δπ.
    assert res.x[w] == pytest.approx(q_star * q_star, abs=1e-3)


def test_weymouth_higher_flow_needs_more_pressure_drop():
    """Monotonicity: doubling the required flow needs ~4x the pressure drop.

    Minimise the upstream squared pressure π_from needed to push a fixed flow
    q0 to a pinned downstream π_to. Since q0² ≤ K·(π_from − π_to), the minimum
    is π_from = π_to + q0²/K, which scales with q0².
    """
    K = 1.5
    pi_to = 4.0

    def _min_pf(q0: float) -> float:
        prob = ConicProblem(n=4)
        q, pf, pt, w = 0, 1, 2, 3
        prob.add_eq({q: 1.0}, q0)
        prob.add_eq({pt: 1.0}, pi_to)
        add_weymouth_pipe(prob, q=q, pi_from=pf, pi_to=pt, w=w,
                          weymouth_k=K, q_max=100.0)
        prob.add_linear_obj({pf: 1.0})   # minimise upstream pressure²
        res = prob.solve(eps=1e-9)
        assert res.status == "optimal"
        return res.x[pf]

    pf1 = _min_pf(2.0)
    pf2 = _min_pf(4.0)
    assert pf1 == pytest.approx(pi_to + 2.0**2 / K, abs=1e-3)
    assert pf2 == pytest.approx(pi_to + 4.0**2 / K, abs=1e-3)
    # 4x flow² ⇒ 4x the pressure-drop term.
    assert (pf2 - pi_to) == pytest.approx(4.0 * (pf1 - pi_to), rel=1e-3)


def test_weymouth_mccormick_not_looser_than_soc():
    """McCormick variant adds secant upper cuts on w; it must not enlarge the
    feasible q for fixed pressures (a tighter-or-equal relaxation)."""
    K = 2.0
    pi1, pi2 = 9.0, 4.0

    def _max_q(relaxation: str) -> float:
        prob = ConicProblem(n=4)
        q, pf, pt, w = 0, 1, 2, 3
        prob.add_eq({pf: 1.0}, pi1)
        prob.add_eq({pt: 1.0}, pi2)
        add_weymouth_pipe(prob, q=q, pi_from=pf, pi_to=pt, w=w,
                          weymouth_k=K, q_max=10.0,
                          relaxation=relaxation, mccormick_segments=4)
        prob.add_linear_obj({q: -1.0})
        res = prob.solve(eps=1e-9)
        assert res.status == "optimal"
        return res.x[q]

    q_soc = _max_q("soc")
    q_mc = _max_q("mccormick")
    assert q_mc <= q_soc + 1e-4


def test_weymouth_bad_params():
    prob = ConicProblem(n=4)
    with pytest.raises(ValueError, match="weymouth_k"):
        add_weymouth_pipe(prob, q=0, pi_from=1, pi_to=2, w=3,
                          weymouth_k=-1.0, q_max=1.0)
    with pytest.raises(ValueError, match="q_max"):
        add_weymouth_pipe(prob, q=0, pi_from=1, pi_to=2, w=3,
                          weymouth_k=1.0, q_max=0.0)
    with pytest.raises(ValueError, match="relaxation"):
        add_weymouth_pipe(prob, q=0, pi_from=1, pi_to=2, w=3,
                          weymouth_k=1.0, q_max=1.0, relaxation="exact")


# ---------------------------------------------------------------------------
# Head-dependent hydro
# ---------------------------------------------------------------------------


def test_hydro_high_soc_gives_more_power_per_discharge():
    """Efficiency rises with SOC; fix discharge and SOC, maximise power.

    With η(SOC) increasing, the per-breakpoint McCormick cut caps power at the
    interpolated local efficiency. Pinning SOC at the top of the range and a
    fixed discharge, the attainable power equals η(soc_high)·Q.
    """
    def eta(s: float) -> float:
        # 0.7 at empty → 0.95 at full, linear in SOC.
        return 0.70 + 0.25 * s   # s in [0, 1]

    def _max_power(soc_fixed: float, q_fixed: float) -> float:
        prob = ConicProblem(n=3)   # [power, discharge, soc]
        P, Q, S = 0, 1, 2
        prob.add_eq({Q: 1.0}, q_fixed)
        prob.add_eq({S: 1.0}, soc_fixed)
        add_head_dependent_hydro(
            prob, power=P, discharge=Q, soc=S,
            soc_min=0.0, soc_max=1.0, eta_of_soc=eta,
            n_breakpoints=5, discharge_max=10.0)
        prob.add_le({P: -1.0}, 0.0)   # P ≥ 0
        prob.add_linear_obj({P: -1.0})   # maximise power
        res = prob.solve(eps=1e-9)
        assert res.status == "optimal"
        return res.x[P]

    p_full = _max_power(1.0, 5.0)
    p_empty = _max_power(0.0, 5.0)
    # At a breakpoint SOC the PWL is exact: P ≈ η(SOC)·Q.
    assert p_full == pytest.approx(eta(1.0) * 5.0, abs=1e-3)
    assert p_empty == pytest.approx(eta(0.0) * 5.0, abs=1e-3)
    # Higher head ⇒ strictly more power for the same water.
    assert p_full > p_empty + 0.1


def test_hydro_lambda_weights_form_convex_combination():
    """λ_k are a valid convex combination tying SOC to the grid."""
    def eta(s: float) -> float:
        return 0.8 + 0.1 * s

    prob = ConicProblem(n=3)
    P, Q, S = 0, 1, 2
    prob.add_eq({Q: 1.0}, 3.0)
    prob.add_eq({S: 1.0}, 0.5)      # mid-range SOC
    hv = add_head_dependent_hydro(
        prob, power=P, discharge=Q, soc=S,
        soc_min=0.0, soc_max=1.0, eta_of_soc=eta,
        n_breakpoints=4, discharge_max=10.0)
    prob.add_le({P: -1.0}, 0.0)
    prob.add_linear_obj({P: -1.0})
    res = prob.solve(eps=1e-9)
    assert res.status == "optimal"

    lam = np.array([res.x[i] for i in hv.breakpoint_weights])
    assert lam.sum() == pytest.approx(1.0, abs=1e-6)
    assert (lam >= -1e-7).all()
    grid = np.linspace(0.0, 1.0, 4)
    assert float(lam @ grid) == pytest.approx(0.5, abs=1e-4)


def test_hydro_bad_params():
    prob = ConicProblem(n=3)
    with pytest.raises(ValueError, match="n_breakpoints"):
        add_head_dependent_hydro(prob, power=0, discharge=1, soc=2,
                                 soc_min=0.0, soc_max=1.0,
                                 eta_of_soc=lambda s: 0.9, n_breakpoints=1)
    with pytest.raises(ValueError, match="soc_max"):
        add_head_dependent_hydro(prob, power=0, discharge=1, soc=2,
                                 soc_min=1.0, soc_max=1.0,
                                 eta_of_soc=lambda s: 0.9)
