"""
Phase 11.2 — RL search heuristics (learn-to-fix) smoke tests.

Coverage:
  (a) RLVarFixer.train runs episodes against a tiny knapsack MILP built
      via a callback-free model_builder and populates a convergence
      curve (gap_history);
  (b) DEFINING PROPERTY: over training the policy's MEAN gap-to-optimum
      improves (late-window mean < early-window mean) and converges
      toward the true MILP optimum solved directly;
  (c) RLVarFixer.solve never returns an infeasible solution — when the
      greedy fixings are infeasible it cold-falls-back to the unfixed
      full solve;
  (d) solve_with_rl_search one-shot helper returns a feasible outcome
      matching the true optimum on this tiny instance.

These are TINY instances (≤6 binaries) — no large sims, no full suite.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_opt as no

from nexus_energy.ml.rl_search import (
    RLVarFixer,
    RLSolveOutcome,
    solve_with_rl_search,
)


# ---------------------------------------------------------------------------
# Tiny 0/1 knapsack MILP + its callback-free model_builder
# ---------------------------------------------------------------------------
#
# max  Σ v_i x_i   s.t.  Σ w_i x_i <= cap,  x_i ∈ {0,1}
# The LP relaxation is fractional at the capacity boundary, so the
# learn-to-fix policy has real work to do.

_W = np.array([3.0, 4.0, 5.0, 2.0, 6.0])
_V = np.array([5.0, 6.0, 7.0, 3.0, 9.0])
_CAP = 10.0


def _knapsack_builder(fix):
    """Return (model, ordered_binary_vars) with `fix` applied as equalities."""
    m = no.Model("knap")
    xs = m.binaries("x", len(_W))
    m.add(sum(float(_W[i]) * xs[i] for i in range(len(_W))) <= _CAP)
    for i, val in fix.items():
        m.add(xs[i] == float(val))
    # Maximise value → minimise negative value (keep a single sense).
    m.minimize(sum(-float(_V[i]) * xs[i] for i in range(len(_W))))
    return m, xs


def _true_optimum():
    m, xs = _knapsack_builder({})
    r = m.solve()
    return float(r.objective)


# ---------------------------------------------------------------------------
# (a) + (b) training populates a converging gap curve
# ---------------------------------------------------------------------------

def test_rl_search_gap_improves_over_episodes():
    # Objective coeffs of the binaries = -V (we minimise -value).
    coeffs = -_V
    fixer = RLVarFixer(seed=0, epsilon=0.4, epsilon_decay=0.97,
                       fix_fraction=0.6)
    n_ep = 60
    fixer.train(_knapsack_builder, n_episodes=n_ep, coeffs=coeffs)

    gaps = np.array(fixer.stats.gap_history)
    assert gaps.shape[0] == n_ep
    assert np.all(np.isfinite(gaps))

    # DEFINING PROPERTY: mean solution quality improves over episodes.
    early = gaps[: n_ep // 3].mean()
    late = gaps[-n_ep // 3:].mean()
    print(f"\n[rl-search] early-mean-gap={early:.4f} late-mean-gap={late:.4f} "
          f"best-gap={gaps.min():.4f} true_opt={_true_optimum():.1f}")
    assert late <= early + 1e-9            # no regression
    assert late < early or late < 1e-6     # genuinely improved or already optimal
    # The policy found the true optimum (gap 0) at least once during search.
    assert gaps.min() < 1e-6


# ---------------------------------------------------------------------------
# (c) greedy solve never infeasible; matches true optimum here
# ---------------------------------------------------------------------------

def test_rl_search_solve_is_feasible_and_optimalish():
    coeffs = -_V
    fixer = RLVarFixer(seed=1, epsilon=0.4, epsilon_decay=0.97,
                       fix_fraction=0.6)
    fixer.train(_knapsack_builder, n_episodes=60, coeffs=coeffs)
    out = fixer.solve(_knapsack_builder, coeffs=coeffs)

    assert isinstance(out, RLSolveOutcome)
    assert out.status == "optimal"
    assert np.isfinite(out.objective)
    opt = _true_optimum()
    # Feasible-guaranteed; learned greedy fix should land on (or near) opt.
    print(f"\n[rl-search] greedy_obj={out.objective:.3f} true_opt={opt:.3f} "
          f"fell_back={out.fell_back} fixed={out.fix}")
    assert out.objective <= opt + 1e-6      # never better than true min
    assert out.objective <= opt + 1e-3      # learned solve is optimal here


# ---------------------------------------------------------------------------
# (c2) infeasible fixings trigger cold fallback, never infeasible result
# ---------------------------------------------------------------------------

def test_rl_search_falls_back_on_infeasible_fixings():
    # A tight builder where fixing too many vars to 1 is infeasible.
    w = np.array([6.0, 6.0, 6.0])
    cap = 6.0

    def builder(fix):
        m = no.Model("tight")
        xs = m.binaries("x", 3)
        m.add(sum(float(w[i]) * xs[i] for i in range(3)) <= cap)
        for i, val in fix.items():
            m.add(xs[i] == float(val))
        m.minimize(sum(-1.0 * xs[i] for i in range(3)))
        return m, xs

    fixer = RLVarFixer(seed=2)
    # Force an infeasible fix: pin all three to 1 (sum weight 18 > cap 6).
    bad_fix = {0: 1.0, 1: 1.0, 2: 1.0}
    _, res = fixer._solve_integer(builder, bad_fix)
    assert getattr(res, "status", None) == "infeasible"

    # The public solve() must still return a feasible objective.
    fixer.train(builder, n_episodes=20)
    out = fixer.solve(builder)
    assert out.status == "optimal"
    assert np.isfinite(out.objective)


# ---------------------------------------------------------------------------
# (d) one-shot helper
# ---------------------------------------------------------------------------

def test_solve_with_rl_search_oneshot():
    out, fixer = solve_with_rl_search(
        _knapsack_builder, n_episodes=50, coeffs=-_V, seed=3)
    assert isinstance(out, RLSolveOutcome)
    assert np.isfinite(out.objective)
    assert len(fixer.stats.gap_history) == 50
    assert out.objective <= _true_optimum() + 1e-3
