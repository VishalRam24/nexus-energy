"""
Phase 11.2 — RL search heuristics (learn-to-fix / learn-to-order).

**Why model-level, not a solver branch callback.** A branch-and-bound
"learn to branch" policy hooks the solver's node callback to pick the
branching variable at each node. nexus_opt exposes *no* such
branch-callback API (see ``Model.solve`` — there is no node hook). So
the valid, callback-free path is to do RL search at the **model
level**: we repeatedly build the model, solve its LP relaxation, let a
policy choose which integer/binary variables to *fix* (and to what
value), re-solve, and learn the policy from the realised
objective/feasibility reward across episodes. This is the
"learn-to-fix / predict-and-search" family (Nair et al. 2020 "Neural
Diving"; Han et al. 2023 "GNN&GBDT predict-and-search"; Khalil et al.
2022) reduced to a contextual-bandit / tabular-Q controller so it
stays pure-numpy and torch-free, in the same spirit as
:class:`LearnedVarFixer` / :class:`VarFixingStats` in ``ml/varfix.py``.

The policy acts on a tiny per-variable feature vector computed from
the LP relaxation:

    - ``lp_value``        the relaxed value in [0, 1] (binary) / scaled,
    - ``fractionality``   ``min(v, 1-v)`` — distance to the nearest int,
    - ``reduced_cost``    objective-coefficient signal (cost to flip),
    - ``pseudo_cost``     running mean objective-degradation when this
                          variable was fixed (online, like B&B pseudo
                          costs).

Each feature is bucketed into a small discrete state; the policy keeps
a tabular Q-value per (state, action) where actions are
``{skip, fix_to_0, fix_to_1}``. Training is contextual-bandit style:
one episode = one full fix-and-resolve, the reward is the negative gap
to the true MILP optimum (penalised on infeasibility), and Q is
updated by an incremental sample-average toward the realised reward.

**Defining property (verified in the smoke test).** Over training
episodes the policy's *mean solution quality* — gap to the true MILP
optimum on a tiny instance we solve directly — improves / converges,
and the final returned solution is **never infeasible**: if the
policy's fixings make the model infeasible we fall back to the
unfixed full MILP solve (exactly the
``solve_with_warm_retry`` cold-fallback idiom).

The module is solver-coupled only through a ``model_builder`` callable
that returns a fresh ``(model, binary_vars)`` for a given fix dict, so
the energy model never has to expose a branch callback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
#
# A ``model_builder`` takes a dict ``{var_index: 0.0 | 1.0}`` of fixings and
# returns ``(model, ordered_binary_vars)`` where ``model`` is a nexus_opt
# ``Model`` already carrying the fixing constraints, and
# ``ordered_binary_vars`` is the list of the binary/integer decision Var
# handles in a stable order — the keys of the fix dict index into exactly
# this list. The builder must add an equality constraint ``var == value``
# for every entry in the fix dict (the model has no in-place bound
# mutation API). Indexing by position (not name) keeps the builder
# trivial: it just fixes ``bvars[i] == value``. Keeping this a pure
# callable means the RL search stays solver-callback-free and reusable
# across problems.

FixDict = dict[int, float]
ModelBuilder = Callable[[FixDict], "tuple[object, Sequence[object]]"]


@dataclass
class RLVarFixStats:
    """Running policy + reward statistics for the RL var-fixer.

    Mirrors :class:`nexus_energy.ml.varfix.VarFixingStats`: a small
    dataclass that accumulates the learnable state. Here that state is a
    tabular Q over ``(state_bucket, action)`` plus per-variable pseudo
    costs and the per-episode quality history used to verify
    convergence.

    Attributes:
        q: ``(n_states, 3)`` Q-value table; columns are the three
            actions ``skip / fix_to_0 / fix_to_1``.
        counts: ``(n_states, 3)`` visit counts (for the sample-average
            update step size).
        pseudo_cost: per-variable (by index) running mean objective
            degradation observed when that variable was fixed away from
            its LP value.
        pseudo_count: visit counts backing ``pseudo_cost``.
        gap_history: gap-to-optimum (fraction) of each training episode's
            realised solution — the convergence curve.
        feasible_history: per-episode bool, whether the fixed model
            solved without falling back.
    """
    n_buckets: int = 4
    q: np.ndarray = field(default=None)  # type: ignore[assignment]
    counts: np.ndarray = field(default=None)  # type: ignore[assignment]
    pseudo_cost: dict[int, float] = field(default_factory=dict)
    pseudo_count: dict[int, int] = field(default_factory=dict)
    gap_history: list[float] = field(default_factory=list)
    feasible_history: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        n_states = self.n_buckets ** 3  # lp_value, fractionality, reduced_cost
        if self.q is None:
            self.q = np.zeros((n_states, 3), dtype=float)
        if self.counts is None:
            self.counts = np.zeros((n_states, 3), dtype=int)


# Action encoding.
ACTION_SKIP = 0
ACTION_FIX0 = 1
ACTION_FIX1 = 2
_N_ACTIONS = 3


def _bucket(x: float, n: int, lo: float, hi: float) -> int:
    """Clip ``x`` to ``[lo, hi]`` and bucket into ``n`` even bins."""
    if hi <= lo:
        return 0
    f = (float(x) - lo) / (hi - lo)
    f = min(max(f, 0.0), 1.0 - 1e-12)
    return int(f * n)


class RLVarFixer:
    """Contextual-bandit / tabular-Q learn-to-fix policy for MILPs.

    Reuses the :class:`LearnedVarFixer` lifecycle shape — construct,
    ``train`` over episodes (the analogue of ``observe``), then
    ``solve`` (the analogue of ``predict`` + apply) — but the "fixing"
    decision is *learned by reward* rather than read off activation
    statistics.

    Typical lifecycle::

        fixer = RLVarFixer(seed=0)
        fixer.train(model_builder, n_episodes=40)   # learns the policy
        out = fixer.solve(model_builder)            # greedy fix + solve
        # out.objective is feasible-guaranteed (falls back to full solve)

    The policy never returns an infeasible final solution: if the
    greedy fixings are infeasible the fixer resolves the unfixed model
    (cold fallback) and reports ``fell_back=True``.
    """

    def __init__(
        self,
        *,
        n_buckets: int = 4,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.95,
        fix_fraction: float = 0.5,
        infeasible_penalty: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if not 0.0 < fix_fraction <= 1.0:
            raise ValueError("fix_fraction must be in (0, 1]")
        self.n_buckets = int(n_buckets)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.fix_fraction = float(fix_fraction)
        self.infeasible_penalty = float(infeasible_penalty)
        self.stats = RLVarFixStats(n_buckets=self.n_buckets)
        self._rng = np.random.default_rng(seed)
        self._true_opt: float | None = None

    # ---- Internal helpers ----------------------------------------------

    @staticmethod
    def _status_ok(result: object) -> bool:
        status = getattr(result, "status", None)
        obj = getattr(result, "objective", None)
        return status == "optimal" and obj is not None and np.isfinite(obj)

    def _solve_relaxation(
        self, model_builder: ModelBuilder, fix: FixDict
    ) -> "tuple[object, Sequence[object], object]":
        model, bvars = model_builder(dict(fix))
        result = model.solve(relax=True)
        return model, bvars, result

    def _solve_integer(
        self, model_builder: ModelBuilder, fix: FixDict
    ) -> "tuple[object, object]":
        model, _ = model_builder(dict(fix))
        result = model.solve()
        return model, result

    def _state_index(self, lp_val: float, frac: float, rcost: float) -> int:
        b = self.n_buckets
        # rcost is squashed to [0,1] by a logistic so its scale is robust.
        rc = 1.0 / (1.0 + np.exp(-rcost))
        i0 = _bucket(lp_val, b, 0.0, 1.0)
        i1 = _bucket(frac, b, 0.0, 0.5)
        i2 = _bucket(rc, b, 0.0, 1.0)
        return (i0 * b + i1) * b + i2

    def _var_features(
        self,
        bvars: Sequence[object],
        lp_result: object,
        coeffs: np.ndarray | None,
    ) -> list[tuple[int, float, float, float]]:
        """Return ``[(var_index, lp_value, fractionality, reduced_cost)]``."""
        feats: list[tuple[int, float, float, float]] = []
        for i, v in enumerate(bvars):
            lp_val = float(lp_result.value(v))
            lp_val = min(max(lp_val, 0.0), 1.0)
            frac = min(lp_val, 1.0 - lp_val)
            rcost = float(coeffs[i]) if coeffs is not None else 0.0
            feats.append((i, lp_val, frac, rcost))
        return feats

    def _choose_actions(
        self,
        feats: list[tuple[int, float, float, float]],
        *,
        explore: bool,
    ) -> tuple[FixDict, list[tuple[int, int]]]:
        """Pick actions per variable; return (fix_dict, [(state, action)]).

        Only the ``fix_fraction`` most-fractional variables are eligible
        for fixing (the rest are forced to SKIP) — fixing an already-
        near-integral LP value is both low-risk and low-information, so
        we spend the policy's budget on the genuinely fractional cells,
        echoing the ``max_fix_fraction`` cap in ``varfix.py``.
        """
        n = len(feats)
        order = sorted(range(n), key=lambda i: -feats[i][2])  # by fractionality
        cap = max(1, int(round(self.fix_fraction * n)))
        eligible = set(order[:cap])

        fix: FixDict = {}
        taken: list[tuple[int, int]] = []
        for i, lp_val, frac, rcost in feats:
            s = self._state_index(lp_val, frac, rcost)
            if i not in eligible:
                taken.append((s, ACTION_SKIP))
                continue
            if explore and self._rng.random() < self.epsilon:
                a = int(self._rng.integers(_N_ACTIONS))
            else:
                row = self.stats.q[s]
                # Tie-break deterministically toward SKIP (action 0).
                a = int(np.argmax(row))
            taken.append((s, a))
            if a == ACTION_FIX0:
                fix[i] = 0.0
            elif a == ACTION_FIX1:
                fix[i] = 1.0
        return fix, taken

    def _update_q(self, taken: list[tuple[int, int]], reward: float) -> None:
        for s, a in taken:
            self.stats.counts[s, a] += 1
            step = 1.0 / self.stats.counts[s, a]
            self.stats.q[s, a] += step * (reward - self.stats.q[s, a])

    # ---- Training -------------------------------------------------------

    def true_optimum(self, model_builder: ModelBuilder) -> float:
        """Solve the unfixed MILP once to obtain the reference optimum."""
        if self._true_opt is None:
            _, res = self._solve_integer(model_builder, {})
            if not self._status_ok(res):
                raise RuntimeError(
                    "RLVarFixer.true_optimum: base MILP is not optimal "
                    f"(status={getattr(res, 'status', None)!r})")
            self._true_opt = float(res.objective)
        return self._true_opt

    def train(
        self,
        model_builder: ModelBuilder,
        *,
        n_episodes: int = 40,
        coeffs: np.ndarray | None = None,
        sense: str = "min",
    ) -> RLVarFixStats:
        """Run ``n_episodes`` learn-to-fix episodes against ``model_builder``.

        Each episode: solve the LP relaxation of the *unfixed* model,
        compute per-variable features, sample fixing actions
        (epsilon-greedy), solve the resulting *integer* model, score the
        realised objective against the true optimum, and fold the reward
        into the tabular Q and the per-variable pseudo costs.

        ``coeffs`` (optional, ``(n_vars,)``) are the objective
        coefficients of the binary vars; used as the reduced-cost
        feature. ``sense`` is ``"min"`` or ``"max"`` and only sets the
        sign convention of the gap.

        Returns the live :class:`RLVarFixStats`. The episode gaps are in
        ``stats.gap_history`` (lower is better) for convergence checks.
        """
        if sense not in ("min", "max"):
            raise ValueError("sense must be 'min' or 'max'")
        opt = self.true_optimum(model_builder)
        denom = max(abs(opt), 1.0)

        # The LP relaxation features of the unfixed model are fixed across
        # episodes (same instance) — solve once, reuse.
        _, bvars, lp_res = self._solve_relaxation(model_builder, {})
        feats = self._var_features(bvars, lp_res, coeffs)

        for _ in range(n_episodes):
            fix, taken = self._choose_actions(feats, explore=True)
            _, res = self._solve_integer(model_builder, fix)
            if self._status_ok(res):
                realised = float(res.objective)
                gap = abs(realised - opt) / denom
                reward = -gap
                feasible = True
            else:
                # Infeasible fixings → strong negative reward.
                gap = float("inf")
                reward = -self.infeasible_penalty
                feasible = False
            self._update_q(taken, reward)
            # Pseudo-cost update for the variables we actually fixed.
            if feasible:
                for i in fix:
                    deg = gap  # objective degradation proxy
                    c = self.stats.pseudo_count.get(i, 0) + 1
                    prev = self.stats.pseudo_cost.get(i, 0.0)
                    self.stats.pseudo_cost[i] = prev + (deg - prev) / c
                    self.stats.pseudo_count[i] = c
            self.stats.gap_history.append(
                gap if np.isfinite(gap) else 1.0 + self.infeasible_penalty)
            self.stats.feasible_history.append(feasible)
            self.epsilon *= self.epsilon_decay
        return self.stats

    # ---- Inference ------------------------------------------------------

    def solve(
        self,
        model_builder: ModelBuilder,
        *,
        coeffs: np.ndarray | None = None,
    ) -> "RLSolveOutcome":
        """Greedy-fix using the learned policy, solve, never return infeasible.

        Solves the LP relaxation, applies the *greedy* (epsilon=0) policy
        fixings, solves the integer model. If that is infeasible, falls
        back to the unfixed full MILP solve. Returns an
        :class:`RLSolveOutcome` carrying the objective, the fix dict, and
        a ``fell_back`` flag.
        """
        _, bvars, lp_res = self._solve_relaxation(model_builder, {})
        feats = self._var_features(bvars, lp_res, coeffs)
        fix, _ = self._choose_actions(feats, explore=False)

        _, res = self._solve_integer(model_builder, fix)
        if self._status_ok(res):
            return RLSolveOutcome(
                objective=float(res.objective),
                fix=fix,
                fell_back=False,
                status=getattr(res, "status", "optimal"),
            )
        # Infeasible fixings → cold fallback (guaranteed feasible if the
        # base model is feasible at all).
        _, res = self._solve_integer(model_builder, {})
        return RLSolveOutcome(
            objective=float(res.objective) if self._status_ok(res)
            else float("nan"),
            fix={},
            fell_back=True,
            status=getattr(res, "status", "unknown"),
        )


@dataclass
class RLSolveOutcome:
    """Result of :meth:`RLVarFixer.solve`.

    Attributes:
        objective: realised objective of the accepted (feasible) solve.
        fix: the fix dict that was applied (empty if we fell back).
        fell_back: ``True`` if the policy's fixings were infeasible and
            we resolved the unfixed model.
        status: the solver status string of the accepted solve.
    """
    objective: float
    fix: FixDict
    fell_back: bool
    status: str


def solve_with_rl_search(
    model_builder: ModelBuilder,
    *,
    n_episodes: int = 40,
    coeffs: np.ndarray | None = None,
    fixer: RLVarFixer | None = None,
    seed: int = 0,
) -> "tuple[RLSolveOutcome, RLVarFixer]":
    """One-shot train-then-solve helper (mirrors :func:`apply_varfix`).

    Trains a fresh :class:`RLVarFixer` (unless one is passed) for
    ``n_episodes`` against ``model_builder`` and returns
    ``(outcome, fixer)``. The ``fixer.stats.gap_history`` is the
    convergence curve; ``outcome.fell_back`` reports whether the final
    greedy solve had to cold-fall-back.
    """
    if fixer is None:
        fixer = RLVarFixer(seed=seed)
    fixer.train(model_builder, n_episodes=n_episodes, coeffs=coeffs)
    outcome = fixer.solve(model_builder, coeffs=coeffs)
    return outcome, fixer
