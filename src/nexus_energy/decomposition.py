"""
Phase 8 — Decomposition engine.

Real Benders decomposition (investment master + per-period operational
subproblems) with three stabilisation strategies:

- plain:         textbook multi-cut Benders.
- trust_region:  LP trust-region on cap_vars around the incumbent master
                 solution. HiGHS has no native quadratic regulariser; the
                 ℓ∞ trust-region box is the LP-friendly analogue of a
                 level-bundle stabiliser.
- adaptive:      adaptive-oracle Benders (Mazzi 2024): subproblems are
                 solved with a loose gap tolerance at first and tightened
                 as the master gap closes, so early iterations don't pay
                 full sub-LP cost. DIFFERENTIATOR vs GenX / SpineOpt.

Also provides:

- ``BendersDecomposer`` — the iteration driver.
- ``solve_with_temporal_benders`` — convenience wrapper: slice an
  ``EnergySystem`` into ``n_periods`` equal windows and run temporal
  Benders on them.
- ``temporal_decomposition`` — rolling-horizon (unchanged from Phase 5).
- ``recommend_decomposition`` — heuristic selector.

Phase-8 depth pass (this file) adds the deferred decomposition items:

- **8.4 Feasibility cuts (Farkas rays).** ``BendersDecomposer`` now emits
  proper Benders FEASIBILITY cuts when a subproblem is infeasible for the
  current master capacities. nexus_opt returns no dual / Farkas certificate
  for an infeasible LP (``Result.duals is None`` when ``status=="infeasible"``),
  so we use the *exact, solver-agnostic* Phase-1 route (Benders 1962 §3;
  the "feasibility / extreme-ray" cut). Each subsystem is made *elastic* by
  adding artificial unmet-/excess-energy generators on every bus; the
  Phase-1 LP minimises total artificial energy (a sum of non-negative
  slacks). If its optimum is > tol the master point is infeasible for the
  true subproblem, and the duals on the ``cap_var == cap_fixed`` pins are
  the components of the extreme dual ray. The resulting cut
  ``Σ_j r_j (cap_j - cap_fixed_j) ≤ -(phase1_opt)`` (equivalently
  ``Σ_j r_j cap_j ≤ Σ_j r_j cap_fixed_j - phase1_opt``) cuts off exactly the
  capacity vectors that keep the subproblem infeasible. This is the standard
  Phase-1 feasibility cut and is mathematically equivalent to a Farkas ray.

- **8.1 True spatial Benders.** ``solve_with_spatial_benders`` now performs
  real zonal decomposition: the master owns inter-zone tie-line flows (one
  signed flow var per tie-line per timestep) and each zone is an
  operational subproblem given fixed boundary injections; convergence uses
  optimality + the feasibility cuts above. Verified equal to the monolithic
  optimum on a tiny 2-zone instance.

- **8.2 Nested Benders.** ``solve_with_nested_benders`` recurses Benders
  across ≥2 coupling levels (Birge 1985 nested L-shaped): the level-0
  master proposes first-stage caps, the level-1 problem is itself a Benders
  master over its own sub-blocks, and cuts propagate up the chain.

- **8.3 Dantzig-Wolfe / column generation.** ``solve_with_dantzig_wolfe``
  (alias ``solve_with_column_generation``) solves a block-diagonal LP by a
  restricted master + per-block pricing subproblems (Dantzig & Wolfe 1960),
  generating columns until all reduced costs ≥ -tol. Verified equal to the
  direct LP optimum on a tiny block-diagonal instance.
"""

from __future__ import annotations

import copy
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

try:
    import nexus as nx
except ImportError:
    import nexus_opt as nx

if TYPE_CHECKING:
    from nexus_energy.core import EnergySystem


# ---------------------------------------------------------------------------
# Parallel-subproblem worker (Milestone 1 — multicore Benders).
#
# Benders subproblems are independent given the master capacities, so the
# subproblem pass is embarrassingly parallel. The Python GIL means thread
# pools give no speed-up across HiGHS solves (the solve path is
# Rust -> PyO3 -> highspy), so we use a *process* pool: each worker process
# has its own interpreter and its own HiGHS instance, so N workers keep N
# cores genuinely busy.
#
# The subsystems are pickled into each worker ONCE via the pool initializer
# and cached in a module global; each per-iteration task then ships only the
# small (period-index, caps, sub_gap) tuple, not the whole system. With
# heavyweight subproblems (full-year, multi-bus) this keeps per-iteration
# overhead negligible and the scaling near-linear up to the serial-master
# (Amdahl) ceiling.
# ---------------------------------------------------------------------------

_WORKER_SUBSYSTEMS: list | None = None


def _benders_worker_init(subsystems: list) -> None:
    """Pool initializer: cache the subsystem list in the worker process."""
    global _WORKER_SUBSYSTEMS
    _WORKER_SUBSYSTEMS = subsystems


def _benders_subproblem_task(arg: tuple) -> tuple[float, dict]:
    """Solve one operational subproblem in a worker process.

    Mirrors ``BendersDecomposer._solve_subproblem`` exactly so the parallel
    backend is bit-for-bit equivalent to the serial one.
    """
    p, caps, sub_gap = arg
    sub_system = _WORKER_SUBSYSTEMS[p]
    # threads=1: parallelism comes from N worker *processes*, so each solve must
    # stay single-threaded or the cores get oversubscribed (N×N threads).
    kwargs = {"benders_fix_caps": caps, "benders_skip_capex": True, "threads": 1}
    if sub_gap is not None:
        kwargs["gap"] = sub_gap
    res = sub_system.optimise(**kwargs)
    if res.status != "optimal":
        return float("inf"), {}
    return float(res.total_cost), dict(res.cap_dual)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BendersIteration:
    """Snapshot of one Benders iteration."""
    iteration: int
    upper_bound: float
    lower_bound: float
    gap: float
    master_capacities: dict[str, float]
    subproblem_costs: list[float]


@dataclass
class BendersResult:
    """Result of Benders decomposition."""
    status: str
    total_cost: float
    iterations: list[BendersIteration]
    final_capacities: dict[str, float]
    solve_time: float
    converged: bool
    # Phase 8 — sub-LPs solved so far (per-iteration * n_periods) for the
    # adaptive-oracle speedup claim. One subproblem per period per iter.
    sub_solves: int = 0
    # Milestone 1 — wall-clock spent in the subproblem pass (the parallelisable
    # kernel). Separating this from solve_time exposes the Amdahl ceiling set
    # by the serial master.
    subproblem_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cvar_at_caps(costs: list[float], probs: list[float], alpha: float) -> float:
    """
    Closed-form CVaR_α of a discrete distribution {(cost_i, prob_i)}.

    Sort losses descending, accumulate mass, and take the probability-weighted
    mean over the worst-α tail. Used by the decomposer to evaluate its
    upper bound under ``objective_mode="cvar"``.
    """
    if alpha <= 0:
        return max(costs)
    order = sorted(zip(costs, probs), key=lambda pair: -pair[0])
    remaining = float(alpha)
    weighted_sum = 0.0
    for c, p in order:
        if remaining <= 0:
            break
        take = min(p, remaining)
        weighted_sum += c * take
        remaining -= take
    return weighted_sum / alpha


def _collect_extendable_names(system: "EnergySystem") -> list[tuple[str, float, float, float]]:
    """
    Return ``(name, lower, upper, annualised_capex)`` for every extendable
    unit in ``system``. Storage contributes two entries (power + energy).
    """
    out: list[tuple[str, float, float, float]] = []
    big = 1e12
    for gen in system._generators:
        if gen.extendable:
            hi = gen.max_capacity if gen.max_capacity != float("inf") else big
            out.append((gen.name, float(gen.min_capacity), float(hi),
                        float(gen.capital_cost)))
    for sto in system._storages:
        if sto.extendable:
            pmax = sto.max_power_capacity \
                if sto.max_power_capacity != float("inf") else big
            emax = sto.max_energy_capacity \
                if sto.max_energy_capacity != float("inf") else big
            out.append((f"{sto.name}_power", float(sto.min_power_capacity),
                        float(pmax), float(sto.capital_cost_power)))
            out.append((f"{sto.name}_energy", float(sto.min_energy_capacity),
                        float(emax), float(sto.capital_cost_energy)))
    for link in system._links:
        if link.extendable:
            hi = link.max_capacity if link.max_capacity != float("inf") else big
            out.append((link.name, float(link.min_capacity), float(hi),
                        float(link.capital_cost)))
    return out


def _slice_system(system: "EnergySystem", start: int, end: int) -> "EnergySystem":
    """
    Deep-copy ``system`` and slice every time-indexed array to ``[start:end)``.
    Keeps ``extendable=True`` so cap_vars still exist for the Benders pin.
    Resets ephemeral solver state (cap_var / dispatch-var handles).
    """
    s = copy.deepcopy(system)
    s.name = f"{system.name}_win_{start}_{end}"
    length = end - start

    for gen in s._generators:
        if gen.carrier_factor is not None:
            gen.carrier_factor = np.asarray(gen.carrier_factor)[start:end]
        gen._cap_var = None
        gen._p_vars = []
        gen._u_vars = []
        gen._v_vars = []
        gen._w_vars = []
        gen._capex_seg_vars = []
        gen._capex_seg_slopes = []

    for sto in s._storages:
        if getattr(sto, "inflow", None) is not None \
                and isinstance(sto.inflow, np.ndarray):
            sto.inflow = sto.inflow[start:end]
        sto._cap_power_var = None
        sto._cap_energy_var = None
        sto._soc_vars = []
        sto._charge_vars = []
        sto._discharge_vars = []
        sto._spill_vars = []
        sto._soc_inter_vars = []
        # Subproblems are independent periods; disable cyclic/LDS plumbing.
        sto.cyclic = False
        sto.long_duration = False

    for link in s._links:
        link._cap_var = None
        link._flow_vars = []
        link._flow_rev_vars = []
        link._flow_signed_vars = []
        link._flow_out_vars = []
        link._inv_vars = []

    for ld in s._loads:
        if isinstance(ld.amount, np.ndarray):
            ld.amount = ld.amount[start:end]

    # Snapshot weights must follow the slice, or be cleared.
    if s._snapshot_weights is not None:
        s._snapshot_weights = np.asarray(s._snapshot_weights)[start:end]
    # Per-period Benders breaks inter-period Kotzur LDS — clear the mapping.
    s._chrono_mapping = None
    s._period_length = None

    s._timesteps = length
    return s


# Sentinel marginal cost for artificial unmet-energy generators. Large enough
# to never be used when the true subproblem is feasible, but finite so the
# elastic subproblem is always solvable. Used by the Phase-1 feasibility cut.
_ARTIFICIAL_PENALTY = 1e9


def _add_artificial_slacks(sub: "EnergySystem") -> list[str]:
    """
    Make ``sub`` *elastic*: add a single signed artificial generator on every
    bus so the operational LP is feasible for any capacity choice.

    A non-committable generator's dispatch ranges over ``[p_min, capacity]``
    (core.py:1053). Setting ``p_min = -BIG`` and ``capacity = +BIG`` therefore
    yields a free injection/absorption variable on the bus — it can cover an
    unmet demand (positive) or relieve an over-supply / flow imbalance
    (negative), both at marginal cost ``_ARTIFICIAL_PENALTY``.

    Returns the artificial-generator names. When the true subproblem is
    feasible the optimal dispatch never uses them (penalty ≫ real cost); the
    *Phase-1* objective (minimise total artificial energy) detects
    infeasibility exactly and its cap-pin duals give the feasibility-cut ray.
    """
    sub._infer_timesteps()
    names: list[str] = []
    for bus in list(sub._buses):
        nm = f"__art__{bus.name}"
        g = sub.add_generator(nm, bus=bus, capacity=1e12,
                              marginal_cost=_ARTIFICIAL_PENALTY)
        g.p_min = -1e12
        names.append(nm)
    return names


# ---------------------------------------------------------------------------
# Benders decomposer
# ---------------------------------------------------------------------------

class BendersDecomposer:
    """
    Investment-master / operational-subproblem Benders.

    The master problem holds the capacity decisions and a θ_p epigraph
    variable per period. Subproblems solve dispatch with ``benders_skip_capex``
    and pin cap_vars via ``benders_fix_caps``; their dual on the pin
    equality becomes the cut coefficient β_pj.

    Stabilisation:

    - ``stabilisation="trust_region"`` clips cap_vars to an ℓ∞-ball of
      radius ``trust_radius`` around the previous master solution. The
      radius halves if the upper bound stalls and doubles on steady
      descent — this is a linear substitute for level-bundle regularisation
      (HiGHS has no quadratic).
    - ``stabilisation="adaptive"`` solves subproblems with gap tolerance
      ``gap_init`` at iter 0 and linearly ramps to ``gap_final`` by
      iteration ``max_iter``. The adaptive-oracle argument: early cuts
      from coarse subproblems are valid lower-bounding hyperplanes; they
      refine as the master converges.
    """

    def __init__(
        self,
        system: "EnergySystem",
        periods: list[tuple[int, int]] | None = None,
        period_weights: list[float] | None = None,
        subsystems: list["EnergySystem"] | None = None,
        max_iter: int = 30,
        tol: float = 1e-3,
        stabilisation: str = "plain",
        trust_radius: float = 500.0,
        gap_init: float = 1e-2,
        gap_final: float = 1e-5,
        objective_mode: str = "expected",
        cvar_alpha: float = 0.05,
        feasibility_cuts: bool = False,
        feas_tol: float = 1e-4,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        system: the first-stage system. Its extendable cap_vars define
            the master; for stochastic use, this is the *nominal* system
            (cap meta lives here even though subproblems get their own
            scenario-parameterised copies).
        periods: list of ``(start, end)`` index ranges to slice ``system``
            into temporal subproblems. Mutually exclusive with ``subsystems``.
        subsystems: list of pre-built ``EnergySystem`` objects, one per
            subproblem. Used for Phase 9 stochastic where each subproblem
            is a scenario-parameterised copy of ``system``. Mutually
            exclusive with ``periods``.
        period_weights: probabilities / weights on each subproblem.
            Expected cost objective: ``obj = capex + Σ_p w_p θ_p``.
        objective_mode: ``"expected"`` (default) | ``"cvar"`` | ``"worst_case"``.
            - cvar: adds VaR var ``z`` + excess vars ``y_p ≥ θ_p - z``;
              ``obj = capex + z + (1/α) Σ w_p y_p``.
            - worst_case: adds ``θ_max`` with ``θ_max ≥ θ_p`` for all p;
              ``obj = capex + θ_max``.
        cvar_alpha: tail probability for CVaR (typ 0.05 = worst-5%).
        """
        if (periods is None) == (subsystems is None):
            raise ValueError(
                "BendersDecomposer requires exactly one of 'periods' or "
                "'subsystems'.")
        self.system = system
        self.periods = periods
        self.subsystems = subsystems
        self.n_periods = len(periods) if periods is not None else len(subsystems)
        if period_weights is None:
            period_weights = [1.0] * self.n_periods
        if len(period_weights) != self.n_periods:
            raise ValueError("period_weights must match n subproblems")
        self.period_weights = period_weights
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        if stabilisation not in ("plain", "trust_region", "adaptive"):
            raise ValueError(
                f"stabilisation must be one of 'plain' | 'trust_region' | "
                f"'adaptive' (got {stabilisation!r})"
            )
        if objective_mode not in ("expected", "cvar", "worst_case"):
            raise ValueError(
                f"objective_mode must be 'expected' | 'cvar' | 'worst_case' "
                f"(got {objective_mode!r})"
            )
        self.stabilisation = stabilisation
        self.trust_radius = float(trust_radius)
        self.gap_init = float(gap_init)
        self.gap_final = float(gap_final)
        self.objective_mode = objective_mode
        self.cvar_alpha = float(cvar_alpha)
        # 8.4 — proper Benders feasibility cuts. When True, subproblems are
        # made elastic (artificial slacks) and an infeasible master point
        # triggers a Phase-1 feasibility cut instead of a trust-region shrink.
        self.feasibility_cuts = bool(feasibility_cuts)
        self.feas_tol = float(feas_tol)
        # Milestone 1 — multicore Benders. n_jobs > 1 solves the subproblem
        # pass across a process pool (see module-level worker notes). n_jobs=1
        # keeps the original serial loop (bit-for-bit unchanged default).
        self.n_jobs = max(1, int(n_jobs))
        self._pool: ProcessPoolExecutor | None = None
        # Accumulated feasibility cuts: (ray: dict[name->r_j], rhs: float).
        # Encodes  Σ_j r_j · cap_j ≤ rhs.
        self._feas_cuts: list[tuple[dict[str, float], float]] = []
        self.verbose = bool(verbose)

        self._cap_info = _collect_extendable_names(system)
        if not self._cap_info:
            raise ValueError(
                "BendersDecomposer: system has no extendable capacity — "
                "Benders has nothing to decompose."
            )

    def _build_master(
        self,
        cuts: list[tuple[int, float, dict[str, float], dict[str, float]]],
        incumbent: dict[str, float] | None,
    ) -> tuple[object, dict[str, object], list[object]]:
        """Assemble the master LP. Returns (model, cap_vars, theta_vars)."""
        model = nx.Model(f"{self.system.name}_master")
        cap_vars: dict[str, object] = {}
        for name, lo, hi, _ in self._cap_info:
            lo_eff, hi_eff = lo, hi
            if self.stabilisation == "trust_region" and incumbent is not None \
                    and name in incumbent:
                prev = incumbent[name]
                lo_eff = max(lo, prev - self.trust_radius)
                hi_eff = min(hi, prev + self.trust_radius)
                if hi_eff < lo_eff:
                    # degenerate — back off to the full box for this var
                    lo_eff, hi_eff = lo, hi
            cap_vars[name] = model.variable(
                f"cap_{name}", lower=lo_eff, upper=hi_eff)

        # Investment cost term.
        obj = None
        for name, _, _, capex in self._cap_info:
            if capex != 0.0:
                term = capex * cap_vars[name]
                obj = term if obj is None else obj + term

        # θ_p: epigraph of each subproblem's operational cost. Lower bound = 0
        # is valid because operational cost is non-negative for sensible
        # input data (no negative marginal costs except via PTC, which is
        # a subsidy tallied separately).
        theta_vars: list[object] = []
        for p in range(self.n_periods):
            theta_vars.append(
                model.variable(f"theta_{p}", lower=0.0, upper=1e15))

        # Risk-aware objective (Phase 9).
        if self.objective_mode == "expected":
            for p in range(self.n_periods):
                term = self.period_weights[p] * theta_vars[p]
                obj = term if obj is None else obj + term
        elif self.objective_mode == "cvar":
            # Rockafellar-Uryasev: min z + (1/α) Σ w_p y_p
            #                      s.t. y_p ≥ θ_p - z, y_p ≥ 0
            z = model.variable("var_z", lower=-1e15, upper=1e15)
            y_vars = [
                model.variable(f"cvar_y_{p}", lower=0.0, upper=1e15)
                for p in range(self.n_periods)
            ]
            for p, y in enumerate(y_vars):
                model.add(y - theta_vars[p] + z >= 0.0, name=f"cvar_exc_{p}")
            term = z
            obj = term if obj is None else obj + term
            for p in range(self.n_periods):
                coef = self.period_weights[p] / self.cvar_alpha
                obj = obj + coef * y_vars[p]
        else:  # worst_case
            theta_max = model.variable("theta_max", lower=0.0, upper=1e15)
            for p in range(self.n_periods):
                model.add(theta_max - theta_vars[p] >= 0.0,
                          name=f"wc_bound_{p}")
            obj = theta_max if obj is None else obj + theta_max

        # Optimality cuts: θ_p >= α_p + Σ_j β_pj * (cap_j - cap_fixed_j)
        #                      = (α_p - Σ_j β_pj * cap_fixed_j) + Σ_j β_pj * cap_j
        for (p, alpha, beta, cap_fixed) in cuts:
            const = alpha
            for j, b in beta.items():
                const -= b * cap_fixed.get(j, 0.0)
            rhs = None
            for j, b in beta.items():
                if b == 0.0:
                    continue
                term = b * cap_vars[j]
                rhs = term if rhs is None else rhs + term
            if rhs is None:
                model.add(theta_vars[p] >= const, name=f"cut_{p}_a")
            else:
                model.add(theta_vars[p] - rhs >= const, name=f"cut_{p}")

        # 8.4 — feasibility cuts: Σ_j r_j · cap_j ≤ rhs. These contain no θ;
        # they purely restrict the capacity polytope to the region where every
        # subproblem is feasible (Benders 1962, the extreme-ray cut).
        for ci, (ray, rhs_val) in enumerate(self._feas_cuts):
            lhs = None
            for j, r in ray.items():
                if r == 0.0 or j not in cap_vars:
                    continue
                term = r * cap_vars[j]
                lhs = term if lhs is None else lhs + term
            if lhs is not None:
                model.add(lhs <= rhs_val, name=f"feascut_{ci}")

        if obj is not None:
            model.minimize(obj)
        return model, cap_vars, theta_vars

    def _solve_subproblem(
        self,
        p: int,
        caps: dict[str, float],
        sub_gap: float | None,
    ) -> tuple[float, dict[str, float]]:
        """Slice + solve sub-LP. Return (op_cost, cap_duals)."""
        if self.subsystems is not None:
            sub_system = self.subsystems[p]
        else:
            start, end = self.periods[p]
            sub_system = _slice_system(self.system, start, end)
        # NB: do NOT pin threads here. Passing threads=1 to a solve that runs
        # in the same process *after* the master solve trips a HiGHS thread-
        # state bug that returns a spurious non-optimal status. These sub-LPs
        # are simplex/serial (~1 core) regardless, so the serial baseline stays
        # honestly single-core. The process-pool worker pins threads=1 safely
        # because workers never solve the master. (Logged in progress_log.)
        kwargs = {
            "benders_fix_caps": caps,
            "benders_skip_capex": True,
        }
        if sub_gap is not None:
            kwargs["gap"] = sub_gap
        res = sub_system.optimise(**kwargs)
        if res.status != "optimal":
            return float("inf"), {}
        return float(res.total_cost), dict(res.cap_dual)

    def _materialise_all_subsystems(self) -> list:
        """All subproblem systems, ready to solve (sliced or caller-supplied).

        Built once and pickled into the worker pool so the per-iteration tasks
        only ship (period, caps, sub_gap).
        """
        if self.subsystems is not None:
            return list(self.subsystems)
        return [_slice_system(self.system, *self.periods[p])
                for p in range(self.n_periods)]

    def _subproblem_pass(
        self, caps: dict[str, float], sub_gap: float | None,
    ) -> list[tuple[float, dict[str, float]]]:
        """Solve every subproblem for the current master caps.

        Returns ``[(op_cost, cap_duals)]`` in period order. Dispatches to the
        process pool when ``n_jobs > 1`` (Milestone 1), else the serial loop.
        Both backends return identical results — the pool only changes *where*
        each independent solve runs, not *what* it computes.
        """
        t0 = time.perf_counter()
        if self._pool is not None:
            tasks = [(p, caps, sub_gap) for p in range(self.n_periods)]
            out = list(self._pool.map(_benders_subproblem_task, tasks))
        else:
            out = [self._solve_subproblem(p, caps, sub_gap)
                   for p in range(self.n_periods)]
        self._subproblem_seconds += time.perf_counter() - t0
        return out

    def _build_subsystem(self, p: int) -> "EnergySystem":
        """Materialise subproblem ``p`` (sliced window or pre-built scenario)."""
        if self.subsystems is not None:
            # Caller-supplied systems are reused across iterations; deep-copy so
            # adding artificial slacks for a Phase-1 pass never mutates them.
            return copy.deepcopy(self.subsystems[p])
        start, end = self.periods[p]
        return _slice_system(self.system, start, end)

    def _phase1_feasibility_cut(
        self,
        p: int,
        caps: dict[str, float],
    ) -> tuple[float, dict[str, float]] | None:
        """
        8.4 — Phase-1 feasibility cut for subproblem ``p`` at ``caps``.

        Build the elastic subproblem (artificial slacks on every bus), pin the
        cap_vars, and minimise the *total artificial energy* Σ_t Σ_b |art_b,t|
        (the Phase-1 / composite-infeasibility objective). nexus_opt returns no
        Farkas ray for a truly infeasible LP, so this elastic Phase-1 LP — which
        is always feasible — is the exact, solver-agnostic substitute:

          * If the Phase-1 optimum ``w*`` ≤ feas_tol the master point is
            feasible for the true subproblem → return ``None`` (caller adds an
            optimality cut instead).
          * Otherwise ``w* > 0`` and the duals ``r_j`` on the ``cap_j == fixed``
            pins are the gradient ∂w*/∂cap_j, i.e. the components of the dual
            extreme ray. The valid feasibility cut is

                w* + Σ_j r_j (cap_j − cap_fixed_j) ≤ 0
              ⇔ Σ_j r_j cap_j ≤ Σ_j r_j cap_fixed_j − w*

            which removes every capacity vector that leaves the subproblem
            infeasible (Benders 1962 §3; Geoffrion 1972 generalised cut).

        Returns ``(w_star, ray)`` to add as a feasibility cut, or ``None`` if
        the point is feasible.
        """
        sub = self._build_subsystem(p)
        art_names = _add_artificial_slacks(sub)

        # Phase-1 objective via model_hook: minimise Σ |artificial dispatch|.
        # Each artificial gen has one _p_vars entry per timestep; it may be
        # signed, so we linearise |p| with an auxiliary u ≥ p, u ≥ −p, min Σ u.
        def _phase1_objective(model, system, _obj):
            obj_terms = None
            for g in system._generators:
                if g.name not in art_names:
                    continue
                for v in g._p_vars:
                    u = model.variable(f"abs_{id(v)}", lower=0.0, upper=1e12)
                    model.add(u - v >= 0.0)
                    model.add(u + v >= 0.0)
                    obj_terms = u if obj_terms is None else obj_terms + u
            return obj_terms

        res = sub.optimise(
            benders_fix_caps=caps,
            benders_skip_capex=True,
            model_hook=_phase1_objective,
        )
        if res.status != "optimal":
            # Phase-1 LP itself failed to solve — cannot construct an exact
            # cut. Signal "could not certify" by returning None.
            return None
        w_star = float(res.total_cost)
        if w_star <= self.feas_tol:
            return None
        return w_star, dict(res.cap_dual)

    def _sub_gap_for(self, it: int) -> float | None:
        if self.stabilisation != "adaptive":
            return None
        if self.max_iter <= 1:
            return self.gap_final
        frac = min(1.0, it / (self.max_iter - 1))
        # Log-linear interpolation: coarse → tight.
        import math
        log_init = math.log10(max(self.gap_init, 1e-12))
        log_final = math.log10(max(self.gap_final, 1e-12))
        return 10.0 ** (log_init + (log_final - log_init) * frac)

    def solve(self) -> BendersResult:
        """Run Benders to convergence.

        When ``n_jobs > 1`` the subproblem pass runs on a process pool whose
        lifetime spans the whole solve: subsystems are pickled into the
        workers once here, and every iteration reuses them. The pool is always
        torn down (even on early return / exception) via the ``with`` block.
        """
        if self.n_jobs <= 1:
            return self._solve_impl()
        subsystems = self._materialise_all_subsystems()
        with ProcessPoolExecutor(
            max_workers=self.n_jobs,
            initializer=_benders_worker_init,
            initargs=(subsystems,),
        ) as pool:
            self._pool = pool
            try:
                return self._solve_impl()
            finally:
                self._pool = None

    def _solve_impl(self) -> BendersResult:
        t0 = time.perf_counter()
        self._subproblem_seconds = 0.0
        iterations: list[BendersIteration] = []
        cuts: list[tuple[int, float, dict[str, float], dict[str, float]]] = []
        incumbent_caps: dict[str, float] | None = None
        best_ub = float("inf")
        best_caps: dict[str, float] = {}
        stall_count = 0
        sub_solves = 0

        for it in range(self.max_iter):
            model, cap_vars, theta_vars = self._build_master(cuts, incumbent_caps)
            master_res = model.solve(verbose=False)
            if master_res.status != "optimal":
                return BendersResult(
                    status=f"master_{master_res.status}",
                    total_cost=float("nan"),
                    iterations=iterations,
                    final_capacities=best_caps,
                    solve_time=time.perf_counter() - t0,
                    converged=False,
                    sub_solves=sub_solves,
                )
            lb = float(master_res.objective)
            caps = {n: float(master_res.value(v)) for n, v in cap_vars.items()}

            # Investment cost of the current master capacities.
            inv_cost = 0.0
            for name, _, _, capex in self._cap_info:
                inv_cost += capex * caps[name]

            # Subproblem pass — independent given caps, so solved in parallel
            # across the process pool when n_jobs > 1 (results are identical to
            # the serial loop; only the placement of each solve changes).
            sub_gap = self._sub_gap_for(it)
            sub_costs: list[float] = []
            cap_duals_by_period: list[dict[str, float]] = []
            infeasible = False
            added_feas_cut = False
            results = self._subproblem_pass(caps, sub_gap)
            for p, (op_cost, beta) in enumerate(results):
                sub_solves += 1
                if not np.isfinite(op_cost):
                    if self.feasibility_cuts:
                        # 8.4 — extract a proper Phase-1 feasibility cut.
                        cut = self._phase1_feasibility_cut(p, caps)
                        sub_solves += 1
                        if cut is not None:
                            w_star, ray = cut
                            # Σ_j r_j cap_j ≤ Σ_j r_j cap_fixed_j − w*
                            rhs = -w_star
                            for j, r in ray.items():
                                rhs += r * caps.get(j, 0.0)
                            self._feas_cuts.append((ray, rhs))
                            added_feas_cut = True
                            if self.verbose:
                                print(f"[benders] iter {it}: period {p} "
                                      f"infeasible (w*={w_star:.3g}); added "
                                      f"feasibility cut {ray}")
                    infeasible = True
                    break
                sub_costs.append(op_cost)
                cap_duals_by_period.append(beta)

            if infeasible:
                if added_feas_cut:
                    # A new feasibility cut was added; re-solve master next iter.
                    continue
                # Fallback (no feasibility cuts requested or Phase-1 failed):
                # shrink the trust region and retry.
                if self.stabilisation == "trust_region" and self.trust_radius > 1.0:
                    self.trust_radius *= 0.5
                    if self.verbose:
                        print(f"[benders] iter {it}: subproblem infeasible, "
                              f"trust radius → {self.trust_radius:.2f}")
                    continue
                return BendersResult(
                    status="subproblem_infeasible",
                    total_cost=float("nan"),
                    iterations=iterations,
                    final_capacities=caps,
                    solve_time=time.perf_counter() - t0,
                    converged=False,
                    sub_solves=sub_solves,
                )

            # Aggregate sub-costs according to risk measure.
            if self.objective_mode == "expected":
                agg = sum(w * c for w, c in zip(self.period_weights, sub_costs))
            elif self.objective_mode == "worst_case":
                agg = max(sub_costs)
            else:  # cvar — Rockafellar-Uryasev closed-form at current caps
                agg = _cvar_at_caps(sub_costs, self.period_weights,
                                    self.cvar_alpha)
            ub_candidate = inv_cost + agg
            if ub_candidate < best_ub:
                best_ub = ub_candidate
                best_caps = caps.copy()
                stall_count = 0
            else:
                stall_count += 1

            gap = (best_ub - lb) / max(abs(best_ub), 1e-6)
            iterations.append(BendersIteration(
                iteration=it, upper_bound=best_ub, lower_bound=lb,
                gap=gap, master_capacities=caps.copy(),
                subproblem_costs=sub_costs.copy(),
            ))

            if self.verbose:
                print(f"[benders] iter {it}: LB={lb:.2f}  UB={best_ub:.2f}  "
                      f"gap={100*gap:.3f}%  stall={stall_count}  "
                      f"caps={ {k: round(v,1) for k,v in caps.items()} }")

            if gap <= self.tol:
                return BendersResult(
                    status="optimal", total_cost=best_ub,
                    iterations=iterations, final_capacities=best_caps,
                    solve_time=time.perf_counter() - t0, converged=True,
                    sub_solves=sub_solves,
                    subproblem_seconds=self._subproblem_seconds,
                )

            # Add one optimality cut per period: θ_p ≥ op_cost_p + Σ β_pj (cap_j - caps_j).
            for p, (op_cost, beta) in enumerate(zip(sub_costs, cap_duals_by_period)):
                cuts.append((p, op_cost, beta, caps.copy()))

            # Trust-region adjustment.
            if self.stabilisation == "trust_region":
                if stall_count >= 2 and self.trust_radius > 1.0:
                    self.trust_radius *= 0.5
                    stall_count = 0
                elif stall_count == 0:
                    self.trust_radius = min(self.trust_radius * 1.5, 1e12)

            incumbent_caps = caps

        return BendersResult(
            status="iteration_limit",
            total_cost=best_ub if np.isfinite(best_ub) else float("nan"),
            iterations=iterations,
            final_capacities=best_caps,
            solve_time=time.perf_counter() - t0,
            converged=False,
            sub_solves=sub_solves,
            subproblem_seconds=self._subproblem_seconds,
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def solve_with_temporal_benders(
    system: "EnergySystem",
    n_periods: int,
    period_length: int | None = None,
    **decomposer_kwargs,
) -> BendersResult:
    """
    Run Benders on ``n_periods`` contiguous, equal-length slices of ``system``.

    If ``period_length`` is omitted, it's inferred as ``T // n_periods``
    (the tail, if any, is appended to the last period).
    """
    system._infer_timesteps()
    T = system._timesteps
    if period_length is None:
        period_length = T // n_periods
    periods: list[tuple[int, int]] = []
    for p in range(n_periods):
        start = p * period_length
        end = start + period_length if p < n_periods - 1 else T
        periods.append((start, end))
    decomp = BendersDecomposer(system, periods, **decomposer_kwargs)
    return decomp.solve()


# ---------------------------------------------------------------------------
# 8.1 — True spatial (zonal) Benders
# ---------------------------------------------------------------------------

@dataclass
class SpatialBendersResult:
    """Result of zonal (spatial) Benders decomposition."""
    status: str
    total_cost: float
    iterations: list[BendersIteration]
    solve_time: float
    converged: bool
    sub_solves: int = 0
    # Final inter-zone tie-line flows: name -> np.ndarray of length T.
    tie_flows: dict[str, np.ndarray] = field(default_factory=dict)


def _build_zone_subsystem(
    system: "EnergySystem",
    zone: int,
    zone_of_bus: dict[str, int],
    elastic: bool,
) -> tuple["EnergySystem", dict[str, object]]:
    """
    Build the operational subsystem for ``zone``: all buses/components whose
    bus is in this zone, plus intra-zone links. Inter-zone (tie-line)
    coupling is injected later as fixed loads. Returns ``(subsystem,
    bus_map)``.
    """
    from nexus_energy.core import EnergySystem

    z = EnergySystem(f"{system.name}_zone_{zone}")
    z._dt = system._dt
    bus_map: dict[str, object] = {}
    for b in system._buses:
        if zone_of_bus[b.name] != zone:
            continue
        carrier = b.carrier.name if b.carrier.name in z._carriers else "electricity"
        bus_map[b.name] = z.add_bus(b.name, carrier=carrier)

    for g in system._generators:
        if g.bus.name not in bus_map:
            continue
        cf = np.asarray(g.carrier_factor) if g.carrier_factor is not None else None
        z.add_generator(
            g.name, bus=bus_map[g.bus.name], capacity=g.capacity,
            marginal_cost=g.marginal_cost, capital_cost=g.capital_cost,
            emission_factor=g.emission_factor, carrier_factor=cf,
            p_min=g.p_min, ramp_up=g.ramp_up, ramp_down=g.ramp_down,
            extendable=False,
        )
    for s in system._storages:
        if s.bus.name not in bus_map:
            continue
        z.add_storage(
            s.name, bus=bus_map[s.bus.name],
            power_capacity=s.power_capacity, energy_capacity=s.energy_capacity,
            efficiency_charge=s.efficiency_charge,
            efficiency_discharge=s.efficiency_discharge,
            self_discharge=s.self_discharge, soc_initial=s.soc_initial,
            cyclic=s.cyclic, marginal_cost=s.marginal_cost,
        )
    for l in system._links:
        if l.bus_from.name in bus_map and l.bus_to.name in bus_map:
            z.add_link(
                l.name, bus_from=bus_map[l.bus_from.name],
                bus_to=bus_map[l.bus_to.name], capacity=l.capacity,
                efficiency=l.efficiency, marginal_cost=l.marginal_cost,
                bidirectional=l.bidirectional,
            )
    for ld in system._loads:
        if ld.bus.name not in bus_map:
            continue
        z.add_load(ld.name, bus=bus_map[ld.bus.name], amount=ld.amount)

    if elastic:
        _add_artificial_slacks(z)
    z.set_timesteps(system._timesteps, dt=system._dt)
    return z, bus_map


def solve_with_spatial_benders(
    system: "EnergySystem",
    zone_of_bus: dict[str, int],
    max_iter: int = 50,
    tol: float = 1e-4,
    verbose: bool = False,
) -> SpatialBendersResult:
    """
    True zonal Benders (Benders 1962; Conejo et al. 2006 power-systems
    application). The complicating variables are the **inter-zone tie-line
    flows**, not capacities.

    Decomposition
    -------------
    * Master owns one signed flow ``f[tie, t]`` per tie-line per timestep
      (bounded by the tie-line's ±capacity), plus an epigraph ``θ_z`` per
      zone. ``min Σ_z θ_z`` subject to optimality + feasibility cuts.
    * Each zone subproblem receives the master flows as **fixed boundary
      injections**: a tie-line ``A→B`` carrying ``f_t`` is an export load
      ``+f_t`` at the sending bus (zone of ``A``) and an import injection
      ``+η·f_t`` (negative load) at the receiving bus (zone of ``B``).
    * The dual on each zone's bus-balance at the boundary bus and timestep
      (``result.bus_shadow_prices``) is exactly ∂(zone cost)/∂(injection),
      giving the **optimality-cut coefficient** on ``f[tie,t]``.
    * If a zone is infeasible for the proposed flows, an elastic Phase-1
      pass produces a **feasibility cut** on the flows (item 8.4 machinery).

    Convergence to the monolithic optimum is guaranteed because the flows
    fully parameterise the only coupling between zones (the tie-line
    balance), and Benders converges finitely for an LP master with a finite
    number of dual extreme points/rays.

    Single-zone systems fall through to a direct solve. Returns a
    :class:`SpatialBendersResult`.
    """
    from nexus_energy.core import EnergySystem  # noqa: F401

    system._infer_timesteps()
    T = system._timesteps
    dt = system._dt
    zones = sorted(set(zone_of_bus.values()))
    if len(zones) <= 1:
        direct = system.optimise()
        return SpatialBendersResult(
            status=direct.status, total_cost=direct.total_cost,
            iterations=[], solve_time=direct.solve_time, converged=True,
        )

    # Identify tie-lines (links crossing a zone boundary).
    ties = []  # (link, zone_from, zone_to)
    for l in system._links:
        zf = zone_of_bus[l.bus_from.name]
        zt = zone_of_bus[l.bus_to.name]
        if zf != zt:
            ties.append((l, zf, zt))
    if not ties:
        # Zones are decoupled — solve each independently and sum.
        total = 0.0
        for zi in zones:
            zsys, _ = _build_zone_subsystem(system, zi, zone_of_bus, elastic=False)
            r = zsys.optimise()
            if r.status != "optimal":
                return SpatialBendersResult(
                    status=f"zone_{zi}_{r.status}", total_cost=float("nan"),
                    iterations=[], solve_time=0.0, converged=False)
            total += r.total_cost
        return SpatialBendersResult(
            status="optimal", total_cost=total, iterations=[],
            solve_time=0.0, converged=True)

    snap_w = system._snapshot_weights
    w = np.ones(T) if snap_w is None else np.asarray(snap_w, dtype=float)

    t0 = time.perf_counter()
    # Master: build incrementally with accumulated cuts.
    # Optimality cuts: (zone, const, {(/tie,t): coef}, {(tie,t): f_fixed})
    opt_cuts: list[tuple[int, float, dict, dict]] = []
    feas_cuts: list[tuple[dict, float]] = []   # Σ coef·f ≤ rhs

    def flow_key(tie_idx, t):
        return (tie_idx, t)

    iterations: list[BendersIteration] = []
    sub_solves = 0
    best_ub = float("inf")
    best_flows: dict[str, np.ndarray] = {}

    for it in range(max_iter):
        m = nx.Model(f"{system.name}_spatial_master")
        # Flow vars per tie per timestep, bounded by ±capacity.
        fvars: dict[tuple[int, int], object] = {}
        for ti, (l, zf, zt) in enumerate(ties):
            cap = float(l.capacity)
            lo = -cap if l.bidirectional else 0.0
            for t in range(T):
                fvars[flow_key(ti, t)] = m.variable(
                    f"f_{l.name}_{t}", lower=lo, upper=cap)
        # θ_z ≥ 0: operational cost is non-negative (marginal costs ≥ 0;
        # artificial penalties are excluded from the real objective).
        theta = {zi: m.variable(f"theta_{zi}", lower=0.0, upper=1e15)
                 for zi in zones}
        obj = None
        for zi in zones:
            obj = theta[zi] if obj is None else obj + theta[zi]

        for (zi, const, coefs, f_fix) in opt_cuts:
            # θ_z ≥ const + Σ coef·(f − f_fix) = (const − Σ coef·f_fix) + Σ coef·f
            c = const
            for k, cf in coefs.items():
                c -= cf * f_fix.get(k, 0.0)
            rhs = None
            for k, cf in coefs.items():
                if cf == 0.0:
                    continue
                term = cf * fvars[k]
                rhs = term if rhs is None else rhs + term
            if rhs is None:
                m.add(theta[zi] >= c)
            else:
                m.add(theta[zi] - rhs >= c)

        for (coefs, rhs_val) in feas_cuts:
            lhs = None
            for k, cf in coefs.items():
                if cf == 0.0:
                    continue
                term = cf * fvars[k]
                lhs = term if lhs is None else lhs + term
            if lhs is not None:
                m.add(lhs <= rhs_val)

        m.minimize(obj)
        mres = m.solve(verbose=False)
        if mres.status != "optimal":
            return SpatialBendersResult(
                status=f"master_{mres.status}", total_cost=float("nan"),
                iterations=iterations, solve_time=time.perf_counter() - t0,
                converged=False, sub_solves=sub_solves)
        lb = float(mres.objective)
        flows = {k: float(mres.value(v)) for k, v in fvars.items()}

        # Subproblem pass over zones.
        zone_costs: dict[int, float] = {}
        zone_infeasible = None
        feas_cut_added = False
        for zi in zones:
            zsys, bus_map = _build_zone_subsystem(
                system, zi, zone_of_bus, elastic=False)
            # Inject boundary flows as fixed loads.
            _inject_boundary(zsys, bus_map, ties, zi, flows, T)
            r = zsys.optimise()
            sub_solves += 1
            if r.status != "optimal":
                # Feasibility cut via elastic Phase-1.
                ze, ze_map = _build_zone_subsystem(
                    system, zi, zone_of_bus, elastic=True)
                art = [g.name for g in ze._generators
                       if g.name.startswith("__art__")]
                _inject_boundary(ze, ze_map, ties, zi, flows, T)

                def _p1(model, sysz, _o, _art=set(art)):
                    terms = None
                    for g in sysz._generators:
                        if g.name not in _art:
                            continue
                        for v in g._p_vars:
                            u = model.variable(f"abs_{id(v)}", lower=0.0,
                                               upper=1e12)
                            model.add(u - v >= 0.0)
                            model.add(u + v >= 0.0)
                            terms = u if terms is None else terms + u
                    return terms

                rp = ze.optimise(model_hook=_p1)
                sub_solves += 1
                if rp.status != "optimal" or rp.total_cost <= tol:
                    zone_infeasible = zi
                    break
                w_star = float(rp.total_cost)
                # ∂w*/∂f via boundary bus shadow prices of the Phase-1 LP.
                coefs = _boundary_coefs(rp, ties, zi, zone_of_bus, w, dt)
                rhs = -w_star
                for k, cf in coefs.items():
                    rhs += cf * flows.get(k, 0.0)
                feas_cuts.append((coefs, rhs))
                feas_cut_added = True
                if verbose:
                    print(f"[spatial] iter {it}: zone {zi} infeasible "
                          f"(w*={w_star:.3g}) → feasibility cut")
                break
            zone_costs[zi] = float(r.total_cost)
            coefs = _boundary_coefs(r, ties, zi, zone_of_bus, w, dt)
            opt_cuts.append((zi, float(r.total_cost), coefs, dict(flows)))

        if zone_infeasible is not None and not feas_cut_added:
            return SpatialBendersResult(
                status=f"zone_{zone_infeasible}_infeasible",
                total_cost=float("nan"), iterations=iterations,
                solve_time=time.perf_counter() - t0, converged=False,
                sub_solves=sub_solves)
        if feas_cut_added:
            continue  # re-solve master with the new feasibility cut

        ub = sum(zone_costs.values())
        if ub < best_ub:
            best_ub = ub
            best_flows = {}
            for ti, (l, zf, zt) in enumerate(ties):
                best_flows[l.name] = np.array(
                    [flows[flow_key(ti, t)] for t in range(T)])
        gap = (best_ub - lb) / max(abs(best_ub), 1e-6)
        iterations.append(BendersIteration(
            iteration=it, upper_bound=best_ub, lower_bound=lb, gap=gap,
            master_capacities={}, subproblem_costs=list(zone_costs.values())))
        if verbose:
            print(f"[spatial] iter {it}: LB={lb:.2f} UB={best_ub:.2f} "
                  f"gap={100*gap:.3f}%")
        if gap <= tol:
            return SpatialBendersResult(
                status="optimal", total_cost=best_ub, iterations=iterations,
                solve_time=time.perf_counter() - t0, converged=True,
                sub_solves=sub_solves, tie_flows=best_flows)

    return SpatialBendersResult(
        status="iteration_limit",
        total_cost=best_ub if np.isfinite(best_ub) else float("nan"),
        iterations=iterations, solve_time=time.perf_counter() - t0,
        converged=False, sub_solves=sub_solves, tie_flows=best_flows)


def _inject_boundary(zsys, bus_map, ties, zone, flows, T):
    """Add fixed boundary-flow loads to zone ``zone``'s subsystem."""
    for ti, (l, zf, zt) in enumerate(ties):
        arr = np.array([flows[(ti, t)] for t in range(T)])
        if zf == zone:
            # Sending side exports f_t → looks like extra load +f_t.
            zsys.add_load(f"__tie_exp__{l.name}",
                          bus=bus_map[l.bus_from.name], amount=arr)
        if zt == zone:
            # Receiving side imports η·f_t → negative load (injection).
            zsys.add_load(f"__tie_imp__{l.name}",
                          bus=bus_map[l.bus_to.name],
                          amount=-float(l.efficiency) * arr)


def _boundary_coefs(result, ties, zone, zone_of_bus, w, dt):
    """
    Cut coefficient on each boundary flow var = ∂(zone cost)/∂f[tie,t].

    The export load at the sending bus contributes +shadow_price·(w·dt);
    the import (negative load) at the receiving bus contributes
    −η·shadow_price·(w·dt). ``bus_shadow_prices`` is already in $/MWh.
    """
    coefs: dict[tuple[int, int], float] = {}
    sp = result.bus_shadow_prices
    T = len(w)
    for ti, (l, zf, zt) in enumerate(ties):
        for t in range(T):
            c = 0.0
            if zf == zone:
                price = sp.get(l.bus_from.name)
                if price is not None and np.isfinite(price[t]):
                    c += float(price[t]) * w[t] * dt
            if zt == zone:
                price = sp.get(l.bus_to.name)
                if price is not None and np.isfinite(price[t]):
                    c += -float(l.efficiency) * float(price[t]) * w[t] * dt
            if c != 0.0:
                coefs[(ti, t)] = c
    return coefs


# ---------------------------------------------------------------------------
# 8.3 — Dantzig-Wolfe decomposition / column generation
# ---------------------------------------------------------------------------

@dataclass
class LPBlock:
    """
    One block of a block-diagonal LP, in the form

        minimise   c · x
        subject to D x  (≤ / = / ≥) d        (block-local constraints)
                   x ≥ 0                       (and optional upper bounds)

    plus a coupling contribution ``A x`` (rows shared across blocks). The
    block's feasible region must be a bounded polytope (so Dantzig-Wolfe's
    Minkowski representation uses extreme points only — no extreme rays).

    Attributes
    ----------
    c : (n,) objective coefficients.
    A : (m_couple, n) this block's columns in the coupling rows.
    D : (m_local, n) block-local constraint matrix.
    d : (m_local,) block-local RHS.
    sense : list of '<=' | '=' | '>=' for each local row.
    ub : optional (n,) upper bounds (default +∞; needed for boundedness if
         D alone doesn't bound x).
    """
    c: np.ndarray
    A: np.ndarray
    D: np.ndarray
    d: np.ndarray
    sense: list[str]
    ub: Optional[np.ndarray] = None


@dataclass
class DantzigWolfeResult:
    status: str
    objective: float
    iterations: int
    columns_generated: int
    converged: bool
    # Recovered block solutions x_k (convex combination of generated columns).
    block_solutions: list[np.ndarray] = field(default_factory=list)


def _solve_block_lp(
    block: LPBlock,
    cost: np.ndarray,
) -> tuple[str, float, np.ndarray]:
    """Solve ``min cost·x s.t. D x (sense) d, 0 ≤ x ≤ ub`` for one block."""
    m = nx.Model("dw_block")
    n = len(block.c)
    xs = []
    for j in range(n):
        hi = 1e12 if block.ub is None else float(block.ub[j])
        xs.append(m.variable(f"x{j}", lower=0.0, upper=hi))
    for i, s in enumerate(block.sense):
        lhs = None
        for j in range(n):
            a = float(block.D[i, j])
            if a == 0.0:
                continue
            term = a * xs[j]
            lhs = term if lhs is None else lhs + term
        if lhs is None:
            continue
        rhs = float(block.d[i])
        if s == "<=":
            m.add(lhs <= rhs)
        elif s == ">=":
            m.add(lhs >= rhs)
        else:
            m.add(lhs == rhs)
    obj = None
    for j in range(n):
        cj = float(cost[j])
        if cj == 0.0:
            continue
        term = cj * xs[j]
        obj = term if obj is None else obj + term
    if obj is not None:
        m.minimize(obj)
    else:
        m.minimize(0.0 * xs[0])
    r = m.solve(verbose=False)
    if r.status != "optimal":
        return r.status, float("nan"), np.zeros(n)
    x = np.array([float(r.value(v)) for v in xs])
    return "optimal", float(r.objective), x


def solve_with_dantzig_wolfe(
    blocks: list[LPBlock],
    coupling_rhs: np.ndarray,
    coupling_sense: list[str],
    max_iter: int = 200,
    tol: float = 1e-7,
    verbose: bool = False,
) -> DantzigWolfeResult:
    """
    Dantzig-Wolfe decomposition (Dantzig & Wolfe 1960) of a block-diagonal LP

        min   Σ_k c_k · x_k
        s.t.  Σ_k A_k x_k  (coupling_sense)  coupling_rhs     (coupling rows)
              x_k ∈ P_k = { D_k x ≤/=/≥ d_k, 0 ≤ x ≤ ub_k }   (block polytopes)

    Each bounded block polytope ``P_k`` is replaced by a convex combination
    of its extreme points (Minkowski–Weyl). The **restricted master** keeps a
    subset of those extreme points (columns) and solves

        min  Σ_k Σ_p (c_k·x_k^p) λ_k^p
        s.t. Σ_k Σ_p (A_k x_k^p) λ_k^p  (sense)  b          [duals π]
             Σ_p λ_k^p = 1   ∀k                              [duals σ_k]
             λ ≥ 0.

    **Pricing**: for block k the reduced cost of a new extreme point is
    ``(c_k − πᵀA_k)·x − σ_k``; we minimise ``(c_k − πᵀA_k)·x`` over ``P_k``.
    If the optimum < σ_k − tol the column improves the master and is added;
    when no block prices out (all reduced costs ≥ −tol) the master is optimal
    and equals the direct LP optimum (LP strong duality).

    Verified equal to the monolithic LP optimum on a tiny block-diagonal
    instance. Returns a :class:`DantzigWolfeResult`.
    """
    K = len(blocks)
    m_couple = len(coupling_rhs)
    for k, blk in enumerate(blocks):
        if blk.A.shape[0] != m_couple:
            raise ValueError(
                f"block {k}: A has {blk.A.shape[0]} coupling rows, "
                f"expected {m_couple}")

    # Generated columns per block: list of (x vector, c·x cost, A·x vector).
    cols: list[list[tuple[np.ndarray, float, np.ndarray]]] = [[] for _ in range(K)]

    # Seed each block with one feasible extreme point (minimise c_k over P_k).
    for k, blk in enumerate(blocks):
        st, cost_val, x = _solve_block_lp(blk, blk.c)
        if st != "optimal":
            return DantzigWolfeResult(
                status=f"block_{k}_{st}", objective=float("nan"),
                iterations=0, columns_generated=0, converged=False)
        cols[k].append((x, float(blk.c @ x), blk.A @ x))

    t_iter = 0
    converged = False
    last_pi = np.zeros(m_couple)
    for t_iter in range(1, max_iter + 1):
        # ---- Restricted master LP ----
        mm = nx.Model("dw_master")
        lam: list[list[object]] = []
        for k in range(K):
            lam.append([mm.variable(f"lam_{k}_{p}", lower=0.0, upper=1e12)
                        for p in range(len(cols[k]))])
        # Big-M artificials on every coupling row (DW Phase-1): a surplus and a
        # slack variable per row so the restricted master is ALWAYS feasible
        # even before enough columns exist to satisfy the coupling. Their huge
        # cost drives them to zero at optimality; while non-zero they keep the
        # duals π well-defined so pricing can find the columns that retire them.
        bigM = 1e9
        couple_idx: list[int] = []
        art_terms = []
        for i in range(m_couple):
            lhs = None
            for k in range(K):
                for p, (_, _, Ax) in enumerate(cols[k]):
                    a = float(Ax[i])
                    if a == 0.0:
                        continue
                    term = a * lam[k][p]
                    lhs = term if lhs is None else lhs + term
            # artificial surplus (subtract) + slack (add) to elasticise the row
            a_pos = mm.variable(f"art_pos_{i}", lower=0.0, upper=1e12)
            a_neg = mm.variable(f"art_neg_{i}", lower=0.0, upper=1e12)
            art_terms.extend([a_pos, a_neg])
            elastic = a_pos - a_neg
            lhs = elastic if lhs is None else lhs + elastic
            couple_idx.append(mm.num_constraints)
            rhs = float(coupling_rhs[i])
            s = coupling_sense[i]
            if s == "<=":
                mm.add(lhs <= rhs)
            elif s == ">=":
                mm.add(lhs >= rhs)
            else:
                mm.add(lhs == rhs)
        # Convexity rows: Σ_p λ_k^p = 1.
        conv_idx: list[int] = []
        for k in range(K):
            lhs = None
            for p in range(len(cols[k])):
                term = lam[k][p]
                lhs = term if lhs is None else lhs + term
            conv_idx.append(mm.num_constraints)
            mm.add(lhs == 1.0)
        # Objective.
        obj = None
        for k in range(K):
            for p, (_, cost_p, _) in enumerate(cols[k]):
                if cost_p == 0.0:
                    continue
                term = cost_p * lam[k][p]
                obj = term if obj is None else obj + term
        for at in art_terms:
            term = bigM * at
            obj = term if obj is None else obj + term
        if obj is not None:
            mm.minimize(obj)
        mr = mm.solve(verbose=False)
        if mr.status != "optimal":
            return DantzigWolfeResult(
                status=f"master_{mr.status}", objective=float("nan"),
                iterations=t_iter, columns_generated=sum(len(c) for c in cols),
                converged=False)
        duals = mr.duals
        pi = np.array([float(duals[couple_idx[i]]) for i in range(m_couple)]) \
            if duals is not None else np.zeros(m_couple)
        sigma = np.array([float(duals[conv_idx[k]]) for k in range(K)]) \
            if duals is not None else np.zeros(K)
        last_pi = pi

        # ---- Pricing: minimise (c_k − πᵀA_k)·x over P_k ----
        any_added = False
        for k, blk in enumerate(blocks):
            mod_cost = blk.c - blk.A.T @ pi
            st, val, x = _solve_block_lp(blk, mod_cost)
            if st != "optimal":
                return DantzigWolfeResult(
                    status=f"pricing_{k}_{st}", objective=float("nan"),
                    iterations=t_iter,
                    columns_generated=sum(len(c) for c in cols),
                    converged=False)
            reduced = val - sigma[k]
            if reduced < -tol:
                cols[k].append((x, float(blk.c @ x), blk.A @ x))
                any_added = True
        if verbose:
            print(f"[dw] iter {t_iter}: master_obj={mr.objective:.6f} "
                  f"cols={sum(len(c) for c in cols)} added={any_added}")
        if not any_added:
            converged = True
            # Recover block primal solutions x_k = Σ_p λ_k^p x_k^p.
            block_sol = []
            for k in range(K):
                xk = np.zeros(len(blocks[k].c))
                for p, (xp, _, _) in enumerate(cols[k]):
                    xk = xk + float(mr.value(lam[k][p])) * xp
                block_sol.append(xk)
            return DantzigWolfeResult(
                status="optimal", objective=float(mr.objective),
                iterations=t_iter,
                columns_generated=sum(len(c) for c in cols),
                converged=True, block_solutions=block_sol)

    return DantzigWolfeResult(
        status="iteration_limit", objective=float("nan"),
        iterations=t_iter, columns_generated=sum(len(c) for c in cols),
        converged=False)


# Alias — column generation is the same algorithm viewed from the master side.
solve_with_column_generation = solve_with_dantzig_wolfe


# ---------------------------------------------------------------------------
# 8.2 — Nested Benders decomposition
# ---------------------------------------------------------------------------

@dataclass
class NestedBendersResult:
    status: str
    objective: float
    iterations: int
    converged: bool
    # First-stage decision vector x0 at the optimum.
    first_stage: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class StageProblem:
    """
    One stage of a (linear) multistage chain

        min  c·x  s.t.  D x (sense) d − T·x_prev,   0 ≤ x ≤ ub

    where ``x_prev`` is the previous stage's decision. ``T`` is this stage's
    linking matrix multiplying the *previous* stage's variables (shape
    ``(m_local, n_prev)``); for the root stage ``T`` is ignored.
    """
    c: np.ndarray
    D: np.ndarray
    d: np.ndarray
    sense: list[str]
    T: Optional[np.ndarray] = None        # (m_local, n_prev)
    ub: Optional[np.ndarray] = None


def _solve_stage(
    stage: StageProblem,
    x_prev: Optional[np.ndarray],
    cuts: list[tuple[float, np.ndarray]],
    cost_to_go: bool,
) -> tuple[str, float, np.ndarray, np.ndarray]:
    """
    Solve one stage given the previous stage's decision ``x_prev``.

    If ``cost_to_go`` add an epigraph ``θ ≥ 0`` and the optimality cuts
    ``θ ≥ α + βᵀx`` (cuts on *this* stage's x). Returns
    ``(status, objective, x, rhs_dual)`` where ``rhs_dual`` is the vector of
    duals on the local rows (used to build the cut passed to the parent:
    coefficient on x_prev is ``−Tᵀ · rhs_dual``).
    """
    m = nx.Model("stage")
    n = len(stage.c)
    xs = [m.variable(f"x{j}", lower=0.0,
                     upper=(1e12 if stage.ub is None else float(stage.ub[j])))
          for j in range(n)]
    theta = None
    if cost_to_go:
        theta = m.variable("theta", lower=0.0, upper=1e15)

    # Effective RHS: d − T·x_prev.
    d_eff = np.array(stage.d, dtype=float)
    if stage.T is not None and x_prev is not None:
        d_eff = d_eff - stage.T @ x_prev

    row_idx: list[int] = []
    for i, s in enumerate(stage.sense):
        lhs = None
        for j in range(n):
            a = float(stage.D[i, j])
            if a == 0.0:
                continue
            term = a * xs[j]
            lhs = term if lhs is None else lhs + term
        if lhs is None:
            lhs = 0.0 * xs[0]
        row_idx.append(m.num_constraints)
        rhs = float(d_eff[i])
        if s == "<=":
            m.add(lhs <= rhs)
        elif s == ">=":
            m.add(lhs >= rhs)
        else:
            m.add(lhs == rhs)

    # Optimality cuts on θ (cost-to-go of the child stage): θ ≥ α + βᵀx.
    if cost_to_go:
        for (alpha, beta) in cuts:
            rhs_expr = None
            for j in range(n):
                b = float(beta[j])
                if b == 0.0:
                    continue
                term = b * xs[j]
                rhs_expr = term if rhs_expr is None else rhs_expr + term
            if rhs_expr is None:
                m.add(theta >= alpha)
            else:
                m.add(theta - rhs_expr >= alpha)

    obj = None
    for j in range(n):
        cj = float(stage.c[j])
        if cj == 0.0:
            continue
        term = cj * xs[j]
        obj = term if obj is None else obj + term
    if cost_to_go:
        obj = theta if obj is None else obj + theta
    if obj is not None:
        m.minimize(obj)

    r = m.solve(verbose=False)
    if r.status != "optimal":
        return r.status, float("nan"), np.zeros(n), np.zeros(len(stage.sense))
    x = np.array([float(r.value(v)) for v in xs])
    duals = r.duals
    rhs_dual = np.array(
        [float(duals[row_idx[i]]) if duals is not None else 0.0
         for i in range(len(row_idx))])
    return "optimal", float(r.objective), x, rhs_dual


def solve_with_nested_benders(
    stages: list[StageProblem],
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> NestedBendersResult:
    """
    Nested Benders (nested L-shaped) decomposition of a deterministic
    multistage LP chain (Birge 1985; Birge & Louveaux 2011 §7.1):

        min  Σ_s c_s · x_s
        s.t. D_0 x_0 (sense_0) d_0
             D_s x_s (sense_s) d_s − T_s x_{s-1}     s = 1 … S−1
             x_s ≥ 0.

    Requires ≥ 2 stages (≥ 1 coupling level). Each stage holds an epigraph
    ``θ_s`` for its successor's cost-to-go; a forward pass fixes
    ``x_0 → x_1 → …`` and a backward pass propagates optimality cuts
    ``θ_{s-1} ≥ α + βᵀ x_{s-1}`` where ``β = −T_sᵀ π_s`` (π_s = duals on
    stage-s rows) and ``α`` closes the cut at the current incumbent. The
    method recurses across every coupling level, so a 3-stage chain exercises
    two nested Benders levels. Converges (UB−LB ≤ tol) to the monolithic
    optimum.

    Returns a :class:`NestedBendersResult`.
    """
    S = len(stages)
    if S < 2:
        raise ValueError("nested Benders needs ≥ 2 stages (≥ 1 coupling level)")

    # Per-stage accumulated cuts on that stage's own x (θ of the stage that
    # *precedes* the cut's owner). cuts_for[s] are cuts added to stage s's θ.
    cuts_for: list[list[tuple[float, np.ndarray]]] = [[] for _ in range(S)]

    best_ub = float("inf")
    best_x0 = np.zeros(len(stages[0].c))

    for it in range(1, max_iter + 1):
        # ---- Forward pass: solve each stage given the previous decision. ----
        xs_path: list[np.ndarray] = []
        stage_objs: list[float] = []          # full objective incl. θ
        stage_local_cost: list[float] = []     # c_s·x_s only
        ok = True
        x_prev = None
        for s in range(S):
            has_child = s < S - 1
            st, sobj, x, _ = _solve_stage(
                stages[s], x_prev, cuts_for[s], cost_to_go=has_child)
            if st != "optimal":
                if s == 0:
                    return NestedBendersResult(
                        status=f"stage0_{st}", objective=float("nan"),
                        iterations=it, converged=False)
                # Forward infeasibility: in a deterministic feasible chain this
                # should not happen once relatively-complete; treat as failure.
                ok = False
                break
            xs_path.append(x)
            stage_objs.append(sobj)
            stage_local_cost.append(float(stages[s].c @ x))
            x_prev = x
        if not ok:
            return NestedBendersResult(
                status="forward_infeasible", objective=float("nan"),
                iterations=it, converged=False)

        # Lower bound = root objective (includes θ_0 = under-estimate of
        # downstream cost). Upper bound = sum of true local costs along path.
        lb = stage_objs[0]
        ub = sum(stage_local_cost)
        if ub < best_ub:
            best_ub = ub
            best_x0 = xs_path[0]
        gap = (best_ub - lb) / max(abs(best_ub), 1e-6)
        if verbose:
            print(f"[nested] iter {it}: LB={lb:.6f} UB={best_ub:.6f} "
                  f"gap={100*gap:.4f}%")
        if gap <= tol:
            return NestedBendersResult(
                status="optimal", objective=best_ub, iterations=it,
                converged=True, first_stage=best_x0)

        # ---- Backward pass: build optimality cuts stage S-1 → 0. ----
        # cost_to_go_value[s] = true cost from stage s onward at this path.
        ctg = [0.0] * S
        ctg[S - 1] = stage_local_cost[S - 1]
        for s in range(S - 2, -1, -1):
            ctg[s] = stage_local_cost[s] + ctg[s + 1]
        # Cut on stage s-1's θ from stage s: re-solve stage s at x_{s-1} to get
        # duals, β = −T_sᵀ π_s, α = (child cost-to-go) − βᵀ x_{s-1}.
        for s in range(S - 1, 0, -1):
            x_prev_s = xs_path[s - 1]
            has_child = s < S - 1
            st, sobj, _x, rhs_dual = _solve_stage(
                stages[s], x_prev_s, cuts_for[s], cost_to_go=has_child)
            if st != "optimal":
                return NestedBendersResult(
                    status=f"backward_stage{s}_{st}", objective=float("nan"),
                    iterations=it, converged=False)
            T_s = stages[s].T
            if T_s is None:
                beta = np.zeros(len(stages[s - 1].c))
            else:
                beta = -(T_s.T @ rhs_dual)
            # sobj is the optimal cost-to-go from stage s (incl. its own θ),
            # evaluated at x_{s-1}. Cut: θ_{s-1} ≥ sobj + βᵀ(x_{s-1} − x̂_{s-1})
            #                                   = (sobj − βᵀx̂_{s-1}) + βᵀ x_{s-1}.
            alpha = sobj - float(beta @ x_prev_s)
            cuts_for[s - 1].append((alpha, beta))

    return NestedBendersResult(
        status="iteration_limit",
        objective=best_ub if np.isfinite(best_ub) else float("nan"),
        iterations=max_iter, converged=False, first_stage=best_x0)


# ---------------------------------------------------------------------------
# Temporal decomposition (rolling horizon) — Phase 5, unchanged
# ---------------------------------------------------------------------------

def temporal_decomposition(
    system: "EnergySystem",
    window_size: int,
    overlap: int = 0,
    initial_soc_override: Optional[dict] = None,
    verbose: bool = False,
) -> dict:
    """
    Decompose a long-horizon problem into overlapping windows.

    For each window:
      1. Solve with the current initial SOC
      2. Extract the dispatch for the "keep" portion (excluding overlap)
      3. Use the final SOC as the initial SOC for the next window
    """
    from nexus_energy.core import EnergySystem

    system._infer_timesteps()
    T_total = system._timesteps
    if T_total <= window_size:
        direct = system.optimise()
        return {
            "status": direct.status,
            "total_cost": direct.total_cost,
            "generator_dispatch": direct.generator_dispatch,
            "storage_charge": direct.storage_charge,
            "storage_discharge": direct.storage_discharge,
            "storage_soc": direct.storage_soc,
            "link_flow": direct.link_flow,
            "n_windows": 1,
            "message": "Window >= total timesteps; solved directly.",
        }

    step = window_size - overlap
    starts = list(range(0, T_total, step))

    all_dispatch: dict = {}
    all_soc: dict = {}
    all_charge: dict = {}
    all_discharge: dict = {}
    all_link_flow: dict = {}
    total_cost = 0.0
    soc_carry = {s.name: s.soc_initial for s in system._storages}

    for i, start in enumerate(starts):
        end = min(start + window_size, T_total)
        if end <= start:
            break

        w = EnergySystem(f"{system.name}_window_{i}")
        bus_map = {}
        for b in system._buses:
            bb = w.add_bus(b.name,
                           carrier=b.carrier.name if b.carrier.name in w._carriers else "electricity")
            bus_map[b.name] = bb

        for g in system._generators:
            cf = None
            if g.carrier_factor is not None:
                cf = np.asarray(g.carrier_factor[start:end])
            w.add_generator(
                g.name, bus=bus_map[g.bus.name],
                capacity=g.capacity, marginal_cost=g.marginal_cost,
                capital_cost=g.capital_cost,
                emission_factor=g.emission_factor,
                carrier_factor=cf,
                p_min=g.p_min,
                ramp_up=g.ramp_up, ramp_down=g.ramp_down,
                extendable=False,
            )

        for s in system._storages:
            w.add_storage(
                s.name, bus=bus_map[s.bus.name],
                power_capacity=s.power_capacity,
                energy_capacity=s.energy_capacity,
                efficiency_charge=s.efficiency_charge,
                efficiency_discharge=s.efficiency_discharge,
                self_discharge=s.self_discharge,
                soc_initial=soc_carry[s.name],
                cyclic=False,
                marginal_cost=s.marginal_cost,
            )

        for l in system._links:
            w.add_link(
                l.name, bus_from=bus_map[l.bus_from.name],
                bus_to=bus_map[l.bus_to.name],
                capacity=l.capacity, efficiency=l.efficiency,
                marginal_cost=l.marginal_cost,
                bidirectional=l.bidirectional,
            )

        for ld in system._loads:
            amount = ld.amount
            if isinstance(amount, np.ndarray):
                amount = amount[start:end]
            w.add_load(ld.name, bus=bus_map[ld.bus.name], amount=amount)

        res = w.optimise()
        if res.status != "optimal":
            return {"status": "failed", "window": i, "result": res}

        keep_start = 0 if i == 0 else overlap
        keep_end = end - start

        for name, arr in res.generator_dispatch.items():
            all_dispatch.setdefault(name, []).append(arr[keep_start:keep_end])
        for name, arr in res.storage_charge.items():
            all_charge.setdefault(name, []).append(arr[keep_start:keep_end])
        for name, arr in res.storage_discharge.items():
            all_discharge.setdefault(name, []).append(arr[keep_start:keep_end])
        for name, arr in res.storage_soc.items():
            all_soc.setdefault(name, []).append(arr[keep_start:keep_end])
        for name, arr in res.link_flow.items():
            all_link_flow.setdefault(name, []).append(arr[keep_start:keep_end])

        for s in system._storages:
            final_soc_mwh = res.storage_soc[s.name][keep_end - 1]
            soc_carry[s.name] = final_soc_mwh / s.energy_capacity if s.energy_capacity > 1e-6 else 0.5

        total_cost += res.total_cost * (keep_end - keep_start) / (end - start)

        if verbose:
            print(f"Window {i}: t={start}..{end}, cost={res.total_cost:.2f}")

    return {
        "status": "optimal",
        "total_cost": total_cost,
        "generator_dispatch": {k: np.concatenate(v) for k, v in all_dispatch.items()},
        "storage_charge": {k: np.concatenate(v) for k, v in all_charge.items()},
        "storage_discharge": {k: np.concatenate(v) for k, v in all_discharge.items()},
        "storage_soc": {k: np.concatenate(v) for k, v in all_soc.items()},
        "link_flow": {k: np.concatenate(v) for k, v in all_link_flow.items()},
        "n_windows": len(starts),
    }


# ---------------------------------------------------------------------------
# Automatic decomposition strategy selector
# ---------------------------------------------------------------------------

def recommend_decomposition(system: "EnergySystem") -> str:
    """Heuristic decomposition-strategy suggestion for human readers."""
    T = system._timesteps
    n_buses = len(system._buses)
    n_components = system.n_components
    has_investments = any(g.extendable for g in system._generators) \
                     or any(s.extendable for s in system._storages) \
                     or any(l.extendable for l in system._links)

    problem_size = T * n_components

    lines = [f"Decomposition Recommendation for '{system.name}':"]
    lines.append(f"  Buses: {n_buses}, Components: {n_components}, Timesteps: {T}")
    lines.append(f"  Investment decisions: {'yes' if has_investments else 'no'}")
    lines.append(f"  Rough problem size: {problem_size:,} (components × timesteps)")
    lines.append("")

    if problem_size < 10_000:
        lines.append("→ Direct solve recommended (problem is small).")
    elif problem_size < 100_000:
        lines.append("→ Direct solve should work; consider presolve.")
    elif has_investments and T > 168:
        lines.append("→ Benders decomposition recommended:")
        lines.append("  - Master: investment decisions")
        lines.append("  - Subproblems: operational dispatch per representative period")
        lines.append("  - Use `solve_with_temporal_benders(system, n_periods)`.")
    elif T > 1000:
        lines.append("→ Temporal decomposition (rolling horizon) recommended:")
        lines.append("  - Use `temporal_decomposition()` with window_size=168 (1 week).")
    elif n_buses > 100:
        lines.append("→ Geographic decomposition (ADMM) recommended:")
        lines.append("  - Partition buses into regions, use ADMM for border flows.")
    else:
        lines.append("→ Time-series aggregation recommended:")
        lines.append("  - Reduce to representative days with k-medoids.")
        lines.append("  - See nexus_energy.temporal.aggregate_to_representative_days().")

    return "\n".join(lines)
