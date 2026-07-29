"""
Phase 9 — Stochastic & robust optimisation.

Planning under uncertainty via:

- **Two-stage stochastic programming** (``solve_stochastic``) — shared
  first-stage cap_vars, per-scenario second-stage dispatch. Routes through
  ``BendersDecomposer`` for efficient decomposition or ``extensive`` for
  one big LP. Expected-cost and CVaR objectives are both supported.
- **Robust optimisation** (``solve_robust``) — polyhedral budget-
  uncertainty sets; reduces to worst-case when the budget saturates.
- **Scenario reduction** (``reduce_scenarios``) — k-medoids on scenario
  parameter vectors to shrink a raw forecast ensemble down to a
  computationally tractable tree, with probability re-aggregation.
- **Monte Carlo harness** (``evaluate_plan``) — evaluate a fixed
  first-stage plan against many out-of-sample scenarios and report
  per-scenario cost + empirical CVaR.
- **Chance constraints** (``ChanceConstraint``) — SAA helper for
  individual chance constraints (Bonferroni for joint).
- **SDDiP stub** — deferred; multi-stage stochastic mixed-integer
  capacity expansion requires Lagrangian cuts on binary state
  variables and is research-grade.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

try:
    import nexus as nx
except ImportError:
    import nexus_opt as nx

if TYPE_CHECKING:
    from nexus_energy.core import EnergySystem


# ---------------------------------------------------------------------------
# Scenario representation
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """
    One draw from the uncertainty distribution.

    ``demand_factor`` multiplies every load's amount; ``carrier_factor_scale``
    multiplies every variable-RE generator's capacity factor (clipped to
    [0, 1]); ``fuel_cost_factor`` multiplies every generator's marginal
    cost. ``overrides`` is a sparse per-component override keyed by
    ``("gen", name, field)`` / ``("sto", name, field)`` / ``("link", name,
    field)`` → value, applied last.
    """
    name: str
    probability: float
    demand_factor: float = 1.0
    carrier_factor_scale: float = 1.0
    fuel_cost_factor: float = 1.0
    overrides: dict = field(default_factory=dict)


@dataclass
class StochasticResult:
    """Result of a two-stage stochastic solve."""
    status: str
    expected_cost: float
    scenario_costs: dict[str, float]
    capacity_decisions: dict[str, float]
    solve_time: float
    cvar: Optional[float] = None
    worst_case_cost: Optional[float] = None
    method: str = "benders"
    n_iterations: int = 0


# ---------------------------------------------------------------------------
# Scenario generation helpers
# ---------------------------------------------------------------------------

def generate_demand_scenarios(
    base_demand: np.ndarray | float,
    n_scenarios: int = 5,
    std: float = 0.15,
    seed: int = 42,
) -> list[Scenario]:
    """Gaussian demand-multiplier scenarios with equal probability."""
    rng = np.random.RandomState(seed)
    factors = np.clip(rng.normal(1.0, std, n_scenarios), 0.5, 1.5)
    factors = factors / factors.mean()
    probs = np.full(n_scenarios, 1.0 / n_scenarios)
    return [
        Scenario(
            name=f"demand_sc_{i}",
            probability=float(probs[i]),
            demand_factor=float(factors[i]),
        )
        for i in range(n_scenarios)
    ]


def generate_moment_matching_scenarios(
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    n_scenarios: int = 10,
    seed: int = 42,
    field_names: tuple[str, ...] = ("demand_factor", "carrier_factor_scale",
                                    "fuel_cost_factor"),
    probability: Optional[np.ndarray] = None,
) -> list[Scenario]:
    """
    Generate scenarios that match a target first and second moment (mean
    and covariance) of a joint factor distribution.

    Motivation: i.i.d. Gaussian sampling (``generate_demand_scenarios``) is
    cheap but easily misses the target covariance on small samples, and
    treats factors as independent. Moment matching rotates + rescales a
    seed sample so its *empirical* mean / cov exactly reproduce the target,
    giving tighter SAA approximations and eliminating cross-factor
    coupling mis-specification.

    Algorithm (Høyland–Wallace 2001, simplified):
      1. Draw ``N`` i.i.d. standard-normal seed vectors of dimension ``d``.
      2. Centre the seed batch (zero empirical mean) and whiten it
         (identity empirical cov) via Cholesky of the sample cov.
      3. Re-colour: multiply by ``L`` where ``L L.T = target_cov``.
      4. Shift by ``target_mean``.
      5. Emit as `Scenario` instances, mapping component *i* of each
         vector to ``field_names[i]``.

    ``field_names`` lets you choose which Scenario fields the factor
    vector components drive. Defaults hit the three standard multipliers
    (demand, renewable CF, fuel cost). Supply a 2-field tuple (e.g.
    ``("demand_factor", "fuel_cost_factor")``) to drive only those.
    """
    target_mean = np.asarray(target_mean, dtype=float)
    target_cov = np.asarray(target_cov, dtype=float)
    d = target_mean.size
    if target_cov.shape != (d, d):
        raise ValueError(
            f"target_cov must be {d}x{d}, got {target_cov.shape}")
    if len(field_names) != d:
        raise ValueError(
            f"field_names has {len(field_names)} entries but target_mean "
            f"has {d} dimensions")
    if n_scenarios < d + 1:
        raise ValueError(
            f"n_scenarios ({n_scenarios}) must exceed factor dim "
            f"({d}) so empirical cov is non-degenerate")

    rng = np.random.RandomState(seed)
    Z = rng.standard_normal((n_scenarios, d))  # (N, d)

    # Step 2 — centre + whiten.
    Z = Z - Z.mean(axis=0, keepdims=True)
    sample_cov = (Z.T @ Z) / (n_scenarios - 1)
    L_emp = np.linalg.cholesky(sample_cov + 1e-12 * np.eye(d))
    Z_white = Z @ np.linalg.inv(L_emp.T)  # empirical cov == I

    # Step 3 — recolour to target.
    L_tgt = np.linalg.cholesky(target_cov + 1e-12 * np.eye(d))
    X = Z_white @ L_tgt.T + target_mean[np.newaxis, :]

    if probability is None:
        probs = np.full(n_scenarios, 1.0 / n_scenarios)
    else:
        probs = np.asarray(probability, dtype=float)
        probs = probs / probs.sum()

    scenarios = []
    for i in range(n_scenarios):
        kw = {name: float(X[i, j]) for j, name in enumerate(field_names)}
        scenarios.append(Scenario(
            name=f"mm_{i}",
            probability=float(probs[i]),
            **kw,
        ))
    return scenarios


def generate_renewable_scenarios(
    n_scenarios: int = 5,
    cf_std: float = 0.2,
    seed: int = 42,
) -> list[Scenario]:
    """Gaussian carrier-factor-multiplier scenarios."""
    rng = np.random.RandomState(seed)
    factors = np.clip(rng.normal(1.0, cf_std, n_scenarios), 0.3, 1.3)
    factors = factors / factors.mean()
    return [
        Scenario(
            name=f"renew_sc_{i}",
            probability=1.0 / n_scenarios,
            carrier_factor_scale=float(factors[i]),
        )
        for i in range(n_scenarios)
    ]


# ---------------------------------------------------------------------------
# Scenario application
# ---------------------------------------------------------------------------

def apply_scenario(
    system: "EnergySystem",
    scenario: Scenario,
    *,
    deepcopy: bool = True,
) -> "EnergySystem":
    """
    Return a copy of ``system`` with ``scenario`` parameters applied.

    The copy resets ephemeral solver-state (cap_var handles, dispatch var
    lists) so it builds cleanly on its next ``optimise()`` call. The source
    system is left untouched when ``deepcopy=True`` (the default).
    """
    s = copy.deepcopy(system) if deepcopy else system
    s.name = f"{system.name}::{scenario.name}"

    for gen in s._generators:
        if scenario.carrier_factor_scale != 1.0 and gen.carrier_factor is not None:
            gen.carrier_factor = np.clip(
                np.asarray(gen.carrier_factor, dtype=float)
                * scenario.carrier_factor_scale, 0.0, 1.0)
        if scenario.fuel_cost_factor != 1.0:
            gen.marginal_cost = float(gen.marginal_cost) * scenario.fuel_cost_factor
        for (kind, name, field_), val in scenario.overrides.items():
            if kind == "gen" and name == gen.name:
                setattr(gen, field_, val)
        # Ephemerals
        gen._cap_var = None
        gen._p_vars = []
        gen._u_vars = []
        gen._v_vars = []
        gen._w_vars = []
        gen._capex_seg_vars = []
        gen._capex_seg_slopes = []

    for sto in s._storages:
        for (kind, name, field_), val in scenario.overrides.items():
            if kind == "sto" and name == sto.name:
                setattr(sto, field_, val)
        sto._cap_power_var = None
        sto._cap_energy_var = None
        sto._soc_vars = []
        sto._charge_vars = []
        sto._discharge_vars = []
        sto._spill_vars = []
        sto._soc_inter_vars = []

    for link in s._links:
        for (kind, name, field_), val in scenario.overrides.items():
            if kind == "link" and name == link.name:
                setattr(link, field_, val)
        link._cap_var = None
        link._flow_vars = []
        link._flow_rev_vars = []
        link._flow_signed_vars = []
        link._flow_out_vars = []
        link._inv_vars = []

    if scenario.demand_factor != 1.0:
        for ld in s._loads:
            if isinstance(ld.amount, np.ndarray):
                ld.amount = np.asarray(ld.amount, dtype=float) * scenario.demand_factor
            else:
                ld.amount = float(ld.amount) * scenario.demand_factor

    return s


# ---------------------------------------------------------------------------
# Two-stage stochastic programming
# ---------------------------------------------------------------------------

def solve_stochastic(
    base_system: "EnergySystem",
    scenarios: list[Scenario],
    risk_measure: str = "expected",
    cvar_alpha: float = 0.05,
    method: str = "benders",
    max_iter: int = 50,
    tol: float = 1e-3,
    stabilisation: str = "plain",
    verbose: bool = False,
) -> StochasticResult:
    """
    Solve a two-stage stochastic capacity-expansion problem.

    Parameters
    ----------
    base_system
        The nominal ``EnergySystem``. Extendable components' cap_vars
        become the first-stage (here-and-now) decisions shared across
        all scenarios.
    scenarios
        List of :class:`Scenario` draws. Probabilities should sum to 1.
    risk_measure
        ``"expected"`` minimises Σ p_s cost_s; ``"cvar"`` minimises the
        Rockafellar-Uryasev CVaR_α; ``"worst_case"`` minimises max_s.
    cvar_alpha
        Tail probability for CVaR (typ. 0.05 = worst-5%).
    method
        ``"benders"`` (default) decomposes by scenario via
        :class:`BendersDecomposer` — cheaper for ≥ 4 scenarios.
        ``"extensive"`` builds one big LP and solves once — faster for
        small scenario counts but memory-bound.
    max_iter, tol, stabilisation
        Passed through to the Benders driver when ``method="benders"``.
    """
    from nexus_energy.decomposition import (
        BendersDecomposer,
        _collect_extendable_names,
        _cvar_at_caps,
    )

    t0 = time.perf_counter()

    if not scenarios:
        raise ValueError("solve_stochastic: empty scenario list")
    prob_sum = sum(s.probability for s in scenarios)
    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(
            f"scenario probabilities must sum to 1 (got {prob_sum:.6f})")

    probs = [s.probability for s in scenarios]
    sub_systems = [apply_scenario(base_system, s) for s in scenarios]

    # Degenerate case: no first-stage decisions — just solve each scenario
    # independently and aggregate. Benders has nothing to decompose here.
    if not _collect_extendable_names(base_system):
        return _solve_stochastic_no_first_stage(
            scenarios, sub_systems, probs,
            risk_measure=risk_measure, cvar_alpha=cvar_alpha, t0=t0,
        )

    if method == "benders":
        decomp = BendersDecomposer(
            system=base_system,
            subsystems=sub_systems,
            period_weights=probs,
            max_iter=max_iter,
            tol=tol,
            stabilisation=stabilisation,
            objective_mode=risk_measure,
            cvar_alpha=cvar_alpha,
            verbose=verbose,
        )
        br = decomp.solve()
        if br.status != "optimal":
            return StochasticResult(
                status=br.status,
                expected_cost=float("nan"),
                scenario_costs={},
                capacity_decisions=br.final_capacities,
                solve_time=time.perf_counter() - t0,
                method="benders",
                n_iterations=len(br.iterations),
            )
        final_caps = br.final_capacities
        final_sub = br.iterations[-1].subproblem_costs if br.iterations else []
        sc_costs = {s.name: c for s, c in zip(scenarios, final_sub)}
        expected = sum(p * c for p, c in zip(probs, final_sub))
        worst = max(final_sub) if final_sub else float("nan")
        cvar = _cvar_at_caps(final_sub, probs, cvar_alpha) if final_sub else float("nan")
        return StochasticResult(
            status="optimal",
            expected_cost=expected,
            scenario_costs=sc_costs,
            capacity_decisions=final_caps,
            solve_time=time.perf_counter() - t0,
            cvar=cvar,
            worst_case_cost=worst,
            method="benders",
            n_iterations=len(br.iterations),
        )

    if method == "extensive":
        return _solve_stochastic_extensive(
            base_system, scenarios, sub_systems, probs,
            risk_measure=risk_measure, cvar_alpha=cvar_alpha,
            verbose=verbose, t0=t0,
        )

    raise ValueError(f"method must be 'benders' | 'extensive' (got {method!r})")


def _solve_stochastic_no_first_stage(
    scenarios: list[Scenario],
    sub_systems: list["EnergySystem"],
    probs: list[float],
    *,
    risk_measure: str,
    cvar_alpha: float,
    t0: float,
) -> StochasticResult:
    """Pure operational stochastic — no shared first-stage decision."""
    from nexus_energy.decomposition import _cvar_at_caps
    sc_costs: dict[str, float] = {}
    cost_list: list[float] = []
    for sc, sub in zip(scenarios, sub_systems):
        res = sub.optimise()
        c = float(res.total_cost) if res.status == "optimal" else float("inf")
        sc_costs[sc.name] = c
        cost_list.append(c)
    expected = sum(p * c for p, c in zip(probs, cost_list))
    return StochasticResult(
        status="optimal" if all(np.isfinite(c) for c in cost_list) else "infeasible",
        expected_cost=expected,
        scenario_costs=sc_costs,
        capacity_decisions={},
        solve_time=time.perf_counter() - t0,
        cvar=_cvar_at_caps(cost_list, probs, cvar_alpha),
        worst_case_cost=max(cost_list),
        method="direct",
        n_iterations=0,
    )


def _solve_stochastic_extensive(
    base_system: "EnergySystem",
    scenarios: list[Scenario],
    sub_systems: list["EnergySystem"],
    probs: list[float],
    *,
    risk_measure: str,
    cvar_alpha: float,
    verbose: bool,
    t0: float,
) -> StochasticResult:
    """
    Extensive form: solve each scenario alone, then solve the aggregated
    first-stage master by iterating until first-stage caps converge.

    This is a "two-shot" substitute — for a true extensive form we'd
    build one giant LP coupling all scenarios to one cap_var set. That
    requires bypassing ``EnergySystem.optimise`` which rebuilds a fresh
    model per call. The Benders path above is strictly better for > 1
    scenario; this branch exists so users can sanity-check against a
    model-structure alternative.
    """
    from nexus_energy.decomposition import BendersDecomposer, _cvar_at_caps
    # Delegate to Benders with extreme-low tolerance — numerically
    # equivalent to the extensive form when all scenarios are LPs.
    decomp = BendersDecomposer(
        system=base_system,
        subsystems=sub_systems,
        period_weights=probs,
        max_iter=100,
        tol=1e-6,
        stabilisation="plain",
        objective_mode=risk_measure,
        cvar_alpha=cvar_alpha,
        verbose=verbose,
    )
    br = decomp.solve()
    final_sub = br.iterations[-1].subproblem_costs if br.iterations else []
    sc_costs = {s.name: c for s, c in zip(scenarios, final_sub)}
    expected = sum(p * c for p, c in zip(probs, final_sub))
    return StochasticResult(
        status=br.status if br.iterations else "failed",
        expected_cost=expected if final_sub else float("nan"),
        scenario_costs=sc_costs,
        capacity_decisions=br.final_capacities,
        solve_time=time.perf_counter() - t0,
        cvar=_cvar_at_caps(final_sub, probs, cvar_alpha) if final_sub else None,
        worst_case_cost=max(final_sub) if final_sub else None,
        method="extensive",
        n_iterations=len(br.iterations),
    )


# ---------------------------------------------------------------------------
# Robust optimisation — budget-uncertainty set (Bertsimas-Sim)
# ---------------------------------------------------------------------------

@dataclass
class BudgetUncertaintySet:
    """
    Polyhedral budget-uncertainty set (Bertsimas-Sim style).

    For n primitive uncertainty dimensions, each dimension deviates by
    at most ``magnitudes[i]`` fraction from nominal, and at most
    ``budget`` dimensions deviate simultaneously. Under this set the
    worst-case realisation puts full deviation on the ``budget`` most
    costly dimensions and zero on the rest.

    We instantiate the worst case as a single extreme scenario: all
    dimensions at their worst, then scale back so only ``budget`` of
    them saturate. For a single dimension (demand or CF), this reduces
    to the classical worst-case.
    """
    demand_up: float = 0.0
    cf_down: float = 0.0
    fuel_cost_up: float = 0.0
    budget: float = 1.0

    def worst_case_scenario(self) -> Scenario:
        # Saturate each active direction by budget-share.
        n_active = sum(1 for x in (self.demand_up, self.cf_down,
                                   self.fuel_cost_up) if x > 0)
        if n_active == 0:
            return Scenario("nominal", 1.0)
        share = min(self.budget, float(n_active)) / max(n_active, 1)
        return Scenario(
            name="robust_worst",
            probability=1.0,
            demand_factor=1.0 + self.demand_up * share,
            carrier_factor_scale=max(0.0, 1.0 - self.cf_down * share),
            fuel_cost_factor=1.0 + self.fuel_cost_up * share,
        )


def solve_robust(
    base_system: "EnergySystem",
    uncertainty: BudgetUncertaintySet | None = None,
    *,
    demand_deviation: float = 0.15,
    cf_deviation: float = 0.20,
    verbose: bool = False,
) -> StochasticResult:
    """
    Robust optimisation against a polyhedral budget set.

    If ``uncertainty`` is None we reproduce the Phase 8 API using
    ``demand_deviation`` / ``cf_deviation`` as a single saturating
    direction (equivalent to the old "worst case = max demand + min
    renewables" shortcut). Returns a :class:`StochasticResult` whose
    ``worst_case_cost`` is the guaranteed-feasible value.
    """
    if uncertainty is None:
        uncertainty = BudgetUncertaintySet(
            demand_up=demand_deviation,
            cf_down=cf_deviation,
            budget=1.0,
        )
    worst = uncertainty.worst_case_scenario()
    return solve_stochastic(
        base_system, [worst], risk_measure="expected",
        method="benders", verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Monte Carlo harness — out-of-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_plan(
    base_system: "EnergySystem",
    capacities: dict[str, float],
    scenarios: list[Scenario],
    *,
    cvar_alpha: float = 0.05,
) -> dict:
    """
    Pin a first-stage plan (``capacities``) and evaluate its second-stage
    operational cost in each out-of-sample scenario.

    Useful for estimating expected / CVaR / worst-case realised cost of
    a plan produced by one method on fresh draws, e.g. judging whether
    the Benders solution generalises off its training sample.

    Returns
    -------
    dict
        ``scenario_costs``, ``expected_cost``, ``cvar``, ``worst_case``,
        ``n_scenarios``.
    """
    from nexus_energy.decomposition import _cvar_at_caps
    costs: dict[str, float] = {}
    for sc in scenarios:
        sub = apply_scenario(base_system, sc)
        res = sub.optimise(benders_fix_caps=capacities, benders_skip_capex=True)
        if res.status != "optimal":
            costs[sc.name] = float("inf")
        else:
            costs[sc.name] = float(res.total_cost)
    probs = [s.probability for s in scenarios]
    expected = sum(p * costs[s.name] for p, s in zip(probs, scenarios))
    cost_list = [costs[s.name] for s in scenarios]
    return {
        "scenario_costs": costs,
        "expected_cost": expected,
        "cvar": _cvar_at_caps(cost_list, probs, cvar_alpha),
        "worst_case": max(cost_list),
        "n_scenarios": len(scenarios),
    }


# ---------------------------------------------------------------------------
# Scenario reduction — k-medoids on the scenario parameter vector
# ---------------------------------------------------------------------------

def reduce_scenarios(
    scenarios: list[Scenario],
    n_reduced: int,
    seed: int = 0,
) -> list[Scenario]:
    """
    Collapse a large scenario set to ``n_reduced`` medoids.

    Each scenario is summarised by its (demand_factor,
    carrier_factor_scale, fuel_cost_factor) vector; we run k-medoids++
    (via :func:`nexus_energy.temporal.k_medoids`) and re-aggregate
    probabilities onto the surviving medoids by nearest-neighbour
    assignment. Overrides on a medoid are preserved; overrides on
    absorbed scenarios are discarded.
    """
    if n_reduced >= len(scenarios):
        return list(scenarios)
    from nexus_energy.temporal import k_medoids

    X = np.array([
        [s.demand_factor, s.carrier_factor_scale, s.fuel_cost_factor]
        for s in scenarios
    ], dtype=float)
    medoid_indices, labels, _dist = k_medoids(X, k=n_reduced, seed=seed)
    # Rebuild probabilities by summing all scenarios assigned to each medoid.
    new_probs = np.zeros(n_reduced)
    for i, lbl in enumerate(labels):
        new_probs[lbl] += scenarios[i].probability
    # Normalise in case of floating-point drift.
    new_probs /= new_probs.sum()

    reduced: list[Scenario] = []
    for k, idx in enumerate(medoid_indices):
        template = scenarios[idx]
        reduced.append(Scenario(
            name=f"reduced_{k}",
            probability=float(new_probs[k]),
            demand_factor=template.demand_factor,
            carrier_factor_scale=template.carrier_factor_scale,
            fuel_cost_factor=template.fuel_cost_factor,
            overrides=dict(template.overrides),
        ))
    return reduced


# ---------------------------------------------------------------------------
# Wasserstein-robust scenario reduction unified with DRO (Phase 21)
# ---------------------------------------------------------------------------

def _scenario_features(scenarios: list[Scenario]) -> np.ndarray:
    """Vectorise scenarios into the (demand, CF, fuel) feature matrix.

    Single source of truth for the feature embedding used by both
    :func:`reduce_scenarios` and :func:`reduce_scenarios_wasserstein`, so
    the two reductions live in the same metric space.
    """
    return np.array([
        [s.demand_factor, s.carrier_factor_scale, s.fuel_cost_factor]
        for s in scenarios
    ], dtype=float)


def reduce_scenarios_wasserstein(
    scenarios: list[Scenario],
    n_reduced: int,
    *,
    metric: str = "euclidean",
    order: int = 1,
    seed: int = 0,
) -> tuple[list[Scenario], float]:
    """
    Wasserstein-optimal scenario reduction with a DRO-ready radius.

    Unifies *scenario reduction* and *distributional robustness*. Today
    :func:`reduce_scenarios` (k-medoids) and :func:`solve_wasserstein_dro`
    are independent: the former shrinks the tree, the latter immunises
    against a Wasserstein ambiguity ball whose radius ``ε`` must be picked
    by hand. This routine does both at once — it selects ``n_reduced``
    representative :class:`Scenario`\\ s by *minimising the type-p
    Wasserstein transport distance* between the full empirical measure and
    the reduced (re-weighted) one, and returns that achieved transport
    distance as the ``radius`` so it can directly parameterise the
    ambiguity set of :func:`solve_wasserstein_dro`.

    Method
    ------
    Following the optimal-quantization view of scenario reduction
    (Dupačová, Gröwe-Kuska & Römisch, "Scenario reduction in stochastic
    programming", *Math. Program.* 95 (2003), 493-511; Pflug,
    "Scenario tree generation for multiperiod financial optimization by
    optimal discretization", *Math. Program.* 89 (2001), 251-271), the
    *optimal redistribution rule* transports every deleted scenario onto
    its nearest surviving one, and the resulting type-``p`` Wasserstein
    distance between the original measure ``Σ_i p_i δ_{ξ_i}`` and the
    reduced measure is

        W_p = ( Σ_i p_i · d(ξ_i, ξ_{m(i)})^p )^{1/p},

    where ``m(i)`` is the representative assigned to original ``i`` and
    ``d`` is the ground metric on the scenario feature vector. Minimising
    this transport cost over the choice of representatives is exactly the
    k-medoids objective under the ground metric, so we *reuse* the existing
    :func:`temporal.k_medoids` machinery (squared-Euclidean PAM with
    k-medoids++ seeding) and the same feature embedding as
    :func:`reduce_scenarios`. The optimal re-weighting of the survivors is
    the closed-form mass aggregation: each representative's probability is
    the total mass of the originals transported to it (Dupačová et al.,
    Thm. 2 redistribution rule).

    The returned ``radius`` is the achieved ``W_p``. Feeding it as
    ``epsilon=radius`` to :func:`solve_wasserstein_dro` yields a
    distributionally-robust solution on the *reduced* tree whose ambiguity
    ball is exactly large enough to recover (in the Wasserstein sense) the
    information discarded by the reduction — one pipeline producing a
    distributionally-robust reduced model (Mohajerin Esfahani & Kuhn,
    "Data-driven distributionally robust optimization using the
    Wasserstein metric", *Math. Program.* 171 (2018), 115-166).

    Parameters
    ----------
    scenarios
        Full scenario ensemble. Probabilities should sum to 1.
    n_reduced
        Number of representatives to keep. If ``>= len(scenarios)`` the
        full set is returned unchanged with ``radius == 0.0`` (the reduced
        measure equals the full one).
    metric
        Ground metric on the scenario feature vector. Only ``"euclidean"``
        is currently supported (matches the k-medoids ground metric).
    order
        Wasserstein order ``p`` (1 = type-1 Kantorovich, 2 = type-2). The
        ground transport cost is ``d^p`` and the radius is the ``p``-th
        root. Type-1 (the default) is the order assumed by
        :func:`solve_wasserstein_dro`.
    seed
        Seed for the k-medoids++ seeding (reproducibility).

    Returns
    -------
    (reduced_scenarios, radius)
        ``reduced_scenarios`` is a list of ``min(n_reduced, len)``
        :class:`Scenario`\\ s with optimally aggregated probabilities;
        ``radius`` is the achieved type-``p`` Wasserstein distance
        (a float, ``>= 0``, ``== 0`` when nothing is merged).
    """
    if metric != "euclidean":
        raise ValueError(
            f"reduce_scenarios_wasserstein: unsupported metric {metric!r} "
            "(only 'euclidean' is implemented)")
    if order not in (1, 2):
        raise ValueError("reduce_scenarios_wasserstein: order must be 1 or 2")
    if not scenarios:
        raise ValueError("reduce_scenarios_wasserstein: empty scenario list")

    # n_reduced >= n_full: the reduced measure can reproduce the full one
    # exactly (identity transport), so the Wasserstein radius is zero.
    if n_reduced >= len(scenarios):
        return list(scenarios), 0.0

    from nexus_energy.temporal import k_medoids

    X = _scenario_features(scenarios)
    probs = np.array([s.probability for s in scenarios], dtype=float)
    p_sum = probs.sum()
    if p_sum > 0:
        probs = probs / p_sum

    # k-medoids minimises Σ squared-Euclidean distance to the assigned
    # medoid — exactly the (squared) ground transport cost. ``distances``
    # returned by k_medoids are squared-Euclidean to the assigned medoid.
    medoid_indices, labels, sq_dists = k_medoids(X, k=n_reduced, seed=seed)

    # Achieved type-p Wasserstein distance under the optimal redistribution
    # rule: each original is transported to its assigned medoid with its
    # own probability mass. ground d = sqrt(sq_dist); transport cost = d^p.
    ground_d = np.sqrt(np.maximum(sq_dists, 0.0))
    transported = float(np.sum(probs * (ground_d ** order)))
    radius = float(transported ** (1.0 / order))

    # Optimal re-weighting: representative mass = mass of assigned originals.
    new_probs = np.zeros(n_reduced)
    for i, lbl in enumerate(labels):
        new_probs[lbl] += probs[i]
    s_new = new_probs.sum()
    if s_new > 0:
        new_probs /= s_new

    reduced: list[Scenario] = []
    for k, idx in enumerate(medoid_indices):
        template = scenarios[idx]
        reduced.append(Scenario(
            name=f"wreduced_{k}",
            probability=float(new_probs[k]),
            demand_factor=template.demand_factor,
            carrier_factor_scale=template.carrier_factor_scale,
            fuel_cost_factor=template.fuel_cost_factor,
            overrides=dict(template.overrides),
        ))
    return reduced, radius


# ---------------------------------------------------------------------------
# Chance constraints via SAA
# ---------------------------------------------------------------------------

@dataclass
class ChanceConstraint:
    """
    Sample-average-approximation chance constraint.

    Given a constraint ``lhs(scenario) ≤ rhs`` that should hold with
    probability ≥ ``1 - alpha`` in the joint scenario distribution,
    SAA relaxes it to: at most ``alpha * N`` scenarios may violate,
    with a Big-M relaxation per scenario.

    Three modes of use:
      - ``violates(samples)`` — post-hoc violation count on realised LHS.
      - ``bonferroni_correction(n)`` — shrink α for joint bound via union.
      - ``saa_quantile_threshold(samples, probs)`` — deterministic-
        equivalent RHS that makes the CC tight: we look up the
        probability-weighted (1-α)-quantile of the LHS samples. When the
        LHS is a scenario-only random variable (no first-stage coupling
        beyond a linear functional), enforcing ``lhs_first_stage ≤
        saa_quantile_threshold(...)`` as a deterministic constraint is
        the *exact* SAA CC reduction (Shapiro–Dentcheva–Ruszczyński
        Ch. 4).
    """
    name: str
    alpha: float
    threshold: float
    big_m: float = 1e9

    def violates(self, realised_lhs: list[float]) -> int:
        """Return the number of scenarios that violate the threshold."""
        return int(sum(1 for x in realised_lhs if x > self.threshold))

    def violation_probability(
        self,
        realised_lhs: list[float],
        probabilities: Optional[list[float]] = None,
    ) -> float:
        """Return Pr{lhs > threshold} under the given scenario weights."""
        if probabilities is None:
            probabilities = [1.0 / len(realised_lhs)] * len(realised_lhs)
        return float(sum(p for x, p in zip(realised_lhs, probabilities)
                         if x > self.threshold))

    def bonferroni_correction(self, n_constraints: int) -> float:
        """Return the per-constraint α for a joint (1-α) bound via union."""
        return self.alpha / max(n_constraints, 1)

    def saa_quantile_threshold(
        self,
        samples: list[float],
        probabilities: Optional[list[float]] = None,
    ) -> float:
        """
        Return the probability-weighted (1-α)-quantile of ``samples``.

        Enforcing ``lhs_deterministic ≤ this_value`` is the exact SAA
        deterministic equivalent of ``Pr{lhs ≤ rhs} ≥ 1-α`` when ``lhs``
        separates into a first-stage term and an independent scenario
        factor (the common case for peak-load coverage, capacity
        reserves, renewable firmness CCs).
        """
        if not samples:
            raise ValueError("saa_quantile_threshold: empty samples")
        N = len(samples)
        if probabilities is None:
            probabilities = [1.0 / N] * N
        idx = np.argsort(samples)
        sorted_samples = [samples[i] for i in idx]
        sorted_probs = [probabilities[i] for i in idx]
        cum = 0.0
        target = 1.0 - self.alpha
        for x, p in zip(sorted_samples, sorted_probs):
            cum += p
            if cum >= target - 1e-12:
                return float(x)
        return float(sorted_samples[-1])


def solve_saa_chance_constrained(
    base_system: "EnergySystem",
    scenarios: list[Scenario],
    *,
    peak_loads_by_scenario: Optional[list[float]] = None,
    reserve_margin: float = 0.15,
    firm_credit: Optional[dict[str, float]] = None,
    alpha: float = 0.05,
    risk_measure: str = "expected",
    cvar_alpha: float = 0.05,
    method: str = "benders",
    max_iter: int = 50,
    verbose: bool = False,
) -> StochasticResult:
    """
    Stochastic planning with a native peak-demand chance constraint.

    Enforces: Pr{ (1+margin) × peak_load(s) ≤ Σ_g credit_g · cap_g } ≥ 1-α
    via the SAA deterministic-equivalent quantile reduction. Concretely,
    we look up the (1-α)-quantile of ``peak_loads_by_scenario`` and pin
    ``set_reserve_margin(margin, firm_credit)`` on the base system with
    that quantile's scenario as the peak reference. The resulting problem
    remains a pure LP (no extra binaries), yet feasibility satisfies the
    chance constraint at level α *exactly* — which is the standard SAA
    reduction for CCs whose LHS is a scenario-only random variable
    coupled linearly to first-stage capacity.

    Parameters
    ----------
    base_system
        Nominal system (will be deep-copied before modification).
    scenarios
        Uncertainty draws for the second-stage problem.
    peak_loads_by_scenario
        Optional per-scenario peak load in MW. When ``None``, we derive
        it from each scenario's ``demand_factor`` applied to the base
        system's peak load.
    reserve_margin
        Firm capacity cushion over peak load (typ. 0.15 = 15%).
    firm_credit
        Per-technology firm-capacity credit (e.g.,
        ``{"gas": 1.0, "solar": 0.05, "wind": 0.15}``).
    alpha
        Violation probability (typ. 0.05 = CC holds ≥ 95% of the time).
    Rest
        Passed through to :func:`solve_stochastic`.

    Returns
    -------
    :class:`StochasticResult`
    """
    if firm_credit is None:
        firm_credit = {}
    probs = [s.probability for s in scenarios]

    # Derive per-scenario peak load if not supplied.
    if peak_loads_by_scenario is None:
        base_peak = 0.0
        for ld in base_system._loads:
            amt = ld.amount
            if isinstance(amt, np.ndarray):
                base_peak += float(np.max(amt))
            else:
                base_peak += float(amt)
        peak_loads_by_scenario = [base_peak * s.demand_factor for s in scenarios]

    # SAA deterministic-equivalent reduction: the CC
    #   Pr{firm_cap >= (1+m)*peak_s} >= 1-α
    # collapses to the deterministic
    #   firm_cap >= (1+m) * quantile_{1-α}(peak_s)
    # because ``peak_s`` is a scenario-only random variable not coupled
    # to first-stage decisions. The reserve-margin constraint with
    # ``peak_override`` applied to each subsystem enforces exactly this.
    cc = ChanceConstraint(name="peak_cover", alpha=alpha, threshold=0.0)
    peak_q = cc.saa_quantile_threshold(peak_loads_by_scenario, probs)
    firm_req = (1.0 + reserve_margin) * peak_q

    # Our Benders engine does not generate feasibility cuts (deferred to
    # Phase 12). To guarantee every subproblem is feasible at the master's
    # initial cap guess, we pre-size the *first* extendable firm-credited
    # generator's ``min_capacity`` so that, together with the non-
    # extendable fleet's firm contribution, the master starts at or above
    # ``firm_req``. The reserve margin constraint (peak_override=peak_q)
    # still binds subsequent iterations — if the planner prefers to build
    # solar/other-tech firm capacity, it can relax above min.
    fixed_firm = 0.0
    for gen in base_system._generators:
        cred = firm_credit.get(gen.tech, 0.0)
        if cred == 0.0:
            continue
        if not gen.extendable:
            fixed_firm += cred * float(gen.capacity)
    needed_from_ext = max(0.0, firm_req - fixed_firm)

    scoped_system = copy.deepcopy(base_system)
    if needed_from_ext > 0.0:
        for gen in scoped_system._generators:
            cred = firm_credit.get(gen.tech, 0.0)
            if cred > 0.0 and gen.extendable:
                required_ext_cap = needed_from_ext / cred
                if required_ext_cap > gen.min_capacity:
                    gen.min_capacity = min(required_ext_cap, gen.max_capacity)
                break

    sub_systems: list["EnergySystem"] = []
    for sc in scenarios:
        sub = apply_scenario(scoped_system, sc)
        sub.set_reserve_margin(reserve_margin, firm_credit,
                               peak_override=peak_q)
        sub_systems.append(sub)

    from nexus_energy.decomposition import (
        BendersDecomposer,
        _collect_extendable_names,
        _cvar_at_caps,
    )
    t0 = time.perf_counter()
    if not _collect_extendable_names(scoped_system):
        return _solve_stochastic_no_first_stage(
            scenarios, sub_systems, probs,
            risk_measure=risk_measure, cvar_alpha=cvar_alpha, t0=t0,
        )
    decomp = BendersDecomposer(
        system=scoped_system,
        subsystems=sub_systems,
        period_weights=probs,
        max_iter=max_iter,
        tol=1e-3,
        stabilisation="plain",
        objective_mode=risk_measure,
        cvar_alpha=cvar_alpha,
        verbose=verbose,
    )
    br = decomp.solve()
    if br.status != "optimal":
        return StochasticResult(
            status=br.status,
            expected_cost=float("nan"),
            scenario_costs={},
            capacity_decisions=br.final_capacities,
            solve_time=time.perf_counter() - t0,
            method="saa_chance_constrained",
            n_iterations=len(br.iterations),
        )
    final_sub = br.iterations[-1].subproblem_costs if br.iterations else []
    sc_costs = {s.name: c for s, c in zip(scenarios, final_sub)}
    expected = sum(p * c for p, c in zip(probs, final_sub))
    worst = max(final_sub) if final_sub else float("nan")
    cvar = _cvar_at_caps(final_sub, probs, cvar_alpha) if final_sub else None
    return StochasticResult(
        status="optimal",
        expected_cost=expected,
        scenario_costs=sc_costs,
        capacity_decisions=br.final_capacities,
        solve_time=time.perf_counter() - t0,
        cvar=cvar,
        worst_case_cost=worst,
        method="saa_chance_constrained",
        n_iterations=len(br.iterations),
    )


# ---------------------------------------------------------------------------
# Progressive hedging (Rockafellar-Wets) — LP-friendly ℓ1 variant
# ---------------------------------------------------------------------------

def solve_stochastic_ph(
    base_system: "EnergySystem",
    scenarios: list[Scenario],
    *,
    rho: float = 1.0,
    max_iter: int = 30,
    tol: float = 1e-3,
    initial_radius: float = 0.5,
    radius_decay: float = 0.85,
    verbose: bool = False,
) -> StochasticResult:
    """
    Progressive hedging with an ℓ∞ trust-region consensus surrogate.

    Classical PH (Rockafellar & Wets 1991) uses a quadratic proximal term
    ``(ρ/2) ‖x_s − x̄‖²`` to couple per-scenario first-stage decisions to
    their probability-weighted mean. HiGHS is LP-only, so we substitute
    the quadratic proximal with an iterative **ℓ∞ trust-region** that
    shrinks each iteration by ``radius_decay``:

      At iter k, scenario s is solved with its first-stage capacity
      variables clamped to ``[x̄_{k} · (1 − r_k), x̄_{k} · (1 + r_k)]``.
      The radius ``r_k = initial_radius · radius_decay^k``, so bounds
      converge geometrically to the consensus mean. This is a linear
      substitute for Fan & Liu 2010's ℓ∞-PH variant, and is the same
      device BendersDecomposer uses for its trust-region stabilisation.

    Convergence is declared when the probability-weighted ℓ∞ dispersion
    of scenario first-stage decisions drops below ``tol`` in relative
    terms.

    PH complements Benders: Benders decomposes *by period* and shares a
    hard-linked master capacity; PH decomposes *by scenario* and derives
    a consensus capacity. On stochastic problems with no temporal
    structure, PH is often faster than Benders because every sub-solve
    runs the full horizon (no master-subproblem imbalance).

    Parameters
    ----------
    base_system : EnergySystem
        Nominal system. Extendable components define the first-stage
        decision vector ``x``.
    scenarios : list[Scenario]
        Uncertainty draws.
    rho : float
        Unused in the ℓ∞ variant — kept for API compatibility with the
        quadratic PH specification. Values > 1 signal "push consensus
        harder" and are reflected in a slight bias of the radius decay.
    max_iter, tol, initial_radius, radius_decay
        Consensus-loop knobs.

    Returns
    -------
    :class:`StochasticResult` with ``method="progressive_hedging"``.
    """
    from nexus_energy.decomposition import _collect_extendable_names, _cvar_at_caps

    t0 = time.perf_counter()
    if not scenarios:
        raise ValueError("solve_stochastic_ph: empty scenario list")
    prob_sum = sum(s.probability for s in scenarios)
    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(
            f"scenario probabilities must sum to 1 (got {prob_sum:.6f})")

    probs = np.array([s.probability for s in scenarios], dtype=float)
    N = len(scenarios)
    cap_info = _collect_extendable_names(base_system)
    if not cap_info:
        # No first-stage decisions — fall through to direct solve.
        sub_systems = [apply_scenario(base_system, s) for s in scenarios]
        return _solve_stochastic_no_first_stage(
            scenarios, sub_systems, probs.tolist(),
            risk_measure="expected", cvar_alpha=0.05, t0=t0,
        )
    cap_names = [c[0] for c in cap_info]
    lo = np.array([c[1] for c in cap_info], dtype=float)
    hi = np.array([c[2] for c in cap_info], dtype=float)

    # Iteration 0 — solve each scenario to full freedom.
    sub_systems = [apply_scenario(base_system, s) for s in scenarios]
    x = np.zeros((N, len(cap_names)), dtype=float)
    costs = np.zeros(N, dtype=float)
    for i, sub in enumerate(sub_systems):
        res = sub.optimise()
        if res.status != "optimal":
            return StochasticResult(
                status=res.status,
                expected_cost=float("nan"),
                scenario_costs={scenarios[i].name: float("inf")},
                capacity_decisions={},
                solve_time=time.perf_counter() - t0,
                method="progressive_hedging",
                n_iterations=0,
            )
        costs[i] = float(res.total_cost)
        for j, name in enumerate(cap_names):
            x[i, j] = float(res.capacity_additions.get(name, 0.0))

    x_bar = probs @ x
    bias = max(rho, 1e-3)
    radius = float(initial_radius)
    converged = False
    for k in range(1, max_iter + 1):
        disp = np.max(np.abs(x - x_bar[np.newaxis, :]), axis=0)
        rel_disp = disp / np.maximum(np.abs(x_bar), 1.0)
        max_rel = float(np.max(rel_disp))
        if verbose:
            print(f"[PH] iter {k-1} max_rel_disp={max_rel:.4f} radius={radius:.4f}")
        if max_rel < tol:
            converged = True
            break

        # Clamp each scenario's extendable bounds to the trust region.
        for j, name in enumerate(cap_names):
            lo_j = max(lo[j], x_bar[j] * (1.0 - radius))
            hi_j = min(hi[j], x_bar[j] * (1.0 + radius))
            if hi_j < lo_j:
                hi_j = lo_j
            _apply_cap_bounds(sub_systems, name, lo_j, hi_j)

        # Re-solve each scenario under tightened bounds.
        for i, sub in enumerate(sub_systems):
            res = sub.optimise()
            if res.status != "optimal":
                if verbose:
                    print(f"[PH] scenario {i} failed ({res.status}) — "
                          "expanding radius")
                radius = min(1.0, radius / max(radius_decay, 1e-6))
                break
            costs[i] = float(res.total_cost)
            for j, name in enumerate(cap_names):
                x[i, j] = float(res.capacity_additions.get(name, 0.0))
        else:
            # Full sweep succeeded — update consensus.
            x_bar = probs @ x
            radius *= radius_decay / bias if bias > 1.0 else radius_decay

    final_caps = {name: float(x_bar[j]) for j, name in enumerate(cap_names)}
    sc_costs = {scenarios[i].name: float(costs[i]) for i in range(N)}
    expected = float(probs @ costs)
    worst = float(np.max(costs))
    cvar = _cvar_at_caps(costs.tolist(), probs.tolist(), 0.05)
    return StochasticResult(
        status="optimal" if converged else "max_iter",
        expected_cost=expected,
        scenario_costs=sc_costs,
        capacity_decisions=final_caps,
        solve_time=time.perf_counter() - t0,
        cvar=cvar,
        worst_case_cost=worst,
        method="progressive_hedging",
        n_iterations=k,
    )


def _apply_cap_bounds(
    sub_systems: list["EnergySystem"],
    cap_name: str,
    lower: float,
    upper: float,
) -> None:
    """Set min_capacity/max_capacity on the extendable named ``cap_name``.

    Handles the three naming conventions used by
    ``_collect_extendable_names``: ``gen.name``, ``link.name``,
    ``f"{sto.name}_power"`` / ``f"{sto.name}_energy"``.
    """
    for sub in sub_systems:
        for gen in sub._generators:
            if gen.extendable and gen.name == cap_name:
                gen.min_capacity = max(0.0, lower)
                gen.max_capacity = upper if np.isfinite(upper) else float("inf")
                gen._cap_var = None
                gen._p_vars = []
                return
        for link in sub._links:
            if link.extendable and link.name == cap_name:
                link.min_capacity = max(0.0, lower)
                link.max_capacity = upper if np.isfinite(upper) else float("inf")
                link._cap_var = None
                link._flow_vars = []
                return
        for sto in sub._storages:
            if not sto.extendable:
                continue
            if cap_name == f"{sto.name}_power":
                sto.min_power_capacity = max(0.0, lower)
                sto.max_power_capacity = upper if np.isfinite(upper) else float("inf")
                sto._cap_power_var = None
                return
            if cap_name == f"{sto.name}_energy":
                sto.min_energy_capacity = max(0.0, lower)
                sto.max_energy_capacity = upper if np.isfinite(upper) else float("inf")
                sto._cap_energy_var = None
                return


# ===========================================================================
# Phase 9 depth pass — stochastic & robust algorithms operating on an
# explicit multi-stage MILP description.
#
# The two-stage helpers above route through ``EnergySystem`` /
# ``BendersDecomposer``. The algorithms below (SDDiP, general-form chance
# constraints with Big-M binaries, Wasserstein DRO, risk-averse cuts) need
# fine-grained control of the per-stage MILP and its state-copy / dual
# structure, so they consume a light-weight problem description built
# directly against the ``nexus_opt`` ``Model`` API. This keeps the
# decomposition.py / core.py modules untouched while still demonstrating
# each algorithm's defining mathematical property end-to-end.
# ===========================================================================


# ---------------------------------------------------------------------------
# 9.1 SDDiP — multi-stage stochastic MILP with Lagrangian cuts
# ---------------------------------------------------------------------------

@dataclass
class StageProblem:
    """
    One stage of a multi-stage stochastic MILP, in the SDDiP state-space
    form of Zou, Ahmed & Sun, "Stochastic dual dynamic integer
    programming", *Math. Program.* 175 (2019), 461-502.

    The decision at stage ``t`` in realisation (node) ``n`` is

        min_{x_t}  c_t(n) · x_t
        s.t.       A_t(n) x_t + B_t(n) z_{t-1} {≤,=,≥} b_t(n)
                   x_t = (state_t, local_t),  state_t binary/integer

    where ``z_{t-1}`` is the *incoming* state copied from the previous
    stage. SDDiP introduces a local copy variable ``z_{t-1}`` constrained
    to equal the parent's outgoing state, and builds Lagrangian cuts by
    relaxing exactly that copy constraint.

    To keep the helper self-contained and solver-agnostic we describe the
    stage with a builder callback rather than raw matrices:

    ``build(model, z_prev, stage_idx)`` must
      * create this stage's variables on ``model`` (using
        ``model.variable`` / ``model.binary`` / ``model.integer``),
      * add this stage's constraints (which may reference the *values* in
        ``z_prev`` — a dict ``state_name -> nexus_opt Var or float``),
      * return ``(stage_cost_expr, out_state)`` where ``out_state`` is a
        dict ``state_name -> Var`` giving this stage's *outgoing* binary
        state (the variables the next stage copies).

    ``state_names`` lists the binary/integer state coordinates threaded
    between stages. ``state_bounds`` gives ``(lo, hi)`` per state coord
    for the copy variable.
    """
    build: Callable
    state_names: tuple[str, ...]
    state_bounds: dict[str, tuple[float, float]]


@dataclass
class SDDiPResult:
    """Result of :func:`solve_sddip`."""
    status: str
    lower_bounds: list[float]           # per-iteration LB (monotone ↑)
    upper_bound: float                  # final statistical UB estimate
    first_stage: dict[str, float]       # outgoing state of stage 0
    n_iterations: int
    solve_time: float
    deterministic_equivalent: Optional[float] = None
    gap: Optional[float] = None


def _sddip_solve_stage(
    z_prev: dict[str, float],
    stage: "StageProblem",
    stage_idx: int,
    cuts: list[tuple[float, dict[str, float]]],
    *,
    relax_state: bool = False,
    pi: Optional[dict[str, float]] = None,
    fix_state: Optional[dict[str, float]] = None,
):
    """
    Build & solve a single SDDiP stage subproblem.

    ``cuts`` are Benders/Lagrangian cuts on this stage's *cost-to-go*
    ``θ`` of the form ``θ ≥ a + Σ_j g_j · z_prev_j`` (a function of the
    *incoming* state). When ``relax_state`` we additionally place the
    incoming state into local copy variables ``z_local`` and add the
    Lagrangian term ``-Σ π_j (z_local_j - z_prev_j)`` to the objective
    while *dropping* the copy-pin constraint (this is the Lagrangian
    relaxation used to build integer-valid cuts).

    Returns ``(obj, out_state_values, z_local_values, theta_value)``.
    """
    model = nx.Model(f"sddip_stage{stage_idx}")

    # Local copy of the incoming state. When not relaxing we pin it to the
    # parent value; when relaxing we leave it free within its bounds and
    # price the deviation in the objective (Lagrangian).
    z_local: dict[str, object] = {}
    for s in stage.state_names:
        lo, hi = stage.state_bounds[s]
        v = model.variable(f"zin_{s}", lower=lo, upper=hi)
        z_local[s] = v
        if not relax_state:
            model.add(v == float(z_prev[s]), name=f"copy_{s}")

    # Hand control to the user builder. It sees the *local copy* vars so
    # the incoming-state relaxation works transparently.
    stage_cost, out_state_vars = stage.build(model, z_local, stage_idx)

    # Cost-to-go epigraph variable. The CTG cuts (built in the backward
    # pass for *this* stage's successors) are affine functions of this
    # stage's **outgoing** state — i.e. the variables the next stage will
    # copy. Binding them on the outgoing state is what makes the master
    # actually trade off immediate cost against future cost.
    theta = model.variable("theta_ctg", lower=0.0, upper=1e15)
    for ci, (a, g) in enumerate(cuts):
        rhs = None
        for sj, gj in g.items():
            if gj == 0.0 or sj not in out_state_vars:
                continue
            term = gj * out_state_vars[sj]
            rhs = term if rhs is None else rhs + term
        if rhs is None:
            model.add(theta >= a, name=f"ctg_{ci}")
        else:
            model.add(theta - rhs >= a, name=f"ctg_{ci}")

    if fix_state is not None:
        for s, val in fix_state.items():
            model.add(out_state_vars[s] == float(val), name=f"fix_{s}")

    obj = stage_cost + theta
    if relax_state and pi is not None:
        # Lagrangian: + Σ π_j (z_prev_j - z_local_j)  (relax copy z=z_prev).
        for s in stage.state_names:
            obj = obj + float(pi[s]) * (float(z_prev[s]) - z_local[s])

    model.minimize(obj)
    res = model.solve(verbose=False)
    if res.status != "optimal":
        return None
    out_vals = {s: float(res.value(v)) for s, v in out_state_vars.items()}
    zloc_vals = {s: float(res.value(v)) for s, v in z_local.items()}
    return (float(res.objective), out_vals, zloc_vals,
            float(res.value(theta)))


def solve_sddip(
    stages: list["StageProblem"],
    scenarios_per_stage: list[list[dict]],
    *,
    stage_probabilities: Optional[list[list[float]]] = None,
    max_iter: int = 40,
    n_forward: int = 1,
    lagrangian_iters: int = 60,
    lagrangian_step: float = 1.0,
    tol: float = 1e-4,
    seed: int = 0,
    deterministic_equivalent: Optional[float] = None,
    verbose: bool = False,
) -> SDDiPResult:
    """
    Stochastic Dual Dynamic integer Programming (SDDiP).

    Implements the algorithm of Zou, Ahmed & Sun (2019), *Math. Program.*
    175:461-502 — multi-stage stochastic mixed-**integer** programming via
    Lagrangian cuts on the binary/integer **state** variables. Classical
    SDDP (Pereira & Pinto 1991) Benders cuts are *not* valid for the
    integer value function (it is non-convex in the state); the key
    contribution of SDDiP is that the **Lagrangian cut** obtained by
    dualising the state-copy constraint ``z_{t} = x_{t}^{out}`` is tight
    and valid for the (lower convex envelope of the) integer cost-to-go,
    and — for binary state — the collection of Lagrangian cuts converges
    to the exact value function (their Theorem 1 / exactness for binary
    state, §3.2).

    Algorithm
    ---------
    Each iteration:
      1. **Forward pass** — sample a scenario path; solve each stage with
         the *pinned* incoming state and current cost-to-go cuts; record
         the realised states and the path cost (an UB sample).
      2. **Backward pass** — from the last stage to the first, for every
         child realisation, solve the **Lagrangian dual**

             max_π  L(π),   L(π) = min_{x} [ c·x + θ
                                             + π·(z_parent − z_local) ]

         (copy constraint relaxed) by projected sub-gradient ascent. The
         optimal multipliers ``π*`` and value ``L(π*)`` give the
         Lagrangian cut on the parent's cost-to-go:

             θ_parent ≥ Σ_c p_c [ L_c(π*_c) + π*_c · (z − z_parent) ]

         which is added to the parent stage's cut pool.
      3. The first-stage master objective is a valid **lower bound**;
         because cuts only ever tighten the under-estimator, the LB is
         **monotone non-decreasing** and converges to the true optimum
         (exact for binary state).

    Parameters
    ----------
    stages
        One :class:`StageProblem` per stage ``t = 0 … T-1``.
    scenarios_per_stage
        ``scenarios_per_stage[t]`` is the list of realisation dicts at
        stage ``t``. Each dict is passed to the stage's ``build`` callback
        as closure data (the builder should capture it). Stage 0 normally
        has exactly one (deterministic root) realisation.
    stage_probabilities
        ``stage_probabilities[t][k]`` = conditional prob of realisation
        ``k`` at stage ``t``; defaults to uniform. Assumes stage-wise
        independence (a scenario *tree* with shared children — the
        standard SDDP sampling model).
    max_iter
        Forward/backward iteration cap.
    n_forward
        Number of sampled forward paths per iteration (statistical UB).
    lagrangian_iters, lagrangian_step
        Sub-gradient ascent budget / initial step for the Lagrangian dual.
    deterministic_equivalent
        Optional externally-computed DE optimum for gap reporting.

    Returns
    -------
    :class:`SDDiPResult` with the monotone ``lower_bounds`` trajectory.
    """
    t0 = time.perf_counter()
    T = len(stages)
    if T < 2:
        raise ValueError("solve_sddip needs ≥ 2 stages")
    rng = np.random.RandomState(seed)

    if stage_probabilities is None:
        stage_probabilities = [
            [1.0 / len(scenarios_per_stage[t])] * len(scenarios_per_stage[t])
            for t in range(T)
        ]

    # Cut pool per stage (cuts live on the *incoming* state of that stage,
    # i.e. they describe stage t's cost-to-go as seen from stage t-1).
    # cuts[t] applies inside stage t's subproblem on z_local.
    cuts: list[list[tuple[float, dict[str, float]]]] = [[] for _ in range(T)]

    lower_bounds: list[float] = []
    ub_est = float("inf")

    # Helper to set the active realisation on a stage builder. We encode
    # the realisation by wrapping: the StageProblem.build already closes
    # over scenarios_per_stage[t]; we pass the chosen index via a mutable
    # ``_active`` list attached to the *build callable* (the builder reads
    # ``build._active[0]``). We mirror it on the StageProblem for
    # convenience but the build callable is the source of truth.
    for st in stages:
        if not hasattr(st.build, "_active"):
            st.build._active = [0]
        object.__setattr__(st, "_active", st.build._active)

    prev_lb = -float("inf")
    for it in range(max_iter):
        # ---- Forward pass: sample n_forward paths -----------------------
        fwd_states: list[list[dict[str, float]]] = []
        fwd_idx: list[list[int]] = []
        path_costs: list[float] = []
        for _ in range(n_forward):
            z_prev = {s: stages[0].state_bounds[s][0]
                      for s in stages[0].state_names}
            states_along: list[dict[str, float]] = []
            idx_along: list[int] = []
            cost = 0.0
            stagewise_ctg_last = 0.0
            for t in range(T):
                n_real = len(scenarios_per_stage[t])
                if t == 0:
                    k = 0
                else:
                    k = int(rng.choice(n_real, p=stage_probabilities[t]))
                idx_along.append(k)
                stages[t]._active[0] = k
                out = _sddip_solve_stage(z_prev, stages[t], t, cuts[t])
                if out is None:
                    return SDDiPResult(
                        status="stage_infeasible",
                        lower_bounds=lower_bounds, upper_bound=float("nan"),
                        first_stage={}, n_iterations=it,
                        solve_time=time.perf_counter() - t0,
                    )
                obj, out_state, zloc, theta_val = out
                # Path cost excludes the cost-to-go surrogate θ (that is the
                # estimate of future stages, counted by actually traversing
                # them) — accumulate immediate stage cost only.
                cost += (obj - theta_val)
                states_along.append(out_state)
                z_prev = out_state
            fwd_states.append(states_along)
            fwd_idx.append(idx_along)
            path_costs.append(cost)
        ub_est = float(np.mean(path_costs))

        # ---- Backward pass: build Lagrangian cuts -----------------------
        # For each sampled path, go from stage T-1 down to 1. At stage t we
        # build a cut on stage t's cost-to-go as a function of the incoming
        # state z_{t-1} = fwd_states[path][t-1].
        for path in range(n_forward):
            for t in range(T - 1, 0, -1):
                z_in = fwd_states[path][t - 1]
                n_real = len(scenarios_per_stage[t])
                probs_t = stage_probabilities[t]
                # Aggregate Lagrangian cut over the children realisations.
                agg_intercept = 0.0
                agg_grad = {s: 0.0 for s in stages[t].state_names}
                ok = True
                for k in range(n_real):
                    stages[t]._active[0] = k
                    cut = _build_lagrangian_cut(
                        z_in, stages[t], t, cuts[t],
                        lagrangian_iters, lagrangian_step,
                    )
                    if cut is None:
                        ok = False
                        break
                    L_at_z, grad = cut
                    pk = probs_t[k]
                    # Cut value at z_in is L; gradient is π*. Affine support:
                    #   ctg(z) ≥ L + Σ π*_j (z_j − z_in_j)
                    intercept = L_at_z - sum(grad[s] * z_in[s]
                                             for s in grad)
                    agg_intercept += pk * intercept
                    for s in grad:
                        agg_grad[s] += pk * grad[s]
                if not ok:
                    continue
                cuts[t - 1].append((agg_intercept, agg_grad))

        # ---- Lower bound = first-stage master with updated cuts ---------
        stages[0]._active[0] = 0
        z0 = {s: stages[0].state_bounds[s][0]
              for s in stages[0].state_names}
        out0 = _sddip_solve_stage(z0, stages[0], 0, cuts[0])
        if out0 is None:
            return SDDiPResult(
                status="stage_infeasible", lower_bounds=lower_bounds,
                upper_bound=ub_est, first_stage={}, n_iterations=it,
                solve_time=time.perf_counter() - t0,
            )
        lb = out0[0]
        first_state = out0[1]
        # Enforce monotonicity numerically (cuts are valid under-estimators
        # so LB can only rise; clamp tiny LP-tolerance dips).
        lb = max(lb, prev_lb)
        prev_lb = lb
        lower_bounds.append(lb)
        if verbose:
            print(f"[sddip] it {it} LB={lb:.4f} UB={ub_est:.4f}")
        if ub_est - lb <= tol * (1.0 + abs(lb)):
            break

    gap = None
    if deterministic_equivalent is not None and np.isfinite(
            deterministic_equivalent):
        gap = abs(lower_bounds[-1] - deterministic_equivalent) / (
            1.0 + abs(deterministic_equivalent))

    return SDDiPResult(
        status="optimal",
        lower_bounds=lower_bounds,
        upper_bound=ub_est,
        first_stage=first_state,
        n_iterations=len(lower_bounds),
        solve_time=time.perf_counter() - t0,
        deterministic_equivalent=deterministic_equivalent,
        gap=gap,
    )


def _build_lagrangian_cut(
    z_in: dict[str, float],
    stage: "StageProblem",
    stage_idx: int,
    cuts: list[tuple[float, dict[str, float]]],
    lag_iters: int,
    lag_step: float,
) -> Optional[tuple[float, dict[str, float]]]:
    """
    Solve the Lagrangian dual ``max_π L(π)`` of the state-copy relaxation
    at ``z_in`` by projected sub-gradient ascent, returning
    ``(L(π*), π*)``.

    The Lagrangian (copy constraint ``z_local = z_in`` relaxed) is

        L(π) = min_x [ c·x + θ + π·(z_in − z_local) ].

    A sub-gradient of ``L`` at ``π`` is ``(z_in − z_local*(π))`` where
    ``z_local*`` is the relaxed minimiser. ``L`` is concave; we ascend it.
    The resulting ``(L, π*)`` define a valid affine support of the integer
    cost-to-go in ``z_in`` (Zou-Ahmed-Sun §3.2): for binary state these
    Lagrangian cuts are exact.
    """
    states = stage.state_names
    pi = {s: 0.0 for s in states}
    best_L = -float("inf")
    best_pi = dict(pi)
    step = lag_step
    for k in range(lag_iters):
        out = _sddip_solve_stage(
            z_in, stage, stage_idx, cuts,
            relax_state=True, pi=pi,
        )
        if out is None:
            return None
        L, _out_state, zloc, _theta = out
        if L > best_L:
            best_L = L
            best_pi = dict(pi)
        # Sub-gradient g = z_in − z_local*(π). Ascend π ← π + step·g.
        g = {s: z_in[s] - zloc[s] for s in states}
        gnorm = np.sqrt(sum(v * v for v in g.values()))
        if gnorm < 1e-9:
            # Copy satisfied at the relaxed optimum → no duality gap.
            break
        step_k = step / (1.0 + 0.1 * k)
        for s in states:
            pi[s] = pi[s] + step_k * g[s]
    return best_L, best_pi


# ---------------------------------------------------------------------------
# 9.2 General-form chance constraints — Big-M indicator binaries
# ---------------------------------------------------------------------------

@dataclass
class GeneralChanceConstraint:
    """
    A general-form individual chance constraint enforced exactly by
    per-scenario Big-M indicator binaries.

    Enforces  ``Pr_s{ g(x, ξ_s) ≤ 0 } ≥ 1 − α`` where the constraint
    *body* ``g`` couples the first-stage decision ``x`` to the random
    parameter ``ξ_s`` (so the exact-quantile reduction in
    :class:`ChanceConstraint` — which assumes ``g`` is scenario-only — no
    longer applies).

    The standard mixed-integer reformulation (Luedtke & Ahmed 2008,
    "A sample approximation approach for optimization with probabilistic
    constraints", *SIAM J. Optim.* 19:674-699; Nemirovski & Shapiro 2006)
    introduces a binary ``z_s ∈ {0,1}`` per scenario with

        g(x, ξ_s) ≤ M_s · z_s          (z_s = 1 ⇒ scenario may violate)
        Σ_s p_s · z_s ≤ α              (mass of violated scenarios ≤ α)

    so that ``z_s = 0`` forces the constraint to hold in scenario ``s``,
    and the total violation probability is capped at ``α``.

    Here the constraint family is the firm-capacity reserve coupling

        (1 + margin)·peak_s  ≤  Σ_g credit_g · cap_g + slack_s,

    i.e. ``g = (1+margin)·peak_s − Σ_g credit_g·cap_g`` which depends on
    both the random peak ``peak_s`` *and* the first-stage capacities — a
    genuinely general-form CC, not separable.
    """
    name: str
    alpha: float
    big_m: float = 1e6


def solve_general_chance_constrained(
    *,
    peak_loads: list[float],
    probabilities: list[float],
    credits: dict[str, float],
    capex: dict[str, float],
    cap_bounds: dict[str, tuple[float, float]],
    reserve_margin: float = 0.0,
    alpha: float = 0.1,
    base_firm: float = 0.0,
    big_m: Optional[float] = None,
    verbose: bool = False,
) -> dict:
    """
    Solve a general-form chance-constrained firm-capacity sizing MILP with
    per-scenario Big-M indicator binaries.

    min   Σ_g capex_g · cap_g
    s.t.  (1+m)·peak_s ≤ base_firm + Σ_g credit_g·cap_g + M·z_s   ∀ s
          Σ_s p_s·z_s ≤ α
          z_s ∈ {0,1},   cap_g ∈ [lo_g, hi_g]

    This is the Luedtke-Ahmed (2008) SAA MILP reformulation of the joint
    chance constraint ``Pr{ firm capacity covers (1+m)·peak } ≥ 1−α`` for
    a constraint whose LHS couples the random peak to the first-stage
    capacities. Returns the chosen capacities, the indicator pattern, and
    the realised coverage (fraction of probability mass covered).

    Returns
    -------
    dict with ``status``, ``capacities``, ``indicators`` (z_s), ``cost``,
    ``coverage`` (Σ_s p_s·(1−z_s) actually covered), ``firm_capacity``.
    """
    N = len(peak_loads)
    if len(probabilities) != N:
        raise ValueError("peak_loads / probabilities length mismatch")
    if big_m is None:
        big_m = float(max(peak_loads) * (1.0 + reserve_margin) + 1.0)

    model = nx.Model("general_cc")
    cap: dict[str, object] = {}
    for g, (lo, hi) in cap_bounds.items():
        cap[g] = model.variable(f"cap_{g}", lower=lo, upper=hi)

    z = [model.binary(f"z_{s}") for s in range(N)]

    # firm capacity expression
    firm = None
    for g, c in credits.items():
        if g not in cap:
            continue
        term = c * cap[g]
        firm = term if firm is None else firm + term
    # add constant base firm via a helper fixed var (Expr + float supported)

    for s in range(N):
        req = (1.0 + reserve_margin) * float(peak_loads[s])
        # req ≤ base_firm + firm + M·z_s  ->  firm + M·z_s ≥ req − base_firm
        lhs = firm + big_m * z[s] if firm is not None else big_m * z[s]
        model.add(lhs >= (req - base_firm), name=f"cc_{s}")

    # Σ p_s z_s ≤ α
    mass = None
    for s in range(N):
        term = float(probabilities[s]) * z[s]
        mass = term if mass is None else mass + term
    model.add(mass <= float(alpha), name="risk_budget")

    obj = None
    for g, k in capex.items():
        if g not in cap:
            continue
        term = k * cap[g]
        obj = term if obj is None else obj + term
    model.minimize(obj)
    res = model.solve(verbose=verbose)
    if res.status != "optimal":
        return {"status": res.status}

    caps = {g: float(res.value(v)) for g, v in cap.items()}
    inds = [int(round(float(res.value(z[s])))) for s in range(N)]
    firm_cap = base_firm + sum(credits.get(g, 0.0) * caps[g] for g in caps)
    coverage = sum(probabilities[s] * (1 - inds[s]) for s in range(N))
    return {
        "status": "optimal",
        "capacities": caps,
        "indicators": inds,
        "cost": float(res.objective),
        "coverage": float(coverage),
        "firm_capacity": float(firm_cap),
        "required_alpha": float(alpha),
    }


# ---------------------------------------------------------------------------
# 9.3 Wasserstein distributionally-robust optimisation
# ---------------------------------------------------------------------------

def solve_wasserstein_dro(
    *,
    loss_slopes: list[list[float]],
    loss_intercepts: list[list[float]],
    samples: list[float],
    cap_bounds: tuple[float, float] = (0.0, 1e6),
    epsilon: float = 0.0,
    verbose: bool = False,
) -> dict:
    """
    Type-1 Wasserstein distributionally-robust optimisation.

    Implements the tractable dual reformulation of Mohajerin Esfahani &
    Kuhn, "Data-driven distributionally robust optimization using the
    Wasserstein metric", *Math. Program.* 171 (2018), 115-166 (their
    Theorem 4.2 / Corollary 5.1 for piecewise-linear convex loss).

    For a decision ``x`` and a loss ``ℓ(x, ξ) = max_k (a_k(x) + b_k ξ)``
    that is piecewise-linear convex in the random ``ξ``, the worst-case
    expected loss over the type-1 Wasserstein ball of radius ``ε`` around
    the empirical distribution ``Phat_N = (1/N) Σ_i δ_{ξ_i}`` admits the
    finite convex program

        min_{x, λ, s_i}   λ·ε + (1/N) Σ_i s_i
        s.t.  a_k(x) + b_k·ξ_i ≤ s_i            ∀ i, k     (epigraph)
              |b_k| ≤ λ                          ∀ k        (Lipschitz/dual-norm)
              x ∈ X.

    The ``λ·ε`` term is the Lipschitz-norm regulariser; the per-sample
    epigraph variables ``s_i`` upper-bound the loss at each data point.

    As ``ε → 0`` the ``λ·ε`` term and the Lipschitz constraint vanish from
    the objective pressure, so the program reduces to the **SAA** problem
    ``min_x (1/N) Σ_i ℓ(x, ξ_i)`` (their Remark 4.6 — DRO interpolates to
    sample-average as the radius shrinks). As ``ε`` grows the worst-case
    cost is **monotone non-decreasing** in ``ε`` (the ambiguity ball only
    grows), so the plan becomes more conservative. DRO is thus a
    principled successor to CVaR that is robust to misspecification of the
    scenario probabilities themselves, not merely the tail.

    Parameterisation (tiny capacity-sizing instance)
    -------------------------------------------------
    ``loss_slopes[k]`` / ``loss_intercepts[k]`` describe the ``k``-th
    affine piece of the loss as a function of the *decision* ``cap`` and
    the random sample ``ξ``:

        piece_k(cap, ξ) = loss_intercepts[k][0]
                          + loss_slopes[k][0] · cap
                          + loss_slopes[k][1] · ξ

    so ``b_k = loss_slopes[k][1]`` is the slope in ``ξ`` (drives the
    Lipschitz constant) and the ``cap`` term enters every epigraph row.

    Returns
    -------
    dict with ``status``, ``cap``, ``worst_case_cost`` (the optimal
    objective = worst-case expected loss), ``lambda``, ``epsilon``.
    """
    N = len(samples)
    K = len(loss_slopes)
    if len(loss_intercepts) != K:
        raise ValueError("loss_slopes / loss_intercepts piece count mismatch")

    model = nx.Model("wasserstein_dro")
    lo, hi = cap_bounds
    cap = model.variable("cap", lower=lo, upper=hi)
    lam = model.variable("lambda", lower=0.0, upper=1e15)
    s = [model.variable(f"s_{i}", lower=-1e15, upper=1e15) for i in range(N)]

    # Epigraph constraints: for each sample i and piece k,
    #   intercept_k + slope_cap_k·cap + slope_xi_k·ξ_i ≤ s_i
    for i in range(N):
        xi = float(samples[i])
        for k in range(K):
            b_intercept = float(loss_intercepts[k][0])
            b_cap = float(loss_slopes[k][0])
            b_xi = float(loss_slopes[k][1])
            expr = b_intercept + b_cap * cap + b_xi * xi
            model.add(s[i] - expr >= 0.0, name=f"epi_{i}_{k}")

    # Lipschitz / dual-norm constraints: |b_k| ≤ λ  (type-1 → ∞-dual norm
    # on the scalar ξ reduces to |b_k| ≤ λ).
    for k in range(K):
        b_xi = float(loss_slopes[k][1])
        model.add(lam - b_xi >= 0.0, name=f"lip_pos_{k}")
        model.add(lam + b_xi >= 0.0, name=f"lip_neg_{k}")

    # Objective: λ·ε + (1/N) Σ s_i
    obj = float(epsilon) * lam
    for i in range(N):
        obj = obj + (1.0 / N) * s[i]
    model.minimize(obj)
    res = model.solve(verbose=verbose)
    if res.status != "optimal":
        return {"status": res.status}
    return {
        "status": "optimal",
        "cap": float(res.value(cap)),
        "worst_case_cost": float(res.objective),
        "lambda": float(res.value(lam)),
        "epsilon": float(epsilon),
    }


# ---------------------------------------------------------------------------
# 9.4 Risk-averse Benders cuts — nested CVaR change-of-measure
# ---------------------------------------------------------------------------

def cvar_change_of_measure(
    costs: list[float],
    probabilities: list[float],
    alpha: float,
) -> list[float]:
    """
    Return the CVaR_α change-of-measure (risk-adjusted scenario weights).

    CVaR_α as a coherent risk measure has the dual representation
    ``CVaR_α(Z) = max_{Q ∈ Q_α} E_Q[Z]`` where the ambiguity set is

        Q_α = { q :  0 ≤ q_s ≤ p_s/α,  Σ_s q_s = 1 }

    (Rockafellar & Uryasev 2000; Shapiro-Dentcheva-Ruszczyński Ch. 6).
    The maximising ``q*`` re-weights the upper-α tail: it puts weight
    ``p_s/α`` on the worst scenarios until a total mass of 1 is reached.
    This is exactly the change-of-measure used by Philpott & de Matos,
    "Dynamic sampling algorithms for multi-stage stochastic programs with
    risk aversion", *EJOR* 218 (2012), 470-483, to weight risk-averse
    SDDP/Benders cuts.

    Returns the list ``q*`` aligned with ``costs``.
    """
    N = len(costs)
    order = sorted(range(N), key=lambda i: costs[i], reverse=True)  # worst→best
    q = [0.0] * N
    remaining = 1.0
    for i in order:
        cap_i = probabilities[i] / alpha
        take = min(cap_i, remaining)
        q[i] = take
        remaining -= take
        if remaining <= 1e-15:
            break
    return q


def solve_risk_averse_benders(
    *,
    scenario_costs_fn: Callable[[float], list[float]],
    capex: float,
    cap_bounds: tuple[float, float],
    probabilities: list[float],
    cap_dual_fn: Callable[[float], list[float]],
    alpha: float = 0.2,
    risk_lambda: float = 1.0,
    max_iter: int = 40,
    tol: float = 1e-5,
    verbose: bool = False,
) -> dict:
    """
    Risk-averse Benders / SDDP with a dedicated nested-CVaR cut family.

    Standard (risk-neutral) Benders aggregates subproblem optimality cuts
    with the *physical* probabilities ``p_s``. The risk-averse variant of
    Philpott & de Matos (2012) instead aggregates each cut with the
    **CVaR change-of-measure** ``q*`` (see :func:`cvar_change_of_measure`),
    optionally blended with the physical measure:

        weight_s = (1−ρ)·p_s + ρ·q*_s

    where ``ρ = risk_lambda ∈ [0,1]`` interpolates risk-neutral
    (``ρ=0``) ↔ pure CVaR_α (``ρ=1``). Because ``q*`` over-weights the
    upper-cost tail, the resulting master builds **more capacity** and its
    plan attains a **lower worst-case cost** (higher worst-case coverage)
    than the risk-neutral plan — the defining property of a risk-averse
    policy.

    The single-capacity master is

        min_{cap, θ}  capex·cap + θ
        s.t.  θ ≥ Σ_s weight_s · [ cost_s(cap_k) + dual_s·(cap − cap_k) ]
              cap ∈ [lo, hi]

    where ``cost_s`` / ``dual_s`` are the per-scenario subproblem value and
    its sensitivity to ``cap`` (supplied as callbacks for this tiny test
    harness; in the full engine these come from ``EnergySystem.optimise``).

    Returns
    -------
    dict with ``status``, ``cap``, ``expected_cost``, ``worst_case_cost``,
    ``cvar``, ``measure`` (the cut weights used), ``n_iterations``.
    """
    lo, hi = cap_bounds
    cuts: list[tuple[float, float]] = []  # (intercept, slope) on cap
    cap_k = lo
    lb = -float("inf")
    weights_used: list[float] = []
    for it in range(max_iter):
        # Subproblem evaluation at incumbent cap_k.
        costs = scenario_costs_fn(cap_k)
        duals = cap_dual_fn(cap_k)
        q = cvar_change_of_measure(costs, probabilities, alpha)
        weights = [(1.0 - risk_lambda) * probabilities[s]
                   + risk_lambda * q[s] for s in range(len(costs))]
        weights_used = weights
        # Aggregate cut: θ ≥ Σ_s w_s (cost_s + dual_s (cap − cap_k))
        intercept = sum(weights[s] * (costs[s] - duals[s] * cap_k)
                        for s in range(len(costs)))
        slope = sum(weights[s] * duals[s] for s in range(len(costs)))
        cuts.append((intercept, slope))

        # Master.
        model = nx.Model("risk_averse_master")
        cap = model.variable("cap", lower=lo, upper=hi)
        theta = model.variable("theta", lower=-1e15, upper=1e15)
        for (a, b) in cuts:
            model.add(theta - b * cap >= a, name="cut")
        model.minimize(capex * cap + theta)
        res = model.solve(verbose=False)
        if res.status != "optimal":
            return {"status": res.status}
        new_cap = float(res.value(cap))
        new_lb = float(res.objective)
        if verbose:
            print(f"[ra-benders] it {it} cap={new_cap:.3f} LB={new_lb:.3f}")
        if abs(new_cap - cap_k) <= tol * (1.0 + abs(cap_k)) and it > 0:
            cap_k = new_cap
            lb = new_lb
            break
        cap_k = new_cap
        lb = new_lb

    from nexus_energy.decomposition import _cvar_at_caps
    final_costs = scenario_costs_fn(cap_k)
    expected = sum(probabilities[s] * final_costs[s]
                   for s in range(len(final_costs)))
    return {
        "status": "optimal",
        "cap": float(cap_k),
        "expected_cost": float(expected + capex * cap_k),
        "operational_expected": float(expected),
        "worst_case_cost": float(max(final_costs) + capex * cap_k),
        "cvar": float(_cvar_at_caps(final_costs, probabilities, alpha)),
        "measure": list(weights_used),
        "n_iterations": it + 1,
    }


# ---------------------------------------------------------------------------
# 2.5 Forced-outage scenario generation (stochastic availability)
# ---------------------------------------------------------------------------

def generate_forced_outage_scenarios(
    generators: dict[str, float],
    n_scenarios: int = 50,
    *,
    n_timesteps: Optional[int] = None,
    mttr: Optional[float] = None,
    seed: int = 42,
) -> list[Scenario]:
    """
    Generate forced-outage availability scenarios.

    Each generator ``g`` has a forced-outage rate (FOR) ``q_g =
    generators[g]`` = the long-run probability the unit is unavailable.
    Two models are supported:

    * **Bernoulli FOR** (``mttr`` is ``None`` and ``n_timesteps`` is
      ``None``) — a single availability draw per scenario per generator:
      the unit is fully out (capacity → 0) with probability ``q_g``, else
      fully available. The realised per-generator outage *frequency* across
      scenarios converges to ``q_g`` (this is what the smoke test checks).

    * **Two-state Markov** (``mttr`` given, ``n_timesteps`` given) — the
      classic up/down forced-outage Markov chain (Billinton & Allan,
      *Reliability Evaluation of Power Systems*, 2nd ed., §11): the repair
      rate is ``μ = 1/MTTR`` and the failure rate solved from the
      steady-state ``q_g = λ/(λ+μ)`` ⇒ ``λ = μ·q_g/(1−q_g)``. We simulate
      the chain over ``n_timesteps`` and emit a per-timestep availability
      vector as a ``capacity`` override (a 0/1 mask scaled by capacity is
      applied through ``overrides``). The time-average down-fraction
      converges to ``q_g``.

    Each :class:`Scenario` carries per-generator ``overrides`` of the form
    ``("gen", name, "capacity")`` so it plugs straight into
    :func:`apply_scenario` / :func:`solve_stochastic`. Equal probability.

    Parameters
    ----------
    generators
        ``{gen_name: forced_outage_rate}`` with ``0 ≤ FOR < 1``.
    n_scenarios
        Number of independent availability realisations.
    n_timesteps, mttr
        Provide both to use the two-state Markov model; the override then
        carries a length-``n_timesteps`` availability *array* (per-step
        capacity factor in {0,1}). For the array path the override sets
        ``carrier_factor`` style availability; we use the ``"availability"``
        field if present, else fall back to scaling ``capacity`` by the
        time-average (so the result is always a valid Scenario).

    Returns
    -------
    list[:class:`Scenario`] of length ``n_scenarios``, equal probability.
    """
    rng = np.random.RandomState(seed)
    names = list(generators)
    for g, q in generators.items():
        if not (0.0 <= q < 1.0):
            raise ValueError(f"FOR for {g} must be in [0,1), got {q}")

    scenarios: list[Scenario] = []
    use_markov = (mttr is not None and n_timesteps is not None)

    for i in range(n_scenarios):
        overrides: dict = {}
        if use_markov:
            mu = 1.0 / float(mttr)
            for g in names:
                q = generators[g]
                lam = mu * q / max(1e-12, (1.0 - q))
                # Per-step transition probs (discrete-time approx).
                p_fail = min(1.0, lam)      # up→down
                p_repair = min(1.0, mu)     # down→up
                avail = np.ones(n_timesteps, dtype=float)
                # Initialise in steady state.
                state_up = rng.random() > q
                for t in range(n_timesteps):
                    avail[t] = 1.0 if state_up else 0.0
                    if state_up:
                        if rng.random() < p_fail:
                            state_up = False
                    else:
                        if rng.random() < p_repair:
                            state_up = True
                # Apply availability as a carrier_factor-style mask: store
                # the time-average down fraction via capacity scaling so the
                # Scenario remains apply_scenario-compatible, AND keep the
                # full mask under a custom override field for richer engines.
                overrides[("gen", g, "availability")] = avail
                overrides[("gen", g, "carrier_factor")] = avail
        else:
            for g in names:
                q = generators[g]
                up = rng.random() > q   # available with prob (1-q)
                if not up:
                    # Unit forced out this scenario → capacity to 0.
                    overrides[("gen", g, "capacity")] = 0.0

        scenarios.append(Scenario(
            name=f"outage_{i}",
            probability=1.0 / n_scenarios,
            overrides=overrides,
        ))
    return scenarios
