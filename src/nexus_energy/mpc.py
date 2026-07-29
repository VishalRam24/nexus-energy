"""
Phase 14: Real-Time / Model Predictive Control.

Enables repeated re-optimisation as new data arrives:
- Parameter updates (demand forecast revisions, weather updates)
- Warm-start from previous solution (fast re-solve)
- Rolling control horizon with fixed control actions

Typical use case: an energy management system that re-optimises every 15
minutes with updated forecasts, applies only the first control action, then
repeats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from nexus_energy.core import EnergySystem, OptimisationResult


@dataclass
class MPCStep:
    """One MPC control step."""
    step: int
    time_index: int
    forecast_horizon: int
    result_status: str
    control_actions: dict  # applied control actions this step
    solve_time: float
    cost_this_step: float


class MPCController:
    """
    Model Predictive Controller for energy systems.

    Repeatedly re-optimises a shrinking horizon, applies only the first
    few control actions, advances time, updates forecasts, and repeats.

    Usage:
        >>> def system_factory(t, horizon):
        ...     sys = build_system(t, t + horizon)  # user-defined
        ...     return sys
        >>> mpc = MPCController(
        ...     system_factory=system_factory,
        ...     total_steps=8760,
        ...     control_horizon=24,
        ...     apply_steps=1,
        ... )
        >>> result = mpc.run()
    """

    def __init__(
        self,
        system_factory: Callable,
        total_steps: int,
        control_horizon: int = 24,
        apply_steps: int = 1,
        warm_start: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            system_factory: callable(start_t, horizon) -> EnergySystem
                Must build a new EnergySystem for the time window.
            total_steps: total simulation steps
            control_horizon: forecast horizon for each re-optimisation
            apply_steps: how many steps to actually apply before re-optimising
            warm_start: whether to pass previous solution as initial guess
            verbose: print progress
        """
        self.system_factory = system_factory
        self.total_steps = total_steps
        self.control_horizon = control_horizon
        self.apply_steps = apply_steps
        self.warm_start = warm_start
        self.verbose = verbose

        self.history: list[MPCStep] = []
        self.applied_dispatch: dict[str, list[float]] = {}
        self.applied_storage_soc: dict[str, list[float]] = {}

    def run(self) -> dict:
        """
        Run the MPC loop.

        Returns dict with:
            - total_cost: sum of applied step costs
            - history: list of MPCStep
            - applied_dispatch: concatenated dispatch for actually-applied timesteps
            - n_resolves: number of re-optimisations performed
        """
        import time as _time

        current_t = 0
        total_cost = 0.0
        prev_soc = None

        while current_t < self.total_steps:
            horizon = min(self.control_horizon, self.total_steps - current_t)
            if horizon <= 0:
                break

            # Build window system
            sys = self.system_factory(current_t, horizon)

            # Override initial SOC if we have it from previous step
            if prev_soc is not None:
                for sto in sys._storages:
                    if sto.name in prev_soc:
                        sto.soc_initial = prev_soc[sto.name] / sto.energy_capacity \
                            if sto.energy_capacity > 1e-6 else 0.5
                        sto.soc_initial = min(max(sto.soc_initial, 0), 1)
                        sto.cyclic = False  # no cyclic constraint in rolling mode

            # Solve
            t_solve_start = _time.perf_counter()
            result = sys.optimise(verbose=False)
            t_solve = _time.perf_counter() - t_solve_start

            if result.status != "optimal":
                if self.verbose:
                    print(f"MPC step {len(self.history)}: t={current_t} "
                          f"solver returned {result.status}; stopping.")
                return {
                    "status": result.status,
                    "total_cost": total_cost,
                    "history": self.history,
                    "applied_dispatch": {k: np.array(v) for k, v in self.applied_dispatch.items()},
                    "applied_storage_soc": {k: np.array(v) for k, v in self.applied_storage_soc.items()},
                    "n_resolves": len(self.history),
                    "stopped_at": current_t,
                }

            # Apply only the first `apply_steps` dispatch decisions
            apply_n = min(self.apply_steps, horizon)
            step_cost = 0.0

            for gen in sys._generators:
                if gen.name not in result.generator_dispatch:
                    continue
                applied = result.generator_dispatch[gen.name][:apply_n]
                self.applied_dispatch.setdefault(gen.name, []).extend(applied.tolist())
                step_cost += gen.marginal_cost * applied.sum() * sys._dt

            # Record SOC at end of applied window for carry-over
            next_soc = {}
            for sto in sys._storages:
                if sto.name not in result.storage_soc:
                    continue
                soc_arr = result.storage_soc[sto.name][:apply_n]
                self.applied_storage_soc.setdefault(sto.name, []).extend(soc_arr.tolist())
                next_soc[sto.name] = float(soc_arr[-1])
            prev_soc = next_soc

            # Record history
            step = MPCStep(
                step=len(self.history),
                time_index=current_t,
                forecast_horizon=horizon,
                result_status=result.status,
                control_actions={
                    g.name: result.generator_dispatch[g.name][:apply_n].tolist()
                    for g in sys._generators
                    if g.name in result.generator_dispatch
                },
                solve_time=t_solve,
                cost_this_step=step_cost,
            )
            self.history.append(step)
            total_cost += step_cost

            if self.verbose and len(self.history) % 10 == 0:
                print(f"MPC step {len(self.history)}: t={current_t}, "
                      f"cost={step_cost:.2f}, solve={t_solve*1000:.1f}ms")

            current_t += apply_n

        return {
            "status": "optimal",
            "total_cost": total_cost,
            "history": self.history,
            "applied_dispatch": {k: np.array(v) for k, v in self.applied_dispatch.items()},
            "applied_storage_soc": {k: np.array(v) for k, v in self.applied_storage_soc.items()},
            "n_resolves": len(self.history),
        }


def warm_start_resolve(
    system: "EnergySystem",
    previous_result: "OptimisationResult",
    updates: Optional[dict] = None,
) -> "OptimisationResult":
    """
    Re-solve a system with updated parameters using the previous solution
    as a warm start.

    Args:
        system: the EnergySystem (parameters mutated in place before calling)
        previous_result: result from the previous solve
        updates: optional dict of parameter updates to apply before solving.
            Keys can be "demand:<load_name>", "capacity:<gen_name>", etc.

    Returns:
        New OptimisationResult.

    Note: warm-start acceleration for MIP is handled via nexus-opt's internal
    warm_start parameter. For pure LP the simplex method handles warm starts
    automatically if the model structure is unchanged.
    """
    if updates is not None:
        for key, value in updates.items():
            if ":" not in key:
                continue
            kind, target = key.split(":", 1)
            if kind == "demand":
                for load in system._loads:
                    if load.name == target:
                        load.amount = value
            elif kind == "capacity":
                for gen in system._generators:
                    if gen.name == target:
                        gen.capacity = value
            elif kind == "marginal_cost":
                for gen in system._generators:
                    if gen.name == target:
                        gen.marginal_cost = value
            elif kind == "carrier_factor":
                for gen in system._generators:
                    if gen.name == target:
                        gen.carrier_factor = value

    # Re-solve (nexus-opt handles LP warm start automatically)
    return system.optimise()


def commitment_fixed_session(
    system: "EnergySystem",
    uc_result: "OptimisationResult",
    *,
    enforce_p_min: bool = True,
    **session_kwargs,
) -> tuple["PersistentDispatchSession", "OptimisationResult"]:
    """Phase 20.x.5 — UC once, then warm LP resolves with u* FIXED.

    Standard rolling-operations pattern: solve the MILP unit commitment
    once (``system.optimise()`` with committable units), then freeze the
    on/off schedule u* and serve the fast intra-period re-solves
    (demand/cf/mc/soc updates) from a :class:`PersistentDispatchSession`
    on the LP that remains.

    Mutates ``system`` in place: committable generators become
    fixed-schedule LP units (availability ×= u*[t]; UC cost terms drop —
    they are constants given u*). Per-period minimum stable output
    ``u*[t]·p_min`` is enforced via column lower bounds after build.

    Returns ``(session, base_result)``.
    """
    schedules = {}
    for g in system._generators:
        if not g.committable:
            continue
        u = uc_result.unit_status.get(g.name)
        if u is None:
            raise ValueError(f"uc_result has no unit_status for {g.name!r}")
        u = np.clip(np.round(np.asarray(u, dtype=float)), 0, None)
        schedules[g.name] = (u, g.p_min)
        cf = (np.ones(len(u)) if g.carrier_factor is None
              else np.asarray(g.carrier_factor, dtype=float))
        g.carrier_factor = cf * np.minimum(u, 1.0)
        g.committable = False
        g.p_min = 0.0
        g.startup_cost = g.shutdown_cost = g.startup_fuel_cost = 0.0
        g.no_load_cost = 0.0

    sess = PersistentDispatchSession(system, **session_kwargs)
    base = sess.build()
    if enforce_p_min and schedules:
        idxs, los, his = [], [], []
        lo_b, hi_b = sess._ph.col_bounds()
        for name, (u, p_min) in schedules.items():
            if p_min <= 0:
                continue
            for t, col in enumerate(sess._p_cols[name]):
                idxs.append(col)
                los.append(float(min(u[t], 1.0) * p_min))
                his.append(hi_b[col])
        if idxs:
            sess._ph.update_col_bounds(idxs, los, his)
            base = sess._extract(sess._ph.resolve())
    return sess, base


# ---------------------------------------------------------------------------
# Phase 18.a — persistent warm-started resolves (NO model rebuild)
# ---------------------------------------------------------------------------

class PersistentDispatchSession:
    """Rolling-horizon LP resolves without rebuilding the model.

    ``build()`` solves once through the normal ``optimise()`` path while
    capturing the assembled nexus-opt Model, then loads it into a
    persistent HiGHS instance. ``advance()`` pushes parameter changes
    (demand, availability, marginal cost, storage start SOC) straight
    into HiGHS columns/rows and re-solves — HiGHS hot-starts from its
    retained simplex basis, so a window resolve costs a handful of
    iterations instead of a full rebuild + cold solve.

    Honest scope (build() raises otherwise):
      * pure LP — no committable / integer-investment / switchable /
        PWL-capex components;
      * fixed capacities — extendable gens put availability into a
        ``p ≤ cap_var·cf`` matrix row, which this path cannot touch;
      * storage start-SOC carry-over needs ``soc_initial_free=True``
        (the start SOC is then a pinnable column; otherwise it is a
        constant folded into the t=0 row → structure change).
    Anything outside scope at advance() time falls back to a full
    rebuild — correct, just slower — and flags it in ``n_rebuilds``.
    """

    def __init__(self, system: "EnergySystem", *, verbose: bool = False,
                 threads: Optional[int] = None,
                 time_limit: Optional[float] = None):
        self.system = system
        self.verbose = verbose
        self.threads = threads
        self.time_limit = time_limit
        self._ph = None
        self._model = None
        self._built = False
        self.n_resolves = 0
        self.n_rebuilds = 0
        self.last_iterations: Optional[int] = None

    # ---- guards -----------------------------------------------------
    def _check_scope(self):
        sys = self.system
        bad = []
        for g in sys._generators:
            if g.committable:
                bad.append(f"generator {g.name!r}: committable")
            if g.extendable:
                bad.append(f"generator {g.name!r}: extendable")
            if g.integer_investment or g.capex_segments or g.heat_rate_segments:
                bad.append(f"generator {g.name!r}: integer/PWL features")
        for lk in sys._links:
            if lk.committable or lk.switchable or lk.integer_investment or lk.extendable:
                bad.append(f"link {lk.name!r}: UC/switch/integer/extendable")
        for st in sys._storages:
            if st.extendable:
                bad.append(f"storage {st.name!r}: extendable")
        if bad:
            raise NotImplementedError(
                "PersistentDispatchSession is LP-dispatch-only; rebuild-free "
                "updates cannot express: " + "; ".join(bad))

    # ---- build ------------------------------------------------------
    def build(self, **optimise_kwargs) -> "OptimisationResult":
        import nexus_opt as nx

        self._check_scope()
        if "model_hook" in optimise_kwargs:
            raise ValueError("the session owns model_hook")
        sys = self.system
        captured: dict = {}

        def _hook(model, _system, _obj):
            captured["model"] = model
            return None

        result = sys.optimise(model_hook=_hook, **optimise_kwargs)
        if result.status != "optimal":
            raise RuntimeError(f"base solve not optimal: {result.status}")
        self._model = captured["model"]
        self._optimise_kwargs = dict(optimise_kwargs)

        raw = result._raw
        names = raw.var_names_list
        if callable(names):
            names = names()
        col_of = {nm: i for i, nm in enumerate(names)}
        T = sys._timesteps
        self._T = T
        # Cost/bound mirrors of the builder (core.py optimise()).
        self._w = (np.ones(T) if sys._snapshot_weights is None
                   else np.asarray(sys._snapshot_weights, dtype=float))
        self._dts = (np.asarray(sys._snapshot_durations, dtype=float)
                     if getattr(sys, "_snapshot_durations", None) is not None
                     else np.full(T, float(sys._dt)))
        self._co2_price = sys._co2_price or 0.0

        self._p_cols: dict[str, list[int]] = {}
        self._p_vars: dict[str, list] = {}
        for g in sys._generators:
            if T == 1:
                cols = [col_of[f"p_{g.name}"]]
            else:
                cols = [col_of[f"p_{g.name}_{t}"] for t in range(T)]
            self._p_cols[g.name] = cols
            self._p_vars[g.name] = list(g._p_vars)
        self._soc_start_col = {
            st.name: col_of[f"soc_start_{st.name}"]
            for st in sys._storages
            if st._soc_start_var is not None and f"soc_start_{st.name}" in col_of
        }
        self._sto_vars = {
            st.name: (list(st._e_vars) if st._e_vars else None,
                      list(st._charge_vars) if st._charge_vars else None,
                      list(st._discharge_vars) if st._discharge_vars else None,
                      list(st._soc_vars) if st._soc_vars else None)
            for st in sys._storages
        }
        self._brow = dict(result._balance_row_idx or {})

        # Base demand per bus (sum of loads) — for exact RHS deltas.
        self._base_demand: dict[str, np.ndarray] = {}
        for ld in sys._loads:
            amt = ld.amount
            row = (np.full(T, float(amt))
                   if np.isscalar(amt) or isinstance(amt, (int, float))
                   else np.asarray(amt, dtype=float))
            self._base_demand[ld.bus.name] = \
                self._base_demand.get(ld.bus.name, np.zeros(T)) + row

        kw = {"model": self._model, "verbose": self.verbose}
        if self.threads is not None:
            kw["threads"] = self.threads
        if self.time_limit is not None:
            kw["time_limit"] = self.time_limit
        self._ph = nx.PersistentHighs.from_model(**kw)
        self._base_row_lower, self._base_row_upper = self._ph.row_bounds()
        self._built = True
        self._last_result = result
        return result

    # ---- advance ----------------------------------------------------
    def advance(self, *, demand: Optional[dict] = None,
                cf: Optional[dict] = None,
                mc: Optional[dict] = None,
                soc_init: Optional[dict] = None) -> "OptimisationResult":
        """Apply parameter updates in-place and hot-resolve.

        Args:
            demand: bus_name → (T,) TOTAL demand on that bus.
            cf: gen_name → (T,) availability in [0, 1].
            mc: gen_name → scalar or (T,) marginal cost.
            soc_init: storage_name → start SOC in MWh (requires
                ``soc_initial_free=True`` storage).
        """
        if not self._built:
            raise RuntimeError("call build() first")
        sys = self.system
        T = self._T
        gen_by = {g.name: g for g in sys._generators}
        sto_by = {st.name: st for st in sys._storages}

        # ---- structure checks → rebuild fallback ----
        def _rebuild():
            self.n_rebuilds += 1
            self._apply_to_system(demand, cf, mc, soc_init)
            return self.build(**self._optimise_kwargs)

        for name in (cf or {}):
            if name not in gen_by:
                return _rebuild()
        for name in (mc or {}):
            if name not in gen_by:
                return _rebuild()
        for name in (soc_init or {}):
            if name not in self._soc_start_col:
                return _rebuild()
        for bus_name, arr in (demand or {}).items():
            if bus_name not in self._base_demand or \
                    np.asarray(arr).shape != (T,):
                return _rebuild()
            if any((bus_name, t) not in self._brow for t in range(T)):
                return _rebuild()  # pure-constant bus rows can't be updated

        self._apply_to_system(demand, cf, mc, soc_init)

        # ---- demand → balance-row RHS deltas ----
        if demand:
            idxs, los, his = [], [], []
            for bus_name, arr in demand.items():
                arr = np.asarray(arr, dtype=float)
                delta = arr - self._base_demand[bus_name]
                for t in range(T):
                    r = self._brow[(bus_name, t)]
                    idxs.append(r)
                    los.append(self._base_row_lower[r] + delta[t])
                    his.append(self._base_row_upper[r] + delta[t])
            self._ph.update_row_bounds(idxs, los, his)

        # ---- cf → p-var upper bounds (mirror of core.py p-var build) ----
        if cf:
            idxs, los, his = [], [], []
            for name, arr in cf.items():
                g = gen_by[name]
                arr = np.asarray(arr, dtype=float)
                lower = 0.0 if g.committable else g.p_min
                upper_base = g.capacity * (g.n_units if g.clustered else 1)
                for t in range(T):
                    idxs.append(self._p_cols[name][t])
                    los.append(lower)
                    his.append(upper_base * float(arr[t]))
            self._ph.update_col_bounds(idxs, los, his)

        # ---- mc → column costs (exact builder mirror, incl. CO2/weights) ----
        if mc:
            idxs, costs = [], []
            for name, val in mc.items():
                g = gen_by[name]
                eff = np.asarray(val, dtype=float) + \
                    self._co2_price * g.emission_factor
                ptc = sys._ptc.get(g.tech, 0.0) if g.tech else 0.0
                eff = eff - ptc
                for t in range(T):
                    e_t = float(eff[t]) if eff.ndim else float(eff)
                    idxs.append(self._p_cols[name][t])
                    costs.append(e_t * self._w[t] * self._dts[t])
            self._ph.update_col_costs(idxs, costs)

        # ---- soc_init → pin the start-SOC column ----
        if soc_init:
            idxs, los, his = [], [], []
            for name, val in soc_init.items():
                idxs.append(self._soc_start_col[name])
                los.append(float(val))
                his.append(float(val))
            self._ph.update_col_bounds(idxs, los, his)

        # ---- hot resolve ----
        raw = self._ph.resolve()
        self.n_resolves += 1
        try:
            self.last_iterations = raw.iterations
        except Exception:
            self.last_iterations = None
        return self._extract(raw)

    def update_coeffs(self, rows, cols, values) -> None:
        """Expert API (Phase 20.x.4): change constraint-matrix
        coefficients in place (HiGHS changeCoeff) — e.g. storage
        efficiency terms in SOC rows. The CALLER must keep the mirrored
        EnergySystem consistent (set the matching component fields), or
        the next rebuild will silently diverge. Prefer a rebuild unless
        the resolve cadence genuinely demands the hot path."""
        if not self._built:
            raise RuntimeError("call build() first")
        self._ph.update_matrix_coeffs(list(rows), list(cols), list(values))

    # ---- helpers ------------------------------------------------------
    def _apply_to_system(self, demand, cf, mc, soc_init):
        """Mirror updates into the EnergySystem so any rebuild stays
        consistent with what the persistent model was told."""
        sys = self.system
        for bus_name, arr in (demand or {}).items():
            arr = np.asarray(arr, dtype=float)
            remaining = [ld for ld in sys._loads if ld.bus.name == bus_name]
            if remaining:
                base_others = np.zeros(self._T)
                for ld in remaining[1:]:
                    amt = ld.amount
                    base_others += (np.full(self._T, float(amt))
                                    if np.isscalar(amt) or isinstance(amt, (int, float))
                                    else np.asarray(amt, dtype=float))
                remaining[0].amount = arr - base_others
            # NOTE: self._base_demand stays frozen at BUILD-time values —
            # row-bound updates are absolute deltas vs the build baseline.
        for name, arr in (cf or {}).items():
            for g in sys._generators:
                if g.name == name:
                    g.carrier_factor = np.asarray(arr, dtype=float)
        for name, val in (mc or {}).items():
            for g in sys._generators:
                if g.name == name:
                    g.marginal_cost = val
        for name, val in (soc_init or {}).items():
            for st in sys._storages:
                if st.name == name and st.energy_capacity > 1e-9:
                    if st._soc_start_var is None:
                        st.soc_initial = min(max(
                            float(val) / st.energy_capacity, 0.0), 1.0)

    def _extract(self, raw) -> "OptimisationResult":
        from nexus_energy.core import OptimisationResult
        sys = self.system
        T = self._T
        result = OptimisationResult(
            status=raw.status,
            total_cost=raw.objective if raw.objective is not None else float("nan"),
            solve_time=raw.solve_time if hasattr(raw, "solve_time") else 0.0,
            _raw=raw,
            _balance_row_idx=dict(self._brow),
        )
        if raw.status not in ("optimal", "time_limit"):
            return result
        for name, pvars in self._p_vars.items():
            result.generator_dispatch[name] = np.array(
                [raw.value(v) for v in pvars])
        for name, (e_v, ch_v, dis_v, soc_v) in self._sto_vars.items():
            if e_v:
                result.storage_soc[name] = np.array(
                    [raw.value(v) for v in e_v])
            elif soc_v:
                result.storage_soc[name] = np.array(
                    [raw.value(v) for v in soc_v])
                if ch_v:
                    result.storage_charge[name] = np.array(
                        [raw.value(v) for v in ch_v])
                if dis_v:
                    result.storage_discharge[name] = np.array(
                        [raw.value(v) for v in dis_v])
        try:
            duals = raw.duals
        except Exception:
            duals = None
        if duals is not None and self._brow:
            for bus in sys._buses:
                arr = np.full(T, np.nan)
                for t in range(T):
                    idx = self._brow.get((bus.name, t))
                    if idx is not None and 0 <= idx < len(duals):
                        arr[t] = float(duals[idx]) / float(sys._dt)
                result.bus_shadow_prices[bus.name] = arr
        return result
