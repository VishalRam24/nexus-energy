"""
Perfect-foresight (and myopic) multi-stage investment planning.

Builds one LP spanning multiple stage-years (e.g. 2030, 2040, 2050) with
vintage-aware capacity accounting: a new-build at stage S is active at stage
S' iff ``year(S) + build_lead_years <= year(S') < year(S) + lifetime_years``.

Supports (Phase 5):
    * Generator / Storage / Link dispatch + extendable capacity per stage.
    * Vintage tracking via ``lifetime_years`` on each tech.
    * Construction lead time via ``build_lead_years``.
    * Scheduled retirement via ``retire_at_year``.
    * Multi-bus stages (per-bus balance, link flows across buses).
    * Discrete transmission expansion via ``Link.integer_investment``.
    * Myopic rolling (solve stages sequentially, freeze earlier stages)
      via ``optimise(myopic=True)``.

Not yet supported (deferred to 10.x+):
    * Endogenous retirement decision variables.
    * PWL CapEx on extendable storages / links.
    * Retrofit / fuel switching (link-level technology replacement).
    * SOCP AC-OPF within a multi-stage planner (DC-OPF stays transport).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import nexus as nx
except ImportError:
    import nexus_opt as nx

from .core import EnergySystem, Generator, Storage, Link


@dataclass
class MultiStageResult:
    """Result of a multi-stage solve (perfect-foresight or myopic)."""
    status: str
    total_cost: float
    solve_time: float
    years: list[int] = field(default_factory=list)
    new_builds: dict[str, np.ndarray] = field(default_factory=dict)
    capacity_active: dict[str, np.ndarray] = field(default_factory=dict)
    # Storage vintaging — separate power and energy capacity tracks.
    storage_new_power: dict[str, np.ndarray] = field(default_factory=dict)
    storage_new_energy: dict[str, np.ndarray] = field(default_factory=dict)
    storage_power_active: dict[str, np.ndarray] = field(default_factory=dict)
    storage_energy_active: dict[str, np.ndarray] = field(default_factory=dict)
    # Link vintaging
    link_new_builds: dict[str, np.ndarray] = field(default_factory=dict)
    link_capacity_active: dict[str, np.ndarray] = field(default_factory=dict)
    # Per-stage dispatch dicts: stage_dispatch[s][gen_name] = np.array[T]
    stage_dispatch: list[dict[str, np.ndarray]] = field(default_factory=list)
    stage_link_flow: list[dict[str, np.ndarray]] = field(default_factory=list)
    stage_storage_soc: list[dict[str, np.ndarray]] = field(default_factory=list)
    _raw: object = field(default=None, repr=False)


def _vintage_active(build_year: int, target_year: int, lead: int, life: int) -> bool:
    """True iff a vintage commissioned at ``build_year`` is live at ``target_year``."""
    return (build_year + lead) <= target_year < (build_year + life)


class MultiStageSystem:
    """Multi-stage planning wrapper over one :class:`EnergySystem` per stage.

    Each stage is a full system snapshot: its own timesteps, loads, generators,
    storages, links. Components are matched *by name* across stages — if
    ``solar`` is extendable in stage 0 and also present in stage 1, the
    stage-1 active capacity equals the live vintage from stage 0 (if still
    within its lifetime) plus any new stage-1 build.

    Buses and links must share the same names across stages (the network
    topology is fixed; only *capacities* change across the horizon).
    """

    def __init__(self, name: str = "multistage"):
        self.name = name
        self._stages: list[tuple[int, EnergySystem]] = []

    def add_stage(self, year: int, system: EnergySystem) -> None:
        """Append a stage. Stages are sorted by ``year`` at solve time."""
        self._stages.append((year, system))

    # ------------------------------------------------------------------
    # Solve pipeline
    # ------------------------------------------------------------------

    def optimise(self, *, solver: Optional[str] = None, verbose: bool = False,
                 presolve: bool = True, myopic: bool = False) -> MultiStageResult:
        if not self._stages:
            raise ValueError("MultiStageSystem: no stages registered.")

        if myopic:
            return self._optimise_myopic(
                solver=solver, verbose=verbose, presolve=presolve)
        return self._optimise_perfect_foresight(
            solver=solver, verbose=verbose, presolve=presolve)

    # ------------------------------------------------------------------

    def _optimise_perfect_foresight(
            self, *, solver: Optional[str], verbose: bool, presolve: bool,
            fixed_newbuilds: Optional[dict] = None) -> MultiStageResult:
        """Build and solve the full multi-stage LP.

        When ``fixed_newbuilds`` is provided (myopic use), the given
        ``{(kind, name, s_idx): value}`` entries pin the corresponding new-
        build variable to that value. ``kind ∈ {"gen", "stor_p", "stor_e",
        "link"}``.
        """
        stages = sorted(self._stages, key=lambda t: t[0])
        years = [s[0] for s in stages]
        systems = [s[1] for s in stages]
        S = len(stages)

        fixed = fixed_newbuilds or {}

        model = nx.Model()

        # ------------------------------------------------------------------
        # Vintage variables per component
        # ------------------------------------------------------------------
        gen_names = _collect_names(systems, lambda sys: sys._generators)
        stor_names = _collect_names(systems, lambda sys: sys._storages)
        link_names = _collect_names(systems, lambda sys: sys._links)
        bus_names = _collect_names(systems, lambda sys: sys._buses)

        gen_by_stage: dict[tuple[str, int], Generator] = {}
        stor_by_stage: dict[tuple[str, int], Storage] = {}
        link_by_stage: dict[tuple[str, int], Link] = {}
        for s_idx, sys in enumerate(systems):
            for g in sys._generators:
                gen_by_stage[(g.name, s_idx)] = g
            for st in sys._storages:
                stor_by_stage[(st.name, s_idx)] = st
            for lk in sys._links:
                link_by_stage[(lk.name, s_idx)] = lk

        # Generators
        gen_newbuild = {n: [None] * S for n in gen_names}
        gen_brownfield = {n: np.zeros(S) for n in gen_names}
        for s_idx, sys in enumerate(systems):
            year = years[s_idx]
            for g in sys._generators:
                retired = (g.retire_at_year is not None and
                           year >= g.retire_at_year)
                if g.extendable:
                    lo = g.min_capacity
                    hi = g.max_capacity if g.max_capacity != float("inf") else 1e12
                    if retired:
                        lo, hi = 0.0, 0.0
                    pin = fixed.get(("gen", g.name, s_idx))
                    if pin is not None:
                        lo = hi = float(pin)
                    v = model.variable(
                        f"newbuild_{g.name}_s{s_idx}", lower=lo, upper=hi)
                    gen_newbuild[g.name][s_idx] = v
                else:
                    gen_brownfield[g.name][s_idx] = 0.0 if retired else g.capacity

        # Storage — power & energy capacity are separately extendable
        stor_new_p = {n: [None] * S for n in stor_names}
        stor_new_e = {n: [None] * S for n in stor_names}
        stor_brownfield_p = {n: np.zeros(S) for n in stor_names}
        stor_brownfield_e = {n: np.zeros(S) for n in stor_names}
        for s_idx, sys in enumerate(systems):
            year = years[s_idx]
            for st in sys._storages:
                retired = (st.retire_at_year is not None and
                           year >= st.retire_at_year)
                if st.extendable:
                    lo_p = st.min_power_capacity
                    hi_p = st.max_power_capacity if st.max_power_capacity != float("inf") else 1e12
                    lo_e = st.min_energy_capacity
                    hi_e = st.max_energy_capacity if st.max_energy_capacity != float("inf") else 1e12
                    if retired:
                        lo_p = hi_p = lo_e = hi_e = 0.0
                    pin_p = fixed.get(("stor_p", st.name, s_idx))
                    pin_e = fixed.get(("stor_e", st.name, s_idx))
                    if pin_p is not None:
                        lo_p = hi_p = float(pin_p)
                    if pin_e is not None:
                        lo_e = hi_e = float(pin_e)
                    vp = model.variable(
                        f"newbuild_pwr_{st.name}_s{s_idx}", lower=lo_p, upper=hi_p)
                    ve = model.variable(
                        f"newbuild_en_{st.name}_s{s_idx}", lower=lo_e, upper=hi_e)
                    stor_new_p[st.name][s_idx] = vp
                    stor_new_e[st.name][s_idx] = ve
                else:
                    if not retired:
                        stor_brownfield_p[st.name][s_idx] = st.power_capacity
                        stor_brownfield_e[st.name][s_idx] = st.energy_capacity

        # Links
        link_newbuild = {n: [None] * S for n in link_names}
        link_brownfield = {n: np.zeros(S) for n in link_names}
        for s_idx, sys in enumerate(systems):
            year = years[s_idx]
            for lk in sys._links:
                retired = (lk.retire_at_year is not None and
                           year >= lk.retire_at_year)
                if lk.extendable:
                    lo = lk.min_capacity
                    hi = lk.max_capacity if lk.max_capacity != float("inf") else 1e12
                    if retired:
                        lo = hi = 0.0
                    pin = fixed.get(("link", lk.name, s_idx))
                    if pin is not None:
                        lo = hi = float(pin)
                    if lk.integer_investment and pin is None:
                        # units × unit_size — integer count, continuous bound
                        # rounded to unit lattice.
                        n_hi = int(np.ceil(hi / max(lk.unit_size, 1e-9))) if hi < 1e12 else 10000
                        n_lo = int(np.ceil(lo / max(lk.unit_size, 1e-9)))
                        n_var = model.integer(
                            f"newbuild_units_{lk.name}_s{s_idx}",
                            lower=n_lo, upper=n_hi)
                        v = model.variable(
                            f"newbuild_{lk.name}_s{s_idx}", lower=lo, upper=hi)
                        model.add(v == lk.unit_size * n_var,
                                  name=f"units_link_{lk.name}_s{s_idx}")
                        link_newbuild[lk.name][s_idx] = v
                    else:
                        v = model.variable(
                            f"newbuild_{lk.name}_s{s_idx}", lower=lo, upper=hi)
                        link_newbuild[lk.name][s_idx] = v
                else:
                    if not retired:
                        link_brownfield[lk.name][s_idx] = lk.capacity

        # ------------------------------------------------------------------
        # Active-capacity expressions per component × stage
        # ------------------------------------------------------------------
        gen_cap_active = _build_vintage_expr(
            gen_names, years, gen_newbuild, gen_brownfield, gen_by_stage)
        stor_cap_p_active = _build_vintage_expr(
            stor_names, years, stor_new_p, stor_brownfield_p, stor_by_stage)
        stor_cap_e_active = _build_vintage_expr(
            stor_names, years, stor_new_e, stor_brownfield_e, stor_by_stage)
        link_cap_active = _build_vintage_expr(
            link_names, years, link_newbuild, link_brownfield, link_by_stage)

        # ------------------------------------------------------------------
        # Retrofit / fuel-switching constraints
        # ------------------------------------------------------------------
        # A retrofit's new-build at stage S can't exceed the amount of its
        # host's capacity retiring between S-1 and S (brownfield that drops
        # to zero + vintage new-builds whose lifetime expires at that
        # boundary). For s=0 there's no prior state so retrofit is forced
        # to zero at stage 0.
        for s_idx, sys in enumerate(systems):
            for g in sys._generators:
                if g.retrofit_of is None:
                    continue
                nb_var = gen_newbuild[g.name][s_idx]
                if nb_var is None:
                    continue
                if g.retrofit_of not in gen_newbuild:
                    raise ValueError(
                        f"Generator {g.name!r} retrofit_of={g.retrofit_of!r}: "
                        f"host generator not present in any stage.")
                host_retiring = _retiring_amt(
                    g.retrofit_of, s_idx, years,
                    gen_newbuild, gen_brownfield, gen_by_stage)
                if host_retiring is None:
                    # No retiring host capacity at this boundary → retrofit = 0
                    model.add(nb_var <= 0.0,
                              name=f"retrofit_{g.name}_s{s_idx}_nohost")
                else:
                    model.add(nb_var <= host_retiring,
                              name=f"retrofit_{g.name}_s{s_idx}")

        # Phase 5.1 — retrofit for storages (power + energy tracks) and links.
        # Same semantics as the generator retrofit above: new-build at stage S
        # is capped by the host component's capacity retiring at that boundary.
        for s_idx, sys in enumerate(systems):
            for st in sys._storages:
                if st.retrofit_of is None:
                    continue
                if st.retrofit_of not in stor_new_p:
                    raise ValueError(
                        f"Storage {st.name!r} retrofit_of={st.retrofit_of!r}: "
                        f"host storage not present in any stage.")
                for track, new_dict, brown_dict in (
                    ("pwr", stor_new_p, stor_brownfield_p),
                    ("en", stor_new_e, stor_brownfield_e),
                ):
                    nb_var = new_dict[st.name][s_idx]
                    if nb_var is None:
                        continue
                    host_retiring = _retiring_amt(
                        st.retrofit_of, s_idx, years,
                        new_dict, brown_dict, stor_by_stage)
                    if host_retiring is None:
                        model.add(nb_var <= 0.0,
                                  name=f"retrofit_{st.name}_{track}_s{s_idx}_nohost")
                    else:
                        model.add(nb_var <= host_retiring,
                                  name=f"retrofit_{st.name}_{track}_s{s_idx}")
            for lk in sys._links:
                if lk.retrofit_of is None:
                    continue
                nb_var = link_newbuild[lk.name][s_idx]
                if nb_var is None:
                    continue
                if lk.retrofit_of not in link_newbuild:
                    raise ValueError(
                        f"Link {lk.name!r} retrofit_of={lk.retrofit_of!r}: "
                        f"host link not present in any stage.")
                host_retiring = _retiring_amt(
                    lk.retrofit_of, s_idx, years,
                    link_newbuild, link_brownfield, link_by_stage)
                if host_retiring is None:
                    model.add(nb_var <= 0.0,
                              name=f"retrofit_{lk.name}_s{s_idx}_nohost")
                else:
                    model.add(nb_var <= host_retiring,
                              name=f"retrofit_{lk.name}_s{s_idx}")

        # ------------------------------------------------------------------
        # Stage dispatch LP
        # ------------------------------------------------------------------
        stage_p_vars: list[dict[str, list]] = [dict() for _ in stages]
        stage_charge: list[dict[str, list]] = [dict() for _ in stages]
        stage_discharge: list[dict[str, list]] = [dict() for _ in stages]
        stage_soc: list[dict[str, list]] = [dict() for _ in stages]
        stage_flow: list[dict[str, list]] = [dict() for _ in stages]

        obj = None

        for s_idx, sys in enumerate(systems):
            T = sys._timesteps
            dt = sys._dt

            # Per-bus balance accumulators
            bus_expr: dict[str, list] = {b.name: [None] * T for b in sys._buses}

            # Generators
            for g in sys._generators:
                p_list = []
                cap_expr = gen_cap_active[g.name][s_idx]
                for t in range(T):
                    cf = 1.0
                    if g.carrier_factor is not None:
                        cf = float(g.carrier_factor[t])
                    loose_hi = g.max_capacity if g.extendable and g.max_capacity != float("inf") else (
                        g.capacity if not g.extendable else 1e12)
                    v = model.variable(
                        f"p_{g.name}_s{s_idx}_t{t}", lower=0.0,
                        upper=max(0.0, loose_hi * cf))
                    p_list.append(v)
                    if cap_expr is not None:
                        model.add(v <= cap_expr * cf,
                                  name=f"cap_{g.name}_s{s_idx}_t{t}")
                    else:
                        model.add(v <= 0.0,
                                  name=f"nocap_{g.name}_s{s_idx}_t{t}")
                    bus_expr[g.bus.name][t] = _add(bus_expr[g.bus.name][t], v)
                stage_p_vars[s_idx][g.name] = p_list

            # Storages
            for st in sys._storages:
                c_list, d_list, soc_list = [], [], []
                cap_p_expr = stor_cap_p_active[st.name][s_idx]
                cap_e_expr = stor_cap_e_active[st.name][s_idx]
                loose_p = st.max_power_capacity if st.extendable and st.max_power_capacity != float("inf") else (
                    st.power_capacity if not st.extendable else 1e12)
                loose_e = st.max_energy_capacity if st.extendable and st.max_energy_capacity != float("inf") else (
                    st.energy_capacity if not st.extendable else 1e12)
                for t in range(T):
                    c = model.variable(
                        f"c_{st.name}_s{s_idx}_t{t}", lower=0.0, upper=max(0.0, loose_p))
                    d = model.variable(
                        f"d_{st.name}_s{s_idx}_t{t}", lower=0.0, upper=max(0.0, loose_p))
                    soc = model.variable(
                        f"soc_{st.name}_s{s_idx}_t{t}", lower=0.0, upper=max(0.0, loose_e))
                    c_list.append(c); d_list.append(d); soc_list.append(soc)
                    if cap_p_expr is not None:
                        model.add(c <= cap_p_expr,
                                  name=f"cpwr_{st.name}_s{s_idx}_t{t}")
                        model.add(d <= cap_p_expr,
                                  name=f"dpwr_{st.name}_s{s_idx}_t{t}")
                    else:
                        model.add(c <= 0.0, name=f"cpwr0_{st.name}_s{s_idx}_t{t}")
                        model.add(d <= 0.0, name=f"dpwr0_{st.name}_s{s_idx}_t{t}")
                    if cap_e_expr is not None:
                        model.add(soc <= cap_e_expr * st.soc_max,
                                  name=f"soc_max_{st.name}_s{s_idx}_t{t}")
                        model.add(soc >= cap_e_expr * st.soc_min,
                                  name=f"soc_min_{st.name}_s{s_idx}_t{t}")
                    else:
                        model.add(soc <= 0.0,
                                  name=f"soc0_{st.name}_s{s_idx}_t{t}")
                    # Storage appears on the bus: +discharge - charge
                    bus_expr[st.bus.name][t] = _add(bus_expr[st.bus.name][t], d)
                    bus_expr[st.bus.name][t] = _sub(bus_expr[st.bus.name][t], c)
                # SOC recurrence: soc[t] = (1 - sd)·soc[t-1] + η_c·c - d/η_d
                for t in range(T):
                    eta_c = st.efficiency_charge
                    eta_d = st.efficiency_discharge
                    sd = st.self_discharge
                    inv_eta_d = 1.0 / eta_d
                    if t == 0:
                        if cap_e_expr is not None:
                            # soc[0] = soc_initial·cap_e + charge/discharge at t=0
                            lhs = soc_list[0]
                            rhs = cap_e_expr * st.soc_initial + eta_c * c_list[0] * dt - inv_eta_d * d_list[0] * dt
                            model.add(lhs == rhs,
                                      name=f"soc_init_{st.name}_s{s_idx}")
                    else:
                        lhs = soc_list[t]
                        rhs = (1.0 - sd) * soc_list[t - 1] + eta_c * c_list[t] * dt - inv_eta_d * d_list[t] * dt
                        model.add(lhs == rhs,
                                  name=f"soc_rec_{st.name}_s{s_idx}_t{t}")
                if st.cyclic and T > 0 and cap_e_expr is not None:
                    model.add(soc_list[T - 1] == cap_e_expr * st.soc_initial,
                              name=f"soc_cyc_{st.name}_s{s_idx}")
                stage_charge[s_idx][st.name] = c_list
                stage_discharge[s_idx][st.name] = d_list
                stage_soc[s_idx][st.name] = soc_list

            # Links — transport model with vintage cap + bidirectional option
            for lk in sys._links:
                f_list = []
                fr_list = []
                cap_expr = link_cap_active[lk.name][s_idx]
                loose_cap = lk.max_capacity if lk.extendable and lk.max_capacity != float("inf") else (
                    lk.capacity if not lk.extendable else 1e12)
                for t in range(T):
                    f = model.variable(
                        f"f_{lk.name}_s{s_idx}_t{t}", lower=0.0, upper=max(0.0, loose_cap))
                    f_list.append(f)
                    if cap_expr is not None:
                        model.add(f <= cap_expr,
                                  name=f"fcap_{lk.name}_s{s_idx}_t{t}")
                    else:
                        model.add(f <= 0.0,
                                  name=f"fcap0_{lk.name}_s{s_idx}_t{t}")
                    # Injection at bus_from is -f, withdrawal at bus_to is +η·f
                    bus_expr[lk.bus_from.name][t] = _sub(bus_expr[lk.bus_from.name][t], f)
                    gain = lk.efficiency * (1.0 - lk.loss)
                    bus_expr[lk.bus_to.name][t] = _add(bus_expr[lk.bus_to.name][t], gain * f)
                    if lk.bidirectional:
                        fr = model.variable(
                            f"fr_{lk.name}_s{s_idx}_t{t}", lower=0.0, upper=max(0.0, loose_cap))
                        fr_list.append(fr)
                        if cap_expr is not None:
                            model.add(fr <= cap_expr,
                                      name=f"frcap_{lk.name}_s{s_idx}_t{t}")
                        bus_expr[lk.bus_to.name][t] = _sub(bus_expr[lk.bus_to.name][t], fr)
                        bus_expr[lk.bus_from.name][t] = _add(bus_expr[lk.bus_from.name][t], gain * fr)
                stage_flow[s_idx][lk.name] = f_list

            # Per-bus balance vs load
            for b in sys._buses:
                for t in range(T):
                    expr = bus_expr[b.name][t]
                    load_val = 0.0
                    for ld in sys._loads:
                        if ld.bus.name != b.name:
                            continue
                        if isinstance(ld.amount, (np.ndarray, list, tuple)):
                            load_val += float(ld.amount[t])
                        else:
                            load_val += float(ld.amount)
                    if expr is None:
                        if load_val != 0.0:
                            raise ValueError(
                                f"Stage {s_idx} ({years[s_idx]}) bus={b.name} "
                                f"t={t}: demand {load_val} with no sources.")
                        continue
                    model.add(expr == load_val,
                              name=f"balance_s{s_idx}_b{b.name}_t{t}")

            # Objective contributions
            # Generators — marginal cost + fixed O&M + capex
            for g in sys._generators:
                if g.marginal_cost != 0.0:
                    for t in range(T):
                        obj = _add(obj, g.marginal_cost * stage_p_vars[s_idx][g.name][t] * dt)
                if g.fixed_om != 0.0 and gen_cap_active[g.name][s_idx] is not None:
                    obj = _add(obj, g.fixed_om * gen_cap_active[g.name][s_idx])
                if g.extendable and gen_newbuild[g.name][s_idx] is not None:
                    if g.capital_cost != 0.0:
                        obj = _add(obj, g.capital_cost * gen_newbuild[g.name][s_idx])

            # Storages
            for st in sys._storages:
                if st.marginal_cost != 0.0:
                    for t in range(T):
                        obj = _add(obj, st.marginal_cost * stage_discharge[s_idx][st.name][t] * dt)
                if st.marginal_cost_charge != 0.0:
                    for t in range(T):
                        obj = _add(obj, st.marginal_cost_charge * stage_charge[s_idx][st.name][t] * dt)
                if st.fixed_om_power != 0.0 and stor_cap_p_active[st.name][s_idx] is not None:
                    obj = _add(obj, st.fixed_om_power * stor_cap_p_active[st.name][s_idx])
                if st.fixed_om_energy != 0.0 and stor_cap_e_active[st.name][s_idx] is not None:
                    obj = _add(obj, st.fixed_om_energy * stor_cap_e_active[st.name][s_idx])
                if st.extendable:
                    if stor_new_p[st.name][s_idx] is not None and st.capital_cost_power != 0.0:
                        obj = _add(obj, st.capital_cost_power * stor_new_p[st.name][s_idx])
                    if stor_new_e[st.name][s_idx] is not None and st.capital_cost_energy != 0.0:
                        obj = _add(obj, st.capital_cost_energy * stor_new_e[st.name][s_idx])

            # Links
            for lk in sys._links:
                if lk.marginal_cost != 0.0:
                    for t in range(T):
                        obj = _add(obj, lk.marginal_cost * stage_flow[s_idx][lk.name][t] * dt)
                if lk.fixed_om != 0.0 and link_cap_active[lk.name][s_idx] is not None:
                    obj = _add(obj, lk.fixed_om * link_cap_active[lk.name][s_idx])
                if lk.extendable and link_newbuild[lk.name][s_idx] is not None:
                    if lk.capital_cost != 0.0:
                        obj = _add(obj, lk.capital_cost * link_newbuild[lk.name][s_idx])

        if obj is not None:
            model.minimize(obj)

        t_start = time.perf_counter()
        kwargs = {"verbose": verbose, "presolve": presolve}
        if solver is not None:
            kwargs["solver"] = solver
        raw = model.solve(**kwargs)
        solve_time = time.perf_counter() - t_start

        result = MultiStageResult(
            status=raw.status,
            total_cost=raw.objective if raw.objective is not None else float("nan"),
            solve_time=solve_time,
            years=years,
            _raw=raw,
        )

        if raw.status == "optimal":
            # Generators
            for name in gen_names:
                nb = np.zeros(S)
                ca = np.zeros(S)
                for s_idx in range(S):
                    v = gen_newbuild[name][s_idx]
                    if v is not None:
                        nb[s_idx] = raw.value(v)
                    ca[s_idx] = _eval_vintage(
                        name, s_idx, years, gen_newbuild, gen_brownfield,
                        gen_by_stage, raw)
                result.new_builds[name] = nb
                result.capacity_active[name] = ca

            # Storages
            for name in stor_names:
                nbp = np.zeros(S); nbe = np.zeros(S)
                cap = np.zeros(S); cae = np.zeros(S)
                for s_idx in range(S):
                    vp = stor_new_p[name][s_idx]
                    ve = stor_new_e[name][s_idx]
                    if vp is not None:
                        nbp[s_idx] = raw.value(vp)
                    if ve is not None:
                        nbe[s_idx] = raw.value(ve)
                    cap[s_idx] = _eval_vintage(
                        name, s_idx, years, stor_new_p, stor_brownfield_p,
                        stor_by_stage, raw)
                    cae[s_idx] = _eval_vintage(
                        name, s_idx, years, stor_new_e, stor_brownfield_e,
                        stor_by_stage, raw)
                result.storage_new_power[name] = nbp
                result.storage_new_energy[name] = nbe
                result.storage_power_active[name] = cap
                result.storage_energy_active[name] = cae

            # Links
            for name in link_names:
                nb = np.zeros(S); ca = np.zeros(S)
                for s_idx in range(S):
                    v = link_newbuild[name][s_idx]
                    if v is not None:
                        nb[s_idx] = raw.value(v)
                    ca[s_idx] = _eval_vintage(
                        name, s_idx, years, link_newbuild, link_brownfield,
                        link_by_stage, raw)
                result.link_new_builds[name] = nb
                result.link_capacity_active[name] = ca

            # Dispatch
            for s_idx, sys in enumerate(systems):
                d: dict[str, np.ndarray] = {}
                for g in sys._generators:
                    p_list = stage_p_vars[s_idx][g.name]
                    d[g.name] = np.array([raw.value(v) for v in p_list])
                result.stage_dispatch.append(d)

                f: dict[str, np.ndarray] = {}
                for lk in sys._links:
                    fl = stage_flow[s_idx][lk.name]
                    f[lk.name] = np.array([raw.value(v) for v in fl])
                result.stage_link_flow.append(f)

                s: dict[str, np.ndarray] = {}
                for st in sys._storages:
                    sl = stage_soc[s_idx][st.name]
                    s[st.name] = np.array([raw.value(v) for v in sl])
                result.stage_storage_soc.append(s)

        return result

    # ------------------------------------------------------------------

    def _optimise_myopic(self, *, solver, verbose, presolve) -> MultiStageResult:
        """Solve stages sequentially; freeze each stage's new-builds after solve.

        Each stage sees only itself plus the *already-committed* capacity from
        earlier stages. This matches GenX's ``MyopicHorizon`` mode — faster
        and more realistic (no foresight of future demand / tech-cost paths),
        at the expense of global optimality.
        """
        stages = sorted(self._stages, key=lambda t: t[0])
        S = len(stages)

        fixed: dict = {}
        t_start = time.perf_counter()
        total_cost = 0.0
        stitched: Optional[MultiStageResult] = None

        for horizon_len in range(1, S + 1):
            # Solve with stages [0, horizon_len) active; later ones unmodeled
            # by temporarily stashing the tail. We truncate and restore.
            saved_tail = self._stages[horizon_len:]
            self._stages = self._stages[:horizon_len]
            try:
                r = self._optimise_perfect_foresight(
                    solver=solver, verbose=verbose, presolve=presolve,
                    fixed_newbuilds=dict(fixed))
            finally:
                self._stages = self._stages[:horizon_len] + saved_tail

            if r.status != "optimal":
                # Propagate failure
                return MultiStageResult(
                    status=r.status, total_cost=float("nan"),
                    solve_time=time.perf_counter() - t_start,
                    years=[y for y, _ in stages],
                )

            # Pin the horizon_len-1 stage's new-builds and iterate
            s_frozen = horizon_len - 1
            for name, arr in r.new_builds.items():
                if (name, s_frozen) in [(g.name, s_frozen) for g in stages[s_frozen][1]._generators if g.extendable]:
                    fixed[("gen", name, s_frozen)] = float(arr[s_frozen])
            for name, arr in r.storage_new_power.items():
                if (name, s_frozen) in [(st.name, s_frozen) for st in stages[s_frozen][1]._storages if st.extendable]:
                    fixed[("stor_p", name, s_frozen)] = float(arr[s_frozen])
            for name, arr in r.storage_new_energy.items():
                if (name, s_frozen) in [(st.name, s_frozen) for st in stages[s_frozen][1]._storages if st.extendable]:
                    fixed[("stor_e", name, s_frozen)] = float(arr[s_frozen])
            for name, arr in r.link_new_builds.items():
                if (name, s_frozen) in [(lk.name, s_frozen) for lk in stages[s_frozen][1]._links if lk.extendable]:
                    fixed[("link", name, s_frozen)] = float(arr[s_frozen])

            stitched = r  # final-horizon result is the return value

        assert stitched is not None
        stitched.solve_time = time.perf_counter() - t_start
        return stitched


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_names(systems, accessor) -> list[str]:
    """De-dup component names across stages in first-encounter order."""
    out: list[str] = []
    for sys in systems:
        for obj in accessor(sys):
            if obj.name not in out:
                out.append(obj.name)
    return out


def _build_vintage_expr(names, years, new_vars, brownfield, by_stage):
    """Return {name: [expr per stage]} summing all live vintages + brownfield."""
    S = len(years)
    out = {n: [None] * S for n in names}
    for name in names:
        for s_idx, year in enumerate(years):
            expr = None
            if brownfield[name][s_idx] != 0.0:
                expr = float(brownfield[name][s_idx])
            for prev_idx in range(s_idx + 1):
                v = new_vars[name][prev_idx]
                if v is None:
                    continue
                obj_prev = by_stage.get((name, prev_idx))
                if obj_prev is None:
                    continue
                life = getattr(obj_prev, "lifetime_years", 25)
                lead = getattr(obj_prev, "build_lead_years", 0)
                if _vintage_active(years[prev_idx], year, lead, life):
                    expr = v if expr is None else expr + v
            out[name][s_idx] = expr
    return out


def _retiring_amt(host_name, s_idx, years, new_vars, brownfield, by_stage):
    """Linear expression: ``host_name`` capacity retiring between s-1 and s.

    Retiring = (active at s-1) AND NOT (active at s), summed over brownfield
    and every vintage new-build. Returns ``None`` when nothing retires at
    this boundary (including s=0, where there is no prior stage).
    """
    if s_idx == 0:
        return None
    year_prev = years[s_idx - 1]
    year_curr = years[s_idx]
    expr = None
    # Brownfield: drop from stage s-1 to s counts as retiring.
    bf_prev = float(brownfield[host_name][s_idx - 1])
    bf_curr = float(brownfield[host_name][s_idx])
    drop = bf_prev - bf_curr
    if drop > 0.0:
        expr = drop
    # Vintages whose lifetime expires at this boundary.
    for prev_idx in range(s_idx):
        v = new_vars[host_name][prev_idx]
        if v is None:
            continue
        obj_prev = by_stage.get((host_name, prev_idx))
        if obj_prev is None:
            continue
        life = getattr(obj_prev, "lifetime_years", 25)
        lead = getattr(obj_prev, "build_lead_years", 0)
        active_prev = _vintage_active(years[prev_idx], year_prev, lead, life)
        active_curr = _vintage_active(years[prev_idx], year_curr, lead, life)
        if active_prev and not active_curr:
            expr = v if expr is None else expr + v
    return expr


def _eval_vintage(name, s_idx, years, new_vars, brownfield, by_stage, raw) -> float:
    val = float(brownfield[name][s_idx])
    year = years[s_idx]
    for prev_idx in range(s_idx + 1):
        v = new_vars[name][prev_idx]
        if v is None:
            continue
        obj_prev = by_stage.get((name, prev_idx))
        if obj_prev is None:
            continue
        life = getattr(obj_prev, "lifetime_years", 25)
        lead = getattr(obj_prev, "build_lead_years", 0)
        if _vintage_active(years[prev_idx], year, lead, life):
            val += raw.value(v)
    return val


def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _sub(a, b):
    if a is None:
        return -b
    if b is None:
        return a
    return a - b
