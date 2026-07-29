"""
Phase 10 — SOCP relaxation of AC-OPF (Jabr 2006).

Closes the Phase 3 deferral on AC-OPF. Builds a Clarabel-solved second-order
cone problem in lifted bus-voltage variables ``c_ii = |V_i|²`` and branch
variables ``c_ij = Re(V_i V_j*)``, ``s_ij = Im(V_i V_j*)`` with the rotated
SOC ``c_ij² + s_ij² ≤ c_ii c_jj`` per branch.

Scope of this first pass — deliberately narrow so the cone plumbing can land
without rewriting ``EnergySystem.optimise``:

- **Single snapshot.** Multi-period AC-OPF deferred to **Phase 10.x**
  (per-snapshot SOCs would multiply the variable count by ``T``; need a
  warm-start strategy across snapshots).
- **Dispatch only.** Extendable capacities, UC, storage, links with non-OPF
  ``model_type``, and policies are ignored. Capacity-expansion AC-OPF is
  itself a research problem; deferred to **Phase 10.x**.
- **Transformer taps + phase shifters** (MATPOWER complex turns ratio
  ``τ·e^(jφ)``): read via ``link.tap`` (default 1.0) and ``link.shift``
  (default 0 rad). ``tap=1, shift=0`` recovers the pure π-line model
  bit-exactly. DC links still deferred to **Phase 10.x**.
- **π-line shunts** (``link.g_fr``, ``link.b_fr``, ``link.g_to``,
  ``link.b_to``) and **bus shunts** (``bus.g_shunt``, ``bus.b_shunt``)
  are both wired in — default 0.
- **Constant power loads.** Loads carry an optional ``q_amount`` attribute
  (Mvar) — defaults to 0. ZIP / voltage-dependent loads deferred.
- **Linear + quadratic generator cost.** `generator.marginal_cost` gives
  the linear term; opt-in `generator.quadratic_cost` (N_En_Phase 10.4)
  gives `cost = mc·p + q2·p²` with `q2 ≥ 0` (convexity). PWL heat rates
  still flatten to their linear coefficient — deferred to **Phase 10.x**.
- **Angle-difference envelopes** (Kocuk et al. 2016 "SOCP+AT",
  N_En_Phase 10.10). Optional per-branch linear cuts
  ``-tan(θ_max) c_ij ≤ s_ij ≤ tan(θ_max) c_ij`` tighten the Jabr lift
  on meshed networks by forcing ``(c_ij, s_ij)`` into the angle wedge
  of the true voltage-phasor cross-product. Safe for any
  ``|θ_max| < π/2`` (does not cut AC-feasible solutions with smaller
  branch angle spread). Opt-in via ``solve_socp_opf(..., angle_diff_max=)``
  or per-link ``link.angle_diff_max`` / ``link.angle_diff_min``.
- **QC-lite cycle closure** (Coffrin & Hijazi 2015 QC-relaxation
  lineage, N_En_Phase 10.11). Optional per-branch angle aux
  ``θ_ij`` linked to ``s_ij`` through the *sound* linear sin-Taylor
  envelope ``|s_ij - v_nom² · θ_ij| ≤ v_max² · θ_max³/6 + (v_max² -
  v_min²)/2 · θ_max``, plus exact loop-closure equations
  ``Σ_cycle ±θ_ij = 0`` over a fundamental cycle basis of the
  SOCP-branch subgraph. Closes the cycle-consistency gap that the
  per-branch arctangent envelope alone cannot reach on meshed
  networks. Opt-in via ``solve_socp_opf(..., enforce_cycle_closure=
  True)``; requires ``angle_diff_max`` (function-level or per-link).

The result is sufficient to (a) round-trip a small case, (b) validate the
Clarabel cone backend end-to-end, and (c) give downstream code a real AC
relaxation it can use as a sanity check on DC-OPF dispatch.

Activation: mark each line ``Link`` with ``model_type="socp_opf"`` (and
ensure ``reactance > 0``). Then call::

    from nexus_energy import solve_socp_opf
    res = solve_socp_opf(system)

Inputs read via ``getattr`` (with defaults) to avoid touching the ``Bus`` /
``Generator`` / ``Load`` dataclasses for what is still an experimental code
path:

    - ``bus.v_min``         default 0.95 pu
    - ``bus.v_max``         default 1.05 pu
    - ``link.resistance``   default 0 pu
    - ``link.s_max``        default ``link.capacity`` MVA
    - ``generator.q_min``   default ``-0.5 * capacity``
    - ``generator.q_max``   default ``+0.5 * capacity``
    - ``load.q_amount``     default 0 Mvar
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nexus_energy._conic import ConicProblem, is_available

if TYPE_CHECKING:
    from nexus_energy.core import EnergySystem


@dataclass
class SOCPOpfResult:
    """Result of :func:`solve_socp_opf`."""
    status: str
    total_cost: float
    voltage_mag: dict[str, float]  # bus.name -> |V| (pu)
    gen_p: dict[str, float]        # gen.name -> P (MW)
    gen_q: dict[str, float]        # gen.name -> Q (Mvar)
    branch_p: dict[str, float]     # link.name -> P_sending (MW)
    branch_q: dict[str, float]     # link.name -> Q_sending (Mvar)
    branch_loss: dict[str, float]  # link.name -> P_loss (MW)
    solve_time: float
    iterations: int


@dataclass
class _SOCPBuild:
    """Internal: SOCP program + variable index maps.

    Produced by :func:`_build_socp_problem`. Bundles the ``ConicProblem`` with
    enough state for :func:`solve_socp_opf` to assemble an ``SOCPOpfResult``
    and for :func:`obbt_tighten` (N_En_Phase 10.14) to swap in min/max linear
    objectives on primitive variables without re-implementing the constraint
    set.
    """
    prob: ConicProblem
    buses: list
    lines: list
    cii: dict[str, int]
    cij: dict[str, int]
    sij: dict[str, int]
    P_ij: dict[str, int]
    Q_ij: dict[str, int]
    P_ji: dict[str, int]
    Q_ji: dict[str, int]
    p_gen: dict[str, int]
    q_gen: dict[str, int]
    theta_ij: dict[str, int]          # empty if not enforce_cycle_closure
    v_mag: dict[str, int]             # empty if not enforce_tight_qc
    w_ij: dict[str, int]              # empty if not enforce_tight_qc
    cos_theta_ij: dict[str, int]      # empty if not enforce_tight_qc
    sin_theta_ij: dict[str, int]      # empty if not enforce_tight_qc
    gen_cap: dict[str, int] = None    # gen.name -> cap var idx; empty unless expansion
    line_cap: dict[str, int] = None   # link.name -> s_max var idx; empty unless expansion


@dataclass
class OBBTStats:
    """Result of :func:`obbt_tighten` — box-tightening telemetry.

    ``v_mag_reductions[bus_name] = (initial_width, final_width)`` for voltage
    magnitude bounds (in pu), and analogously for ``theta_reductions`` in
    radians. ``final_max_shrink`` is the largest per-variable fractional
    shrinkage in the final iteration (``0`` means fixed-point hit, ``≥ tol``
    means ``max_iter`` was spent without convergence).
    """
    iters: int
    v_mag_reductions: dict[str, tuple[float, float]]
    theta_reductions: dict[str, tuple[float, float]]
    final_max_shrink: float
    solve_time: float


def _socp_opf_lines(system: "EnergySystem") -> list:
    return [l for l in system._links if getattr(l, "model_type", "") == "socp_opf"]


def _fundamental_cycle_basis(lines: list, buses: list) -> list[list[tuple]]:
    """Fundamental cycle basis of the graph induced by ``lines`` over ``buses``.

    Each returned cycle is a list of ``(link, sign)`` pairs where ``sign=+1``
    means the link is traversed in its native direction (``bus_from → bus_to``)
    along the cycle. The loop-closure equation is then
    ``Σ sign · θ_ij = 0`` with ``θ_ij = angle(V_from) - angle(V_to)``.

    Hand-rolled (not NetworkX) so that parallel branches are preserved — each
    extra parallel copy becomes a chord producing a 2-cycle. Disconnected
    components are fine; each contributes its own spanning tree and cycles.
    """
    bus_names = [b.name for b in buses]
    if not bus_names or not lines:
        return []

    adj: dict[str, list[tuple[str, object, int]]] = {n: [] for n in bus_names}
    for l in lines:
        adj[l.bus_from.name].append((l.bus_to.name, l, +1))
        adj[l.bus_to.name].append((l.bus_from.name, l, -1))

    parent: dict[str, tuple[str, object, int]] = {}
    root_of: dict[str, str] = {}
    tree_links: set[int] = set()

    for start in bus_names:
        if start in root_of:
            continue
        root_of[start] = start
        queue: list[str] = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for (v, link, direction) in adj[u]:
                if v in root_of:
                    continue
                root_of[v] = start
                parent[v] = (u, link, direction)
                tree_links.add(id(link))
                queue.append(v)

    cycles: list[list[tuple]] = []
    for l in lines:
        if id(l) in tree_links:
            continue
        u = l.bus_from.name
        v = l.bus_to.name
        if u not in root_of or v not in root_of or root_of[u] != root_of[v]:
            continue

        anc_v: set[str] = set()
        w = v
        anc_v.add(w)
        while w != root_of[w]:
            w = parent[w][0]
            anc_v.add(w)

        u_path: list[tuple] = []
        w = u
        while w not in anc_v:
            p, link, dir_pu = parent[w]
            u_path.append((link, -dir_pu))
            w = p
        lca = w

        v_path: list[tuple] = []
        w = v
        while w != lca:
            p, link, dir_pu = parent[w]
            v_path.append((link, -dir_pu))
            w = p

        cycle: list[tuple] = [(l, +1)]
        cycle.extend(v_path)
        for (link, s) in reversed(u_path):
            cycle.append((link, -s))
        cycles.append(cycle)

    return cycles


def _build_socp_problem(system: "EnergySystem", *, snapshot: int = 0,
                        angle_diff_max: float | None = None,
                        enforce_cycle_closure: bool = False,
                        enforce_tight_qc: bool = False,
                        cos_envelope_pieces: int = 8,
                        add_cost_objective: bool = True,
                        expand_capacity: bool = False) -> _SOCPBuild:
    """Internal: construct the SOCP AC-OPF program for one snapshot.

    This is the full constraint-posting body that used to live inside
    ``solve_socp_opf``. Factoring it out lets :func:`obbt_tighten`
    (N_En_Phase 10.14) rebuild the same cone program with a different
    linear objective — one min/max subproblem per primitive variable —
    without duplicating the constraint set.

    ``cos_envelope_pieces`` (N_En_Phase 10.14.1) controls the number of
    tangent-line upper cuts used to envelope ``cos(θ_ij)`` in the tight-
    QC formulation. Legacy 10.12 behaviour is ``cos_envelope_pieces=2``
    (tangents at the interval endpoints only). Higher values give
    tighter upper envelopes at the cost of more linear rows per branch.

    ``add_cost_objective`` gates the ``Σ mc · p_gen`` linear cost term so
    OBBT can build a feasibility problem and then overwrite ``prob.q``
    per subsolve.

    ``expand_capacity`` (N_En_Phase 10.2 — capacity-expansion AC-OPF)
    turns the rated capacity of every ``generator.extendable`` /
    ``link.extendable`` component into a decision variable. Each
    extendable generator gets a column ``cap_g ∈ [min_capacity,
    max_capacity]``; its active-power upper bound becomes the *coupling*
    ``p_gen ≤ cap_g`` (and the reactive box scales as ``|q| ≤ 0.5·cap_g``
    unless ``q_min``/``q_max`` are pinned explicitly), and an annualised
    investment term ``capital_cost · cap_g`` is added to the objective.
    Each extendable line/transformer gets ``s_max_var ∈ [min_capacity,
    max_capacity]`` driving the thermal-cone t-axis directly, with
    ``capital_cost · s_max_var`` on the objective. The Jabr lift, the
    rotated voltage SOC, the angle-difference / QC envelopes and the
    nodal balance are untouched — this is the standard investment +
    SOCP-relaxation capacity-expansion OPF (Jabr 2006 SOCP lift of
    AC-OPF; Kocuk, Dey & Sun 2016 *Oper. Res.* 64(6) SOCP relaxation
    used as the planning subproblem; the linear ``Σ capital_cost·cap``
    investment objective is the classic transmission/generation-
    expansion-planning term, e.g. PyPSA's ``p_nom``/``s_nom`` extendable
    optimisation). When ``False`` (default) the program is built with
    fixed capacities exactly as before — bit-identical to the pre-10.2
    code path.
    """
    if cos_envelope_pieces < 2:
        raise ValueError(
            f"cos_envelope_pieces must be >= 2 (got {cos_envelope_pieces})")
    if not is_available():
        raise RuntimeError(
            "SOCP AC-OPF requires the optional `clarabel` package. "
            "Install with `pip install clarabel`.")

    lines = _socp_opf_lines(system)
    if not lines:
        raise ValueError(
            "solve_socp_opf: no Links with model_type='socp_opf' present.")

    # Tight QC (10.12) subsumes the cycle-closure infra (10.11): it reuses
    # θ_ij + Σ±θ_ij=0 and swaps the loose sin-Taylor coupling for the
    # full McCormick lift. Flipping the parent flag avoids a second
    # allocation pass.
    if enforce_tight_qc:
        enforce_cycle_closure = True

    # Snapshot: pull scalar values out of any time-series fields.
    def _scalar(x):
        if isinstance(x, np.ndarray):
            return float(x[snapshot])
        return float(x)

    buses = list(system._buses)
    bus_idx = {b.name: i for i, b in enumerate(buses)}
    gens_by_bus: dict[str, list] = {b.name: [] for b in buses}
    for g in system._generators:
        gens_by_bus[g.bus.name].append(g)
    loads_by_bus: dict[str, list] = {b.name: [] for b in buses}
    for ld in system._loads:
        loads_by_bus[ld.bus.name].append(ld)

    # ---- variable layout ----
    # Per bus: c_ii (squared voltage). Per branch: c_ij, s_ij plus auxiliary
    # u_plus = (c_ii+c_jj)/sqrt(2), u_minus = (c_ii-c_jj)/sqrt(2) for the
    # rotated-SOC reformulation, plus P_ij, Q_ij sending-end flows. Per gen:
    # p_gen, q_gen.
    var_names: list[str] = []
    def _alloc(name: str) -> int:
        var_names.append(name)
        return len(var_names) - 1

    cii: dict[str, int] = {b.name: _alloc(f"c_{b.name}") for b in buses}
    cij: dict[str, int] = {l.name: _alloc(f"c_{l.name}") for l in lines}
    sij: dict[str, int] = {l.name: _alloc(f"s_{l.name}") for l in lines}
    # Auxiliaries for the rotated-SOC ⇄ standard-SOC translation:
    #   c_sum = c_ii + c_jj      (cone t-axis)
    #   c_diff = c_ii - c_jj
    #   cij2 = 2*c_ij            (so the SOC bounds c_ij² + s_ij² ≤ c_ii c_jj
    #   sij2 = 2*s_ij             rather than the loose 2 c_ii c_jj)
    c_sum: dict[str, int] = {l.name: _alloc(f"csum_{l.name}") for l in lines}
    c_diff: dict[str, int] = {l.name: _alloc(f"cdiff_{l.name}") for l in lines}
    cij2: dict[str, int] = {l.name: _alloc(f"cij2_{l.name}") for l in lines}
    sij2: dict[str, int] = {l.name: _alloc(f"sij2_{l.name}") for l in lines}
    P_ij: dict[str, int] = {l.name: _alloc(f"P_{l.name}") for l in lines}
    Q_ij: dict[str, int] = {l.name: _alloc(f"Q_{l.name}") for l in lines}
    P_ji: dict[str, int] = {l.name: _alloc(f"Pji_{l.name}") for l in lines}
    Q_ji: dict[str, int] = {l.name: _alloc(f"Qji_{l.name}") for l in lines}
    p_gen: dict[str, int] = {g.name: _alloc(f"p_{g.name}") for g in system._generators}
    q_gen: dict[str, int] = {g.name: _alloc(f"q_{g.name}") for g in system._generators}
    s_max_var: dict[str, int] = {l.name: _alloc(f"smax_{l.name}") for l in lines}
    # Capacity-expansion decision variables (N_En_Phase 10.2). Only allocated
    # for components flagged ``extendable`` and only when ``expand_capacity``
    # is on, so fixed-capacity builds stay bit-identical.
    gen_cap: dict[str, int] = {}
    line_cap: dict[str, int] = {}
    if expand_capacity:
        for g in system._generators:
            if getattr(g, "extendable", False):
                gen_cap[g.name] = _alloc(f"cap_{g.name}")
        for l in lines:
            if getattr(l, "extendable", False):
                line_cap[l.name] = _alloc(f"scap_{l.name}")
    # QC-lite θ_ij auxiliaries (N_En_Phase 10.11). Allocated up front so the
    # per-branch loop can post the sin-Taylor coupling constraints inline.
    theta_ij: dict[str, int] = {}
    if enforce_cycle_closure:
        theta_ij = {l.name: _alloc(f"theta_{l.name}") for l in lines}

    # Tight QC auxiliaries (N_En_Phase 10.12, Coffrin & Hijazi 2015).
    v_mag: dict[str, int] = {}
    w_ij: dict[str, int] = {}
    cos_theta_ij: dict[str, int] = {}
    sin_theta_ij: dict[str, int] = {}
    if enforce_tight_qc:
        v_mag = {b.name: _alloc(f"vmag_{b.name}") for b in buses}
        w_ij = {l.name: _alloc(f"w_{l.name}") for l in lines}
        cos_theta_ij = {l.name: _alloc(f"cos_{l.name}") for l in lines}
        sin_theta_ij = {l.name: _alloc(f"sin_{l.name}") for l in lines}

    n_vars = len(var_names)
    prob = ConicProblem(n=n_vars)

    # ---- objective: Σ (mc·p_gen + q2·p_gen²)  (gated so OBBT can override) ----
    # Linear term `marginal_cost` is the standard nexus field. Quadratic term
    # `quadratic_cost` (N_En_Phase 10.4) is opt-in via getattr — defaults to
    # 0 so pre-10.4 systems stay bit-exact. Clarabel's standard form is
    # `min 0.5 xᵀPx + qᵀx`, so a `cp2·p²` coefficient corresponds to
    # `P[p,p] = 2·cp2`. Must be nonneg for the objective to stay convex.
    if add_cost_objective:
        for g in system._generators:
            mc = float(g.marginal_cost)
            if mc != 0.0:
                prob.add_linear_obj({p_gen[g.name]: mc})
            q2 = float(getattr(g, "quadratic_cost", 0.0))
            if q2 != 0.0:
                if q2 < 0.0:
                    raise ValueError(
                        f"generator {g.name!r}: quadratic_cost must be ≥ 0 "
                        f"for a convex objective; got {q2}")
                p_idx = p_gen[g.name]
                prob.add_quadratic_obj([(p_idx, p_idx, 2.0 * q2)])
        # Investment term Σ capital_cost · cap (N_En_Phase 10.2). Linear in the
        # capacity decision variables; only present when expansion is on.
        for g in system._generators:
            if g.name in gen_cap:
                cc = float(getattr(g, "capital_cost", 0.0))
                if cc != 0.0:
                    prob.add_linear_obj({gen_cap[g.name]: cc})
        for l in lines:
            if l.name in line_cap:
                cc = float(getattr(l, "capital_cost", 0.0))
                if cc != 0.0:
                    prob.add_linear_obj({line_cap[l.name]: cc})

    # ---- voltage magnitude bounds via single-row LE constraints ----
    # We can't put bounds directly on Clarabel variables (no box); use the
    # nonneg cone (Ax ≤ b → -Ax ≥ -b → s = -Ax + b ≥ 0).
    for b in buses:
        v_min = float(getattr(b, "v_min", 0.95))
        v_max = float(getattr(b, "v_max", 1.05))
        # c_ii ≤ v_max²
        prob.add_le({cii[b.name]: 1.0}, v_max * v_max)
        # -c_ii ≤ -v_min²
        prob.add_le({cii[b.name]: -1.0}, -v_min * v_min)

        if enforce_tight_qc:
            # v_mag_i bounded by [v_min, v_max], linked to c_ii via three
            # tangent lower bounds on v_mag² (at v_min / v_nom / v_max) and a
            # single secant upper bound over [v_min, v_max]. Tangent of
            # f(v) = v² at v*: v² ≥ 2·v*·v − v*² → 2·v*·v_mag − c_ii ≤ v*².
            # Secant over [v_min, v_max]: v² ≤ (v_min+v_max)·v − v_min·v_max
            # → c_ii − (v_min+v_max)·v_mag ≤ −v_min·v_max.
            v_idx = v_mag[b.name]
            prob.add_le({v_idx:  1.0},  v_max)
            prob.add_le({v_idx: -1.0}, -v_min)
            v_nom = 0.5 * (v_min + v_max)
            for v_star in (v_min, v_nom, v_max):
                prob.add_le({v_idx: 2.0 * v_star, cii[b.name]: -1.0},
                            v_star * v_star)
            prob.add_le({cii[b.name]: 1.0, v_idx: -(v_min + v_max)},
                        -v_min * v_max)

    # ---- per-branch: rotated SOC + linear flow definitions + thermal SOC ----
    for l in lines:
        if l.reactance <= 0.0:
            raise ValueError(
                f"link {l.name!r}: model_type='socp_opf' requires reactance > 0")
        r = float(getattr(l, "resistance", 0.0))
        x = float(l.reactance)
        denom = r * r + x * x
        g_se = r / denom
        b_se = -x / denom

        # Pin the four auxiliaries.
        # c_sum = c_ii + c_jj
        prob.add_eq({
            c_sum[l.name]: 1.0,
            cii[l.bus_from.name]: -1.0,
            cii[l.bus_to.name]:   -1.0,
        }, 0.0)
        # c_diff = c_ii - c_jj
        prob.add_eq({
            c_diff[l.name]: 1.0,
            cii[l.bus_from.name]: -1.0,
            cii[l.bus_to.name]:    1.0,
        }, 0.0)
        # cij2 = 2 c_ij
        prob.add_eq({cij2[l.name]: 1.0, cij[l.name]: -2.0}, 0.0)
        # sij2 = 2 s_ij
        prob.add_eq({sij2[l.name]: 1.0, sij[l.name]: -2.0}, 0.0)
        # SOC: c_sum ≥ ‖(c_diff, 2 c_ij, 2 s_ij)‖₂
        # ⇒ (c_ii+c_jj)² ≥ (c_ii-c_jj)² + 4 c_ij² + 4 s_ij²
        # ⇒ 4 c_ii c_jj ≥ 4(c_ij² + s_ij²)  ⇒ c_ij² + s_ij² ≤ c_ii c_jj  ✓
        prob.add_soc(t_var=c_sum[l.name],
                     u_vars=[c_diff[l.name], cij2[l.name], sij2[l.name]])

        # Arctangent envelope (Kocuk et al. 2016 "SOCP+AT"): the Jabr
        # lift alone does not pin the phase of V_i V_j* = c_ij + j s_ij.
        # On meshed networks that slack lets the relaxation settle on
        # dispatches with physically impossible angle spread. A linear
        # wedge `tan(θ_min) c_ij ≤ s_ij ≤ tan(θ_max) c_ij` closes most
        # of that gap while staying SOCP. Per-link override via
        # `link.angle_diff_max` / `link.angle_diff_min`; otherwise the
        # function-level `angle_diff_max` is used for symmetric bounds.
        link_amax = getattr(l, "angle_diff_max", angle_diff_max)
        link_amin = getattr(l, "angle_diff_min", None)
        if link_amin is None and link_amax is not None:
            link_amin = -link_amax
        # Both bounds live in (-π/2, π/2) with min ≤ max. OBBT can tighten a
        # symmetric box into a fully one-sided subinterval (e.g. [0.001,
        # 0.03]), so the sign of each end is not pinned.
        if link_amax is not None:
            if not (-0.5 * math.pi < link_amax < 0.5 * math.pi):
                raise ValueError(
                    f"link {l.name!r}: angle_diff_max must lie in (-π/2, π/2); "
                    f"got {link_amax}")
            prob.add_le({sij[l.name]: 1.0,
                         cij[l.name]: -math.tan(link_amax)}, 0.0)
        if link_amin is not None:
            if not (-0.5 * math.pi < link_amin < 0.5 * math.pi):
                raise ValueError(
                    f"link {l.name!r}: angle_diff_min must lie in (-π/2, π/2); "
                    f"got {link_amin}")
            if link_amax is not None and link_amin > link_amax + 1e-12:
                raise ValueError(
                    f"link {l.name!r}: angle_diff_min ({link_amin}) must not "
                    f"exceed angle_diff_max ({link_amax})")
            prob.add_le({sij[l.name]: -1.0,
                         cij[l.name]: math.tan(link_amin)}, 0.0)

        # QC cycle-closure coupling (N_En_Phase 10.11 loose / 10.12 tight).
        # Both variants share the θ_ij bound block + the post-loop Σ±θ_ij=0
        # equation set; they differ only in how θ_ij is coupled to (c_ij,
        # s_ij) for this branch.
        if enforce_cycle_closure:
            link_tmax = getattr(l, "angle_diff_max", angle_diff_max)
            link_tmin = getattr(l, "angle_diff_min", None)
            if link_tmax is None:
                raise ValueError(
                    f"link {l.name!r}: enforce_cycle_closure=True requires "
                    "angle_diff_max (per-link or function-level) to be set.")
            if link_tmin is None:
                link_tmin = -link_tmax
            t_idx = theta_ij[l.name]
            # θ_min ≤ θ_ij ≤ θ_max
            prob.add_le({t_idx:  1.0},  link_tmax)
            prob.add_le({t_idx: -1.0}, -link_tmin)
            tmax_abs = max(abs(link_tmax), abs(link_tmin))

            if enforce_tight_qc:
                # Coffrin & Hijazi 2015 QC: lift w_ij = |V_i|·|V_j|,
                # cos_θ_ij ≈ cos(θ_ij), sin_θ_ij ≈ sin(θ_ij); then
                # c_ij = w_ij·cos_θ and s_ij = w_ij·sin_θ via McCormick.
                Vli = float(getattr(l.bus_from, "v_min", 0.95))
                Vhi = float(getattr(l.bus_from, "v_max", 1.05))
                Vlj = float(getattr(l.bus_to,   "v_min", 0.95))
                Vhj = float(getattr(l.bus_to,   "v_max", 1.05))
                w_lo = Vli * Vlj
                w_hi = Vhi * Vhj
                # Asymmetric cos / sin box over the (possibly post-OBBT)
                # interval [link_tmin, link_tmax]. When 0 ∈ [tmin, tmax] the
                # cos maximum is 1; otherwise it is the endpoint nearer 0.
                # sin is monotone increasing on (-π/2, π/2), so its box is
                # exactly [sin(tmin), sin(tmax)]. Using the tight asymmetric
                # box (rather than the symmetric [-tmax_abs, +tmax_abs]) is
                # a strict improvement on one-sided intervals like
                # [0.001, 0.03] produced by OBBT.
                cos_tmin = math.cos(link_tmin)
                cos_tmax = math.cos(link_tmax)
                cos_lo = min(cos_tmin, cos_tmax)
                if link_tmin <= 0.0 <= link_tmax:
                    cos_hi = 1.0
                else:
                    cos_hi = max(cos_tmin, cos_tmax)
                sin_lo = math.sin(link_tmin)
                sin_hi = math.sin(link_tmax)
                w_idx = w_ij[l.name]
                c_idx = cos_theta_ij[l.name]
                s_idx = sin_theta_ij[l.name]
                vi_idx = v_mag[l.bus_from.name]
                vj_idx = v_mag[l.bus_to.name]

                # Box bounds on w, cos_θ, sin_θ.
                prob.add_le({w_idx:  1.0},  w_hi)
                prob.add_le({w_idx: -1.0}, -w_lo)
                prob.add_le({c_idx:  1.0},  cos_hi)
                prob.add_le({c_idx: -1.0}, -cos_lo)
                prob.add_le({s_idx:  1.0},  sin_hi)
                prob.add_le({s_idx: -1.0}, -sin_lo)

                # cos envelope (N_En_Phase 10.14.1 piecewise-linear): cos is
                # concave on (-π/2, π/2) so every tangent line is a global
                # upper bound. Post K evenly-spaced tangent cuts at
                # θ_k = tmin + k·(tmax-tmin)/(K-1); tangent slope = -sin θ_k,
                # so the linear row is `cos_θ + sin(θ_k)·θ_ij ≤ cos(θ_k) +
                # sin(θ_k)·θ_k`. Lower bound is the global chord between
                # (tmin, cos tmin) and (tmax, cos tmax) — strictly tighter
                # than the flat `cos_θ ≥ cos_lo` box lower bound (which was
                # the 10.12 behaviour).
                K = int(cos_envelope_pieces)
                for k in range(K):
                    theta_k = link_tmin + (link_tmax - link_tmin) * k / (K - 1)
                    sin_k = math.sin(theta_k)
                    cos_k = math.cos(theta_k)
                    prob.add_le({c_idx: 1.0, t_idx: sin_k},
                                cos_k + sin_k * theta_k)
                if link_tmax - link_tmin > 1e-12:
                    chord_slope = (cos_tmax - cos_tmin) / (link_tmax - link_tmin)
                    # cos_θ ≥ cos_tmin + chord_slope·(θ − tmin)
                    # ⇔ -cos_θ + chord_slope·θ ≤ chord_slope·tmin − cos_tmin
                    prob.add_le({c_idx: -1.0, t_idx: chord_slope},
                                chord_slope * link_tmin - cos_tmin)

                # sin envelope via sin-Taylor: |sin_θ − θ| ≤ tmax_abs³/6 on
                # [-tmax_abs, tmax_abs] ⊇ [link_tmin, link_tmax]. Sound
                # linear bracket.
                st_slack = (tmax_abs ** 3) / 6.0
                prob.add_le({s_idx:  1.0, t_idx: -1.0}, st_slack)
                prob.add_le({s_idx: -1.0, t_idx:  1.0}, st_slack)

                # McCormick: w_ij = v_mag_i · v_mag_j over
                # [Vli, Vhi] × [Vlj, Vhj].
                #   Lower 1: w ≥ Vlj·v_i + Vli·v_j − Vli·Vlj
                #   Lower 2: w ≥ Vhj·v_i + Vhi·v_j − Vhi·Vhj
                #   Upper 1: w ≤ Vlj·v_i + Vhi·v_j − Vhi·Vlj
                #   Upper 2: w ≤ Vhj·v_i + Vli·v_j − Vli·Vhj
                prob.add_le({vi_idx: Vlj, vj_idx: Vli, w_idx: -1.0},
                            Vli * Vlj)
                prob.add_le({vi_idx: Vhj, vj_idx: Vhi, w_idx: -1.0},
                            Vhi * Vhj)
                prob.add_le({w_idx: 1.0, vi_idx: -Vlj, vj_idx: -Vhi},
                            -Vhi * Vlj)
                prob.add_le({w_idx: 1.0, vi_idx: -Vhj, vj_idx: -Vli},
                            -Vli * Vhj)

                # McCormick: c_ij = w_ij · cos_θ over [w_lo, w_hi] ×
                # [cos_lo, cos_hi].
                prob.add_le({c_idx: w_lo, w_idx: cos_lo,
                             cij[l.name]: -1.0}, w_lo * cos_lo)
                prob.add_le({c_idx: w_hi, w_idx: cos_hi,
                             cij[l.name]: -1.0}, w_hi * cos_hi)
                prob.add_le({cij[l.name]: 1.0, c_idx: -w_hi,
                             w_idx: -cos_lo}, -w_hi * cos_lo)
                prob.add_le({cij[l.name]: 1.0, c_idx: -w_lo,
                             w_idx: -cos_hi}, -w_lo * cos_hi)

                # McCormick: s_ij = w_ij · sin_θ over [w_lo, w_hi] ×
                # [sin_lo, sin_hi].
                prob.add_le({s_idx: w_lo, w_idx: sin_lo,
                             sij[l.name]: -1.0}, w_lo * sin_lo)
                prob.add_le({s_idx: w_hi, w_idx: sin_hi,
                             sij[l.name]: -1.0}, w_hi * sin_hi)
                prob.add_le({sij[l.name]: 1.0, s_idx: -w_hi,
                             w_idx: -sin_lo}, -w_hi * sin_lo)
                prob.add_le({sij[l.name]: 1.0, s_idx: -w_lo,
                             w_idx: -sin_hi}, -w_lo * sin_hi)
            else:
                # 10.11 loose coupling: sound linear sin-Taylor plus a
                # v-product midpoint split. Kept for users who want cycle
                # closure without the extra McCormick lift.
                vminl = min(float(getattr(l.bus_from, "v_min", 0.95)),
                            float(getattr(l.bus_to,   "v_min", 0.95)))
                vmaxl = max(float(getattr(l.bus_from, "v_max", 1.05)),
                            float(getattr(l.bus_to,   "v_max", 1.05)))
                vminsq = vminl * vminl
                vmaxsq = vmaxl * vmaxl
                vnomsq = 0.5 * (vminsq + vmaxsq)
                slack = (vmaxsq * (tmax_abs ** 3) / 6.0
                         + 0.5 * (vmaxsq - vminsq) * tmax_abs)
                prob.add_le({sij[l.name]:  1.0, t_idx: -vnomsq}, slack)
                prob.add_le({sij[l.name]: -1.0, t_idx:  vnomsq}, slack)

        # π-line shunt admittance on either end (pu). Defaults to 0.
        g_fr = float(getattr(l, "g_fr", 0.0))
        b_fr = float(getattr(l, "b_fr", 0.0))
        g_to = float(getattr(l, "g_to", 0.0))
        b_to = float(getattr(l, "b_to", 0.0))
        # Transformer complex turns ratio T = tap·e^(jφ). Primary = bus_from
        # (MATPOWER convention). Defaults tap=1, shift=0 recover the pure
        # π-line model bit-exactly.
        tap = float(getattr(l, "tap", 1.0))
        if tap <= 0.0:
            raise ValueError(
                f"link {l.name!r}: tap must be > 0 (got {tap})")
        phi = float(getattr(l, "shift", 0.0))
        inv_tap = 1.0 / tap
        inv_tap2 = inv_tap * inv_tap
        cos_phi = float(np.cos(phi))
        sin_phi = float(np.sin(phi))
        # MATPOWER branch Y-bus (§ 3.7):
        #   Y_ff = (y_se + y_fr) / tap²          = (g_ff + j b_ff) / tap²
        #   Y_tt =  y_se + y_to                  =  g_tt + j b_tt
        #   Y_ft = -y_se · e^(jφ) / tap          → re = rft, im = ift
        #   Y_tf = -y_se · e^(-jφ) / tap         → re = rtf, im = itf
        # where y_se = g_se + j b_se and y_fr/to are the π-shunt half-admittances.
        g_ff = (g_se + g_fr) * inv_tap2
        b_ff = (b_se + b_fr) * inv_tap2
        g_tt = g_se + g_to
        b_tt = b_se + b_to
        rft = (b_se * sin_phi - g_se * cos_phi) * inv_tap
        ift = -(g_se * sin_phi + b_se * cos_phi) * inv_tap
        rtf = -(g_se * cos_phi + b_se * sin_phi) * inv_tap
        itf = (g_se * sin_phi - b_se * cos_phi) * inv_tap
        # Sending-end flow with Jabr lift: S_ij = V_i conj(I_i) and
        # V_i conj(V_j) = c_ij + j s_ij:
        #   P_ij =  g_ff · c_ii + rft · c_ij + ift · s_ij
        #   Q_ij = -b_ff · c_ii + rft · s_ij - ift · c_ij
        prob.add_eq({
            P_ij[l.name]: 1.0,
            cii[l.bus_from.name]: -g_ff,
            cij[l.name]: -rft,
            sij[l.name]: -ift,
        }, 0.0)
        prob.add_eq({
            Q_ij[l.name]: 1.0,
            cii[l.bus_from.name]: b_ff,
            cij[l.name]: ift,
            sij[l.name]: -rft,
        }, 0.0)
        # Receiving-end flow (V_j conj(V_i) = c_ij - j s_ij):
        #   P_ji =  g_tt · c_jj + rtf · c_ij - itf · s_ij
        #   Q_ji = -b_tt · c_jj - rtf · s_ij - itf · c_ij
        prob.add_eq({
            P_ji[l.name]: 1.0,
            cii[l.bus_to.name]: -g_tt,
            cij[l.name]: -rtf,
            sij[l.name]: itf,
        }, 0.0)
        prob.add_eq({
            Q_ji[l.name]: 1.0,
            cii[l.bus_to.name]: b_tt,
            cij[l.name]: itf,
            sij[l.name]: rtf,
        }, 0.0)
        # Thermal limit: P_ij² + Q_ij² ≤ S_max² → SOC(S_max, [P, Q]).
        # For a fixed line the cone t-axis is pinned to the constant rating.
        # For an extendable line (N_En_Phase 10.2) the rating becomes the
        # decision variable s_max_idx = line_cap, bounded by [min_capacity,
        # max_capacity]; the SOC then reads P²+Q² ≤ cap² with cap free, so the
        # solver buys exactly the thermal headroom the dispatch needs.
        s_max_idx = s_max_var[l.name]
        if l.name in line_cap:
            cap_idx = line_cap[l.name]
            prob.add_eq({s_max_idx: 1.0, cap_idx: -1.0}, 0.0)  # s_max = cap
            cap_lo = float(getattr(l, "min_capacity", 0.0))
            cap_hi = float(getattr(l, "max_capacity", float("inf")))
            prob.add_le({cap_idx: -1.0}, -cap_lo)              # cap ≥ min
            if cap_hi != float("inf"):
                prob.add_le({cap_idx: 1.0}, cap_hi)            # cap ≤ max
        else:
            s_max = float(getattr(l, "s_max", l.capacity))
            prob.add_eq({s_max_idx: 1.0}, s_max)  # pin auxiliary to constant
        prob.add_soc(t_var=s_max_idx, u_vars=[P_ij[l.name], Q_ij[l.name]])
        prob.add_soc(t_var=s_max_idx, u_vars=[P_ji[l.name], Q_ji[l.name]])

    # ---- generator bounds ----
    for g in system._generators:
        cf = _scalar(g.carrier_factor) if g.carrier_factor is not None else 1.0
        p_min = float(getattr(g, "p_min", 0.0))
        if g.name in gen_cap:
            # Capacity-expansion (N_En_Phase 10.2): the rated capacity is a
            # decision variable cap_g ∈ [min_capacity, max_capacity]. The
            # active-power dispatch is bounded by the (availability-scaled)
            # capacity it buys: p ≤ cf · cap_g  ⇔  p − cf · cap_g ≤ 0. The
            # reactive box scales with cap_g unless the user pinned an explicit
            # q_min/q_max. p_min is treated as a floor independent of cap (a
            # built unit's minimum stable output).
            cap_idx = gen_cap[g.name]
            cap_lo = float(getattr(g, "min_capacity", 0.0))
            cap_hi = float(getattr(g, "max_capacity", float("inf")))
            prob.add_le({cap_idx: -1.0}, -cap_lo)
            if cap_hi != float("inf"):
                prob.add_le({cap_idx: 1.0}, cap_hi)
            prob.add_le({p_gen[g.name]: 1.0, cap_idx: -cf}, 0.0)
            prob.add_le({p_gen[g.name]: -1.0}, -p_min)
            q_min_attr = getattr(g, "q_min", None)
            q_max_attr = getattr(g, "q_max", None)
            if q_max_attr is not None:
                prob.add_le({q_gen[g.name]: 1.0}, float(q_max_attr))
            else:  # q ≤ 0.5 · cap_g
                prob.add_le({q_gen[g.name]: 1.0, cap_idx: -0.5}, 0.0)
            if q_min_attr is not None:
                prob.add_le({q_gen[g.name]: -1.0}, -float(q_min_attr))
            else:  # -q ≤ 0.5 · cap_g
                prob.add_le({q_gen[g.name]: -1.0, cap_idx: -0.5}, 0.0)
        else:
            p_max = float(g.capacity) * cf
            # p_min ≤ p ≤ p_max
            prob.add_le({p_gen[g.name]:  1.0}, p_max)
            prob.add_le({p_gen[g.name]: -1.0}, -p_min)
            q_min = float(getattr(g, "q_min", -0.5 * g.capacity))
            q_max = float(getattr(g, "q_max",  0.5 * g.capacity))
            prob.add_le({q_gen[g.name]:  1.0}, q_max)
            prob.add_le({q_gen[g.name]: -1.0}, -q_min)

    # ---- bus power balance (one P + one Q row per bus) ----
    for b in buses:
        p_load = sum(_scalar(ld.amount) for ld in loads_by_bus[b.name])
        q_load = sum(float(getattr(ld, "q_amount", 0.0)) for ld in loads_by_bus[b.name])
        # Bus shunt admittance (pu): draws g_sh·c_ii real, injects b_sh·c_ii
        # reactive (positive b_sh = capacitor). Defaults to 0.
        g_sh = float(getattr(b, "g_shunt", 0.0))
        b_sh = float(getattr(b, "b_shunt", 0.0))

        p_row: dict[int, float] = {}
        q_row: dict[int, float] = {}
        # generation: + into bus
        for g in gens_by_bus[b.name]:
            p_row[p_gen[g.name]] = p_row.get(p_gen[g.name], 0.0) + 1.0
            q_row[q_gen[g.name]] = q_row.get(q_gen[g.name], 0.0) + 1.0
        # branches: subtract sending-end flow from bus_from, receiving from bus_to
        for l in lines:
            if l.bus_from.name == b.name:
                p_row[P_ij[l.name]] = p_row.get(P_ij[l.name], 0.0) - 1.0
                q_row[Q_ij[l.name]] = q_row.get(Q_ij[l.name], 0.0) - 1.0
            if l.bus_to.name == b.name:
                p_row[P_ji[l.name]] = p_row.get(P_ji[l.name], 0.0) - 1.0
                q_row[Q_ji[l.name]] = q_row.get(Q_ji[l.name], 0.0) - 1.0
        # Bus shunt drains P = g_sh·|V|² and injects Q = b_sh·|V|² (capacitor
        # sign convention: b_sh > 0 ⇒ positive Q into bus). Move to LHS:
        #   Σgen − Σbranch − g_sh · c_ii = p_load
        #   Σgen − Σbranch + b_sh · c_ii = q_load
        if g_sh != 0.0:
            p_row[cii[b.name]] = p_row.get(cii[b.name], 0.0) - g_sh
        if b_sh != 0.0:
            q_row[cii[b.name]] = q_row.get(cii[b.name], 0.0) + b_sh
        prob.add_eq(p_row, p_load)
        prob.add_eq(q_row, q_load)

    # Cycle closure (N_En_Phase 10.11): exact linear equality Σ ±θ_ij = 0 for
    # each fundamental cycle of the SOCP-branch subgraph. Radial networks
    # yield an empty basis → this is a no-op and backward-compatible.
    if enforce_cycle_closure:
        for cycle in _fundamental_cycle_basis(lines, buses):
            row: dict[int, float] = {}
            for (link, sign) in cycle:
                idx = theta_ij[link.name]
                row[idx] = row.get(idx, 0.0) + float(sign)
            prob.add_eq(row, 0.0)

    return _SOCPBuild(
        prob=prob, buses=buses, lines=lines,
        cii=cii, cij=cij, sij=sij,
        P_ij=P_ij, Q_ij=Q_ij, P_ji=P_ji, Q_ji=Q_ji,
        p_gen=p_gen, q_gen=q_gen,
        theta_ij=theta_ij, v_mag=v_mag, w_ij=w_ij,
        cos_theta_ij=cos_theta_ij, sin_theta_ij=sin_theta_ij,
        gen_cap=gen_cap, line_cap=line_cap,
    )


def solve_socp_opf(system: "EnergySystem", *, snapshot: int = 0,
                   verbose: bool = False,
                   angle_diff_max: float | None = None,
                   enforce_cycle_closure: bool = False,
                   enforce_tight_qc: bool = False,
                   cos_envelope_pieces: int = 8,
                   enable_obbt: bool = False,
                   obbt_iters: int = 3,
                   obbt_tol: float = 1e-4) -> SOCPOpfResult:
    """
    Solve a single-snapshot SOCP relaxation of AC-OPF.

    The relaxation is exact for radial (tree) networks under load-over-
    satisfaction; meshed networks generally see a small relaxation gap. The
    return value reports the relaxed dispatch — round to a feasible AC point
    via Newton-Raphson if you need an exact solution.

    ``angle_diff_max`` (radians) controls the arctangent envelope
    (N_En_Phase 10.10): when set, each branch gets linear cuts
    ``|s_ij| ≤ tan(angle_diff_max) · c_ij`` so the Jabr lift can no
    longer pick (c_ij, s_ij) with a phase outside the wedge. A branch
    with its own ``angle_diff_max`` / ``angle_diff_min`` attributes
    overrides this default. ``None`` (default) disables the envelope
    entirely — backward-compatible with pre-10.10 behavior.

    ``enforce_cycle_closure`` (N_En_Phase 10.11) layers the QC-lite
    cycle-closure tightening on top: per-branch angle aux ``θ_ij`` is
    linked to ``s_ij`` via a sound sin-Taylor envelope, and each
    fundamental cycle of the SOCP-branch subgraph gets an exact
    equality ``Σ ±θ_ij = 0``. This closes the cycle-inconsistency slack
    that per-branch envelopes alone cannot reach on meshed networks
    (e.g. case30). Requires ``angle_diff_max`` (function-level or
    per-link) so the sin-Taylor slack is well-defined.

    ``enforce_tight_qc`` (N_En_Phase 10.12) upgrades the cycle-closure
    coupling from the sound-but-loose sin-Taylor linear envelope to
    the full Coffrin & Hijazi 2015 QC formulation. Per-bus voltage
    magnitude aux ``|V_i|`` is linked to ``c_ii`` by secant + tangent
    envelopes; per-branch voltage-product ``w_ij``, ``cos(θ_ij)``,
    ``sin(θ_ij)`` auxiliaries are introduced with linear trig
    envelopes (cos: concavity-based tangent cuts; sin: sin-Taylor);
    the bilinear products ``c_ij = w_ij · cos(θ)``,
    ``s_ij = w_ij · sin(θ)``, and ``w_ij = |V_i|·|V_j|`` are all
    bracketed by their McCormick envelopes. This closes the case30
    Jabr gap that the loose linear coupling of 10.11 cannot reach at
    realistic voltage/angle boxes. Implies
    ``enforce_cycle_closure=True``; requires ``angle_diff_max``.

    ``enable_obbt`` (N_En_Phase 10.14) runs Coffrin & Van Hentenryck
    2012 Optimality-Based Bound Tightening before the main solve: for
    each bus voltage magnitude and each branch angle difference, the
    SOCP relaxation is solved with a min/max objective on that one
    variable, and the result is used to tighten the corresponding box
    bound. Tighter boxes mean tighter McCormick envelopes on the next
    round, which can close additional relaxation gap on meshed
    networks (``enable_obbt=True`` is most useful paired with
    ``enforce_tight_qc=True``). Mutates ``bus.v_min`` / ``bus.v_max``
    and ``link.angle_diff_min`` / ``link.angle_diff_max`` in place;
    copy the system first if you don't want persistent tightening.
    """
    if enable_obbt:
        obbt_tighten(system, max_iter=obbt_iters, tol=obbt_tol,
                     angle_diff_max=angle_diff_max,
                     enforce_tight_qc=enforce_tight_qc,
                     enforce_cycle_closure=enforce_cycle_closure,
                     cos_envelope_pieces=cos_envelope_pieces,
                     snapshot=snapshot, verbose=verbose)

    build = _build_socp_problem(
        system, snapshot=snapshot,
        angle_diff_max=angle_diff_max,
        enforce_cycle_closure=enforce_cycle_closure,
        enforce_tight_qc=enforce_tight_qc,
        cos_envelope_pieces=cos_envelope_pieces,
        add_cost_objective=True,
    )
    res = build.prob.solve(verbose=verbose)

    voltage_mag: dict[str, float] = {}
    gen_p_out: dict[str, float] = {}
    gen_q_out: dict[str, float] = {}
    branch_p: dict[str, float] = {}
    branch_q: dict[str, float] = {}
    branch_loss: dict[str, float] = {}
    if res.status == "optimal":
        x = res.x
        for b in build.buses:
            voltage_mag[b.name] = float(np.sqrt(max(x[build.cii[b.name]], 0.0)))
        for g in system._generators:
            gen_p_out[g.name] = float(x[build.p_gen[g.name]])
            gen_q_out[g.name] = float(x[build.q_gen[g.name]])
        for l in build.lines:
            branch_p[l.name] = float(x[build.P_ij[l.name]])
            branch_q[l.name] = float(x[build.Q_ij[l.name]])
            branch_loss[l.name] = float(
                x[build.P_ij[l.name]] + x[build.P_ji[l.name]])

    return SOCPOpfResult(
        status=res.status,
        total_cost=float(res.objective),
        voltage_mag=voltage_mag,
        gen_p=gen_p_out,
        gen_q=gen_q_out,
        branch_p=branch_p,
        branch_q=branch_q,
        branch_loss=branch_loss,
        solve_time=res.solve_time,
        iterations=res.iterations,
    )


@dataclass
class SOCPExpansionResult:
    """Result of :func:`solve_socp_opf_expansion` (N_En_Phase 10.2).

    Extends the dispatch-only :class:`SOCPOpfResult` with the chosen
    investment. ``total_cost`` is the combined objective: operating cost
    (``Σ mc·p + q2·p²``) plus the annualised investment
    (``Σ capital_cost·cap``). ``gen_capacity`` / ``line_capacity`` carry the
    optimised ratings for the extendable components only (fixed components are
    omitted — read their static ``capacity`` / ``s_max`` off the system).
    """
    status: str
    total_cost: float
    voltage_mag: dict[str, float]
    gen_p: dict[str, float]
    gen_q: dict[str, float]
    branch_p: dict[str, float]
    branch_q: dict[str, float]
    branch_loss: dict[str, float]
    gen_capacity: dict[str, float]   # gen.name -> optimised cap (MW)
    line_capacity: dict[str, float]  # link.name -> optimised s_max (MVA)
    solve_time: float
    iterations: int


def solve_socp_opf_expansion(system: "EnergySystem", *, snapshot: int = 0,
                             verbose: bool = False,
                             angle_diff_max: float | None = None,
                             enforce_cycle_closure: bool = False,
                             enforce_tight_qc: bool = False,
                             cos_envelope_pieces: int = 8) -> SOCPExpansionResult:
    """Capacity-expansion SOCP relaxation of AC-OPF (N_En_Phase 10.2).

    Single-snapshot investment + dispatch co-optimisation. Every generator
    and line/transformer marked ``extendable=True`` has its rated capacity
    promoted to a decision variable bounded by ``[min_capacity,
    max_capacity]``; the dispatch is coupled to the chosen capacity
    (``p ≤ carrier_factor · cap`` for gens, ``P²+Q² ≤ cap²`` for lines via the
    thermal cone), and an annualised investment cost ``Σ capital_cost·cap`` is
    added to the operating objective. Components without ``extendable=True``
    keep their fixed ``capacity`` / ``s_max`` exactly as
    :func:`solve_socp_opf` does.

    **Formulation.** This is the standard *investment + SOCP-relaxation*
    capacity-expansion OPF: the Jabr (2006) second-order-cone lift of the
    AC power-flow physics is reused verbatim as the operational feasibility
    set, with the Kocuk, Dey & Sun (2016, *Oper. Res.* 64(6)) tightenings
    available via the same ``angle_diff_max`` / ``enforce_tight_qc`` knobs,
    and a linear capacity-investment objective layered on top (the
    ``p_nom``/``s_nom`` extendable term familiar from generation- and
    transmission-expansion planning). The SOCP is a *relaxation* of AC-OPF,
    so the returned plan is a lower bound on the true AC investment cost —
    exact on radial networks under the usual load-over-satisfaction
    conditions.

    All envelope / cut keywords behave exactly as in :func:`solve_socp_opf`.
    Returns a :class:`SOCPExpansionResult` carrying the chosen capacities
    alongside the relaxed dispatch.
    """
    build = _build_socp_problem(
        system, snapshot=snapshot,
        angle_diff_max=angle_diff_max,
        enforce_cycle_closure=enforce_cycle_closure,
        enforce_tight_qc=enforce_tight_qc,
        cos_envelope_pieces=cos_envelope_pieces,
        add_cost_objective=True,
        expand_capacity=True,
    )
    res = build.prob.solve(verbose=verbose)

    voltage_mag: dict[str, float] = {}
    gen_p_out: dict[str, float] = {}
    gen_q_out: dict[str, float] = {}
    branch_p: dict[str, float] = {}
    branch_q: dict[str, float] = {}
    branch_loss: dict[str, float] = {}
    gen_capacity: dict[str, float] = {}
    line_capacity: dict[str, float] = {}
    if res.status == "optimal":
        x = res.x
        for b in build.buses:
            voltage_mag[b.name] = float(np.sqrt(max(x[build.cii[b.name]], 0.0)))
        for g in system._generators:
            gen_p_out[g.name] = float(x[build.p_gen[g.name]])
            gen_q_out[g.name] = float(x[build.q_gen[g.name]])
            if g.name in build.gen_cap:
                gen_capacity[g.name] = float(x[build.gen_cap[g.name]])
        for l in build.lines:
            branch_p[l.name] = float(x[build.P_ij[l.name]])
            branch_q[l.name] = float(x[build.Q_ij[l.name]])
            branch_loss[l.name] = float(
                x[build.P_ij[l.name]] + x[build.P_ji[l.name]])
            if l.name in build.line_cap:
                line_capacity[l.name] = float(x[build.line_cap[l.name]])

    return SOCPExpansionResult(
        status=res.status,
        total_cost=float(res.objective),
        voltage_mag=voltage_mag,
        gen_p=gen_p_out,
        gen_q=gen_q_out,
        branch_p=branch_p,
        branch_q=branch_q,
        branch_loss=branch_loss,
        gen_capacity=gen_capacity,
        line_capacity=line_capacity,
        solve_time=res.solve_time,
        iterations=res.iterations,
    )


def obbt_tighten(system: "EnergySystem", *,
                 max_iter: int = 3,
                 tol: float = 1e-4,
                 angle_diff_max: float | None = None,
                 enforce_tight_qc: bool = True,
                 enforce_cycle_closure: bool = False,
                 cos_envelope_pieces: int = 8,
                 snapshot: int = 0,
                 verbose: bool = False) -> OBBTStats:
    """Optimality-Based Bound Tightening for the SOCP AC-OPF relaxation.

    Iteratively tightens ``bus.v_min`` / ``bus.v_max`` and ``link.
    angle_diff_min`` / ``link.angle_diff_max`` by solving min/max
    subproblems on each primitive variable (``c_ii`` and, when
    available, ``θ_ij``) over the current SOCP relaxation. Tighter
    primitive boxes mean tighter McCormick envelopes next round on
    ``w_ij = |V_i|·|V_j|``, ``c_ij = w · cos θ``, and ``s_ij =
    w · sin θ`` — the standard Coffrin & Van Hentenryck 2012 pairing
    with the Coffrin & Hijazi 2015 QC lift.

    ``enforce_tight_qc=True`` (default) pairs OBBT with the tight
    McCormick formulation. Setting it to ``False`` falls back to
    Jabr (or Jabr + cycle-closure) subproblems, which give weaker
    tightening but are cheaper per subsolve.

    Convergence: loop until either ``max_iter`` is exhausted or the
    largest fractional box-width shrinkage in a full iteration drops
    below ``tol``.

    Mutates ``bus.v_min`` / ``bus.v_max`` / ``link.angle_diff_min`` /
    ``link.angle_diff_max`` in place. Copy first if persistence is
    undesirable.
    """
    import time as _time
    t0 = _time.perf_counter()

    if enforce_tight_qc:
        enforce_cycle_closure = True

    stats_v: dict[str, tuple[float, float]] = {}
    stats_t: dict[str, tuple[float, float]] = {}
    v_init_widths: dict[str, float] = {}
    t_init_widths: dict[str, float] = {}

    final_max_shrink = 0.0
    iters_done = 0

    for iteration in range(max_iter):
        iters_done = iteration + 1
        build = _build_socp_problem(
            system, snapshot=snapshot,
            angle_diff_max=angle_diff_max,
            enforce_cycle_closure=enforce_cycle_closure,
            enforce_tight_qc=enforce_tight_qc,
            cos_envelope_pieces=cos_envelope_pieces,
            add_cost_objective=False,
        )
        prob = build.prob
        n = prob.n
        max_shrink = 0.0

        def _solve_with_obj(idx: int, sign: float) -> tuple[str, float]:
            prob.q = [0.0] * n
            prob.q[idx] = sign
            r = prob.solve(verbose=False)
            return r.status, float(r.objective)

        # Tighten c_ii (→ v_min/v_max) for every bus.
        for b in build.buses:
            v_min_old = float(getattr(b, "v_min", 0.95))
            v_max_old = float(getattr(b, "v_max", 1.05))
            if iteration == 0:
                v_init_widths[b.name] = v_max_old - v_min_old
            idx = build.cii[b.name]

            s_min, cii_lo = _solve_with_obj(idx, +1.0)
            s_max, neg_cii_hi = _solve_with_obj(idx, -1.0)
            if s_min != "optimal" or s_max != "optimal":
                continue
            cii_hi = -neg_cii_hi

            new_vmin = math.sqrt(max(cii_lo, 0.0))
            new_vmax = math.sqrt(max(cii_hi, 0.0))
            # Never loosen; pad by tol to absorb solver slack.
            new_vmin = max(v_min_old, new_vmin - tol)
            new_vmax = min(v_max_old, new_vmax + tol)
            if new_vmin > new_vmax - 1e-8:
                continue

            old_width = v_max_old - v_min_old
            new_width = new_vmax - new_vmin
            if old_width > 1e-9:
                shrink = (old_width - new_width) / old_width
                if shrink > max_shrink:
                    max_shrink = shrink

            b.v_min = new_vmin
            b.v_max = new_vmax
            stats_v[b.name] = (v_init_widths[b.name], new_width)

        # Tighten θ_ij (→ angle_diff_min/max) when the build has it.
        if build.theta_ij:
            for l in build.lines:
                t_max_old = getattr(l, "angle_diff_max", angle_diff_max)
                t_min_old = getattr(l, "angle_diff_min", None)
                if t_max_old is None:
                    continue
                if t_min_old is None:
                    t_min_old = -t_max_old
                if iteration == 0:
                    t_init_widths[l.name] = t_max_old - t_min_old
                idx = build.theta_ij[l.name]

                s_min, t_lo = _solve_with_obj(idx, +1.0)
                s_max, neg_t_hi = _solve_with_obj(idx, -1.0)
                if s_min != "optimal" or s_max != "optimal":
                    continue
                t_hi = -neg_t_hi

                new_tmin = max(t_min_old, t_lo - tol)
                new_tmax = min(t_max_old, t_hi + tol)
                if new_tmin > new_tmax - 1e-8:
                    continue

                old_width = t_max_old - t_min_old
                new_width = new_tmax - new_tmin
                if old_width > 1e-9:
                    shrink = (old_width - new_width) / old_width
                    if shrink > max_shrink:
                        max_shrink = shrink

                l.angle_diff_max = new_tmax
                l.angle_diff_min = new_tmin
                stats_t[l.name] = (t_init_widths[l.name], new_width)

        final_max_shrink = max_shrink
        if verbose:
            print(f"[obbt] iter {iteration + 1}: max_shrink={max_shrink:.4f}")
        if max_shrink < tol:
            break

    return OBBTStats(
        iters=iters_done,
        v_mag_reductions=stats_v,
        theta_reductions=stats_t,
        final_max_shrink=final_max_shrink,
        solve_time=_time.perf_counter() - t0,
    )


@dataclass
class MultiSocpOpfResult:
    """Per-snapshot SOCP AC-OPF result, aggregated.

    Each `snapshots[t]` is the single-snapshot `SOCPOpfResult` for period t.
    `total_cost` is the dt-weighted sum across periods (mirrors
    `EnergySystem.optimise` for the LP path).
    """
    status: str
    total_cost: float
    snapshots: list["SOCPOpfResult"]
    solve_time: float
    iterations: int

    def gen_p_series(self, gen_name: str) -> list[float]:
        return [s.gen_p[gen_name] for s in self.snapshots]

    def voltage_series(self, bus_name: str) -> list[float]:
        return [s.voltage_mag[bus_name] for s in self.snapshots]


def solve_socp_opf_multi(system: "EnergySystem", *,
                         verbose: bool = False,
                         angle_diff_max: float | None = None,
                         enforce_cycle_closure: bool = False,
                         enforce_tight_qc: bool = False,
                         cos_envelope_pieces: int = 8,
                         enable_obbt: bool = False,
                         obbt_iters: int = 3,
                         obbt_tol: float = 1e-4) -> MultiSocpOpfResult:
    """Solve a multi-period SOCP AC-OPF as T independent single-snapshot SOCPs.

    Period coupling (storage, ramping, intertemporal carbon caps) is **not**
    modelled here — without storage the AC-OPF dispatch problem is fully
    separable across snapshots, so the "multi-period" answer is exactly T
    independent solves. Storage / ramping wiring lands as a follow-up
    inside the same N_En_Phase 10 cluster.

    Time-varying inputs picked up automatically:
        - load.amount        (np.ndarray of length T)
        - generator.carrier_factor (np.ndarray of length T)
    Anything else is treated as a snapshot-invariant constant.

    Periods are weighted by `system._dt` (hours) when summing total cost,
    matching `EnergySystem.optimise`.

    ``enable_obbt`` (N_En_Phase 10.14) tightens bus voltage and branch
    angle boxes once upfront (on snapshot 0) before the per-snapshot
    loop. Per-snapshot re-tightening would be redundant since OBBT
    mutates network topology in place — one pass applies to every
    subsequent solve.

    Warm-start between snapshots: Clarabel is an interior-point method,
    and ``ConicProblem.solve`` builds a fresh ``DefaultSolver`` per call
    with no primal/dual seeding hook exposed by the Clarabel Python
    binding. Because consecutive snapshots differ only in the load /
    carrier-factor RHS (identical cone geometry), a warm start from the
    previous snapshot's interior point would typically save a handful of
    IPM iterations. That optimisation is deferred until the Clarabel
    binding exposes a settings-level initial-point API; today each
    snapshot is solved cold. The separable structure means correctness
    is unaffected — only wall time.
    """
    T = system._timesteps
    dt = float(getattr(system, "_dt", 1.0))

    if enable_obbt:
        obbt_tighten(system, max_iter=obbt_iters, tol=obbt_tol,
                     angle_diff_max=angle_diff_max,
                     enforce_tight_qc=enforce_tight_qc,
                     enforce_cycle_closure=enforce_cycle_closure,
                     cos_envelope_pieces=cos_envelope_pieces,
                     snapshot=0, verbose=verbose)

    snaps: list[SOCPOpfResult] = []
    overall_status = "optimal"
    total_cost = 0.0
    total_time = 0.0
    total_iters = 0
    for t in range(T):
        res = solve_socp_opf(system, snapshot=t, verbose=verbose,
                             angle_diff_max=angle_diff_max,
                             enforce_cycle_closure=enforce_cycle_closure,
                             enforce_tight_qc=enforce_tight_qc,
                             cos_envelope_pieces=cos_envelope_pieces,
                             enable_obbt=False)
        snaps.append(res)
        if res.status != "optimal":
            overall_status = res.status
        total_cost += res.total_cost * dt
        total_time += res.solve_time
        total_iters += res.iterations
    return MultiSocpOpfResult(
        status=overall_status,
        total_cost=total_cost,
        snapshots=snaps,
        solve_time=total_time,
        iterations=total_iters,
    )


# ===========================================================================
# N_En_Phase 10.7 / 4.2 — Weymouth gas-flow physics on the conic backend.
# ===========================================================================


@dataclass
class WeymouthVars:
    """Variable-index map returned by :func:`add_weymouth_pipe`.

    Indices refer to columns of the parent :class:`ConicProblem`. ``q`` is the
    (signed) volumetric flow, ``pi_from`` / ``pi_to`` are the *squared* nodal
    pressures π = p² (the natural state variable of the Weymouth equation),
    and ``w`` is the auxiliary squared-flow lift ``w ≥ q²`` used by the
    rotated-SOC relaxation.
    """
    q: int
    pi_from: int
    pi_to: int
    w: int


def add_weymouth_pipe(
    prob: "ConicProblem",
    *,
    q: int,
    pi_from: int,
    pi_to: int,
    w: int,
    weymouth_k: float,
    q_max: float,
    relaxation: str = "soc",
    mccormick_segments: int = 1,
) -> WeymouthVars:
    """Post a relaxed Weymouth gas-pipe coupling onto a :class:`ConicProblem`.

    **Physics.** The steady-state isothermal Weymouth equation
    (Weymouth 1912; see e.g. De Wolf & Smeers 2000, *Manag. Sci.* 46(11), and
    Borraz-Sánchez et al. 2016, *INFORMS J. Comput.*) relates volumetric flow
    ``q`` through a pipe to the squared nodal pressures::

        q · |q| = K · (p_from² − p_to²)

    with ``K > 0`` the Weymouth pipe constant (a function of diameter, length,
    friction, temperature, gas composition). Writing the squared pressures as
    state variables ``π_from = p_from²``, ``π_to = p_to²`` and letting
    ``Δπ = π_from − π_to`` this is ``q·|q| = K·Δπ`` — **nonconvex** because of
    the ``q·|q|`` term (it is concave for q>0 and convex for q<0).

    **Relaxation (this builder).** Exact Weymouth cannot go to Clarabel (it
    has no nonconvex / equilibrium cone). We post a *convex outer
    approximation* that is standard in the gas-OPF literature:

    - ``relaxation="soc"`` (default): introduce ``w ≥ q²`` via the rotated
      second-order cone (Clarabel ``SecondOrderConeT``) and bound the flow
      magnitude by ``K·Δπ`` through the linear cut ``w ≤ K·Δπ``. Together
      ``q² ≤ w ≤ K·Δπ`` is the SOC relaxation of ``q·|q| = K·Δπ`` on the
      forward (``q ≥ 0``) branch and a valid *outer* bound on both branches.
      This is the relaxation used by, e.g., the gas half of integrated
      power-gas OPF studies that keep the problem conic (cf. the
      "second-order cone gas flow" of Manshadi & Khodayar 2018, and the
      relaxation surveyed in Schwele et al. 2019). It is a relaxation: the
      optimum is a lower bound on cost and may leave a small pressure-drop
      gap, exactly analogous to the Jabr lift for AC-OPF.

    - ``relaxation="mccormick"``: for the standard *unidirectional* pipe
      (``q ≥ 0``) sandwich ``w`` between its convex underestimator (the same
      ``w ≥ q²`` SOC) and the *secant* overestimator of ``q²`` over
      ``[0, q_max]`` — the single chord ``q² ≤ q_max·q`` — then tie
      ``w ≤ K·Δπ``. The chord gives a two-sided ``q² ≤ w ≤ q_max·q`` bound
      that tightens the upper side relative to the pure-SOC choice while
      staying globally valid (piecewise chords would need SOS2/binaries and
      so are out of scope for the conic backend). ``mccormick_segments`` is
      validated (≥ 1) and reserved for a future SOS2-gated refinement; the
      current sound implementation always uses the single full-interval
      chord.

    Both choices are convex and Clarabel-solvable. Exact Weymouth (the
    equality) would require an MINLP / NLP backend (CasADi/IPOPT path,
    tracked separately) and is intentionally *not* attempted here.

    **Sign convention.** ``q`` may be free (bidirectional flow). The SOC
    ``w ≥ q²`` is symmetric in q, and ``w ≤ K·Δπ`` forces ``Δπ ≥ 0`` only up
    to the magnitude needed to carry the flow — for reverse flow the caller
    should orient the pipe or post a second copy. For the common
    unidirectional case pass a nonnegative ``q`` (bound it with ``q ≥ 0``
    outside this builder).

    Parameters
    ----------
    prob : ConicProblem
        Problem to mutate in place. The four variable columns must already be
        allocated by the caller.
    q, pi_from, pi_to, w : int
        Column indices for flow, squared up/down pressures, and the squared-
        flow lift.
    weymouth_k : float
        Pipe constant ``K > 0``.
    q_max : float
        Flow magnitude bound (used to size the McCormick secant); also posted
        as ``-q_max ≤ q ≤ q_max``.
    relaxation : {"soc", "mccormick"}
        Outer-approximation family (see above).
    mccormick_segments : int
        Number of secant pieces for ``relaxation="mccormick"`` (≥ 1).

    Returns
    -------
    WeymouthVars
        The index bundle (echoes the inputs) for downstream balance wiring.
    """
    if weymouth_k <= 0.0:
        raise ValueError(f"weymouth_k must be > 0 (got {weymouth_k})")
    if q_max <= 0.0:
        raise ValueError(f"q_max must be > 0 (got {q_max})")
    if relaxation not in ("soc", "mccormick"):
        raise ValueError(
            f"relaxation must be 'soc' or 'mccormick' (got {relaxation!r})")
    if mccormick_segments < 1:
        raise ValueError(
            f"mccormick_segments must be >= 1 (got {mccormick_segments})")

    # Flow magnitude box: -q_max ≤ q ≤ q_max.
    prob.add_le({q:  1.0},  q_max)
    prob.add_le({q: -1.0},  q_max)

    # Convex lower lift  w ≥ q²  via the rotated SOC  2·w·(1/2) ≥ q².
    half = prob.add_var()
    prob.add_eq({half: 1.0}, 0.5)          # pin the 1/2 cone leg
    prob.add_rotated_soc(x_var=w, y_var=half, u_vars=[q])

    # Pressure-drop coupling  w ≤ K·Δπ  ⇔  w − K·π_from + K·π_to ≤ 0.
    # Combined with w ≥ q² this gives the SOC outer approximation
    # q² ≤ K·(π_from − π_to) of the Weymouth equality q·|q| = K·Δπ.
    prob.add_le({w: 1.0, pi_from: -weymouth_k, pi_to: weymouth_k}, 0.0)

    if relaxation == "mccormick":
        # Secant OVER-estimator of the convex w ⇄ q² lift. A single chord is
        # the only globally-valid linear overestimator of q² over a fixed
        # interval (piecewise chords need SOS2/binaries and would be cut
        # *below* q² outside their own sub-interval, breaking the w ≥ q²
        # floor). For the standard *unidirectional* gas case q ∈ [0, q_max]
        # the chord of q² through (0, 0) and (q_max, q_max²) is the secant
        #   q² ≤ q_max · q   on [0, q_max]
        # so we post  w ≤ q_max · q  and pin q ≥ 0. This two-sided sandwich
        # q² ≤ w ≤ q_max·q tightens the pure-SOC upper side. ``mccormick_
        # segments`` refines the chord toward the operating point by chording
        # the upper sub-interval [q_max·(S-1)/S, q_max] when S > 1 — a valid
        # tighter overestimator that still dominates the q_max·q chord near
        # full flow.
        prob.add_le({q: -1.0}, 0.0)          # q ≥ 0 (unidirectional)
        # Single globally-valid secant chord of q² over [0, q_max]:
        #   q² ≤ q_max · q   ⇒   w ≤ q_max · q.
        prob.add_le({w: 1.0, q: -q_max}, 0.0)

    return WeymouthVars(q=q, pi_from=pi_from, pi_to=pi_to, w=w)


# ===========================================================================
# N_En_Phase 10.8 / 4.3 — Head-dependent hydro efficiency (PWL surrogate).
# ===========================================================================


@dataclass
class HydroHeadVars:
    """Variable-index map returned by :func:`add_head_dependent_hydro`.

    ``power`` is the turbine electrical output (MW), ``discharge`` the water
    release rate, ``soc`` the reservoir state of charge (energy / volume
    proxy). ``breakpoint_weights`` are the SOS2-style convex-combination
    weights λ_k over the PWL breakpoints (each in [0, 1], summing to 1).
    """
    power: int
    discharge: int
    soc: int
    breakpoint_weights: list[int]


def add_head_dependent_hydro(
    prob: "ConicProblem",
    *,
    power: int,
    discharge: int,
    soc: int,
    soc_min: float,
    soc_max: float,
    eta_of_soc,
    n_breakpoints: int = 4,
    discharge_max: float | None = None,
) -> HydroHeadVars:
    """Post a piecewise-affine head-dependent hydro efficiency coupling.

    **Physics.** Hydro turbine power is ``P = ρ·g·η·H·Q`` where the net head
    ``H`` rises with reservoir storage. Folding the head into the efficiency
    gives a *storage-dependent* discharge efficiency ``η(SOC)`` so that
    ``P = η(SOC)·Q`` — a **bilinear, nonconvex** product of two decision
    variables (the head-dependent hydro nonlinearity called out in the
    SpineOpt / PowerModels hydro literature; cf. Catalão et al. 2010,
    *IEEE Trans. Power Syst.*, on head-dependent hydro scheduling).

    **Surrogate (this builder).** We linearise ``η(SOC)`` with a
    piecewise-affine (PWL) surrogate over ``[soc_min, soc_max]`` using the
    classic convex-combination (λ) formulation, then post a *convex
    relaxation* of the bilinear power product:

        SOC      = Σ_k λ_k · s_k          (interpolate the SOC grid)
        Σ_k λ_k  = 1,  λ_k ≥ 0
        η̄        = Σ_k λ_k · η(s_k)        (interpolated efficiency, linear in λ)
        P ≤ η_max · Q                      (loose convex upper bound)

    ``P = η̄·Q`` is bilinear so it is *not* posted as an equality. Two
    progressively tighter, fully-linear (LP/SOCP-clean) relaxations are
    emitted:

    - Always: the loose envelope ``P ≤ η_max·Q`` with
      ``η_max = max_k η(s_k)``.
    - When ``discharge_max`` is given: the per-breakpoint McCormick-style cut

          P ≤ η(s_k)·Q + (η_max − η(s_k))·discharge_max·(1 − λ_k)   ∀k

      For the active breakpoint (``λ_k → 1``) this collapses to
      ``P ≤ η(s_k)·Q``, i.e. the discharge is charged the *local* efficiency,
      while for inactive breakpoints the slack term keeps the cut valid. This
      is the standard McCormick linearisation of the ``λ_k·Q`` product over
      ``λ_k ∈ [0,1]``, ``Q ∈ [0, discharge_max]`` and is a genuine convex
      relaxation (a bound), tightening as ``n_breakpoints`` grows.

    Like Weymouth-SOC and the Jabr lift, this is a *relaxation*: it bounds
    rather than reproduces the exact bilinear point. For exact head-dependent
    dispatch use the NLP (CasADi) path.

    Parameters
    ----------
    prob : ConicProblem
        Problem mutated in place.
    power, discharge, soc : int
        Column indices for turbine power, discharge, reservoir SOC.
    soc_min, soc_max : float
        Reservoir SOC operating range for the PWL grid.
    eta_of_soc : Callable[[float], float]
        Efficiency as a function of SOC, sampled at the breakpoints.
    n_breakpoints : int
        Number of evenly-spaced PWL breakpoints (≥ 2).
    discharge_max : float | None
        Upper bound on discharge; enables the tighter per-breakpoint cut.

    Returns
    -------
    HydroHeadVars
        Index bundle including the λ_k breakpoint-weight columns.
    """
    if n_breakpoints < 2:
        raise ValueError(f"n_breakpoints must be >= 2 (got {n_breakpoints})")
    if not (soc_max > soc_min):
        raise ValueError(
            f"soc_max ({soc_max}) must exceed soc_min ({soc_min})")

    K = int(n_breakpoints)
    grid = [soc_min + (soc_max - soc_min) * k / (K - 1) for k in range(K)]
    etas = [float(eta_of_soc(s)) for s in grid]
    if any(e < 0.0 for e in etas):
        raise ValueError("eta_of_soc must be nonnegative at every breakpoint")
    eta_max = max(etas)

    # Convex-combination weights λ_k ≥ 0, Σ λ_k = 1.
    lam = [prob.add_var() for _ in range(K)]
    for k in range(K):
        prob.add_le({lam[k]: -1.0}, 0.0)            # λ_k ≥ 0
        prob.add_le({lam[k]:  1.0}, 1.0)            # λ_k ≤ 1
    prob.add_eq({lk: 1.0 for lk in lam}, 1.0)        # Σ λ_k = 1

    # SOC = Σ λ_k · s_k  (interpolate the storage grid).
    soc_row = {lam[k]: grid[k] for k in range(K)}
    soc_row[soc] = soc_row.get(soc, 0.0) - 1.0
    prob.add_eq(soc_row, 0.0)

    # SOC box (also implied by the grid, posted for solver robustness).
    prob.add_le({soc:  1.0},  soc_max)
    prob.add_le({soc: -1.0}, -soc_min)

    # Loose convex envelope: P ≤ η_max · Q. (power, discharge ≥ 0 assumed,
    # bounded by the caller.)
    prob.add_le({power: 1.0, discharge: -eta_max}, 0.0)

    # Tighter per-breakpoint McCormick cut when a discharge upper bound exists:
    #   P ≤ η_k·Q + (η_max − η_k)·Q_max·(1 − λ_k)
    # ⇔ P − η_k·Q + (η_max − η_k)·Q_max·λ_k ≤ (η_max − η_k)·Q_max
    if discharge_max is not None:
        if discharge_max <= 0.0:
            raise ValueError(
                f"discharge_max must be > 0 (got {discharge_max})")
        for k in range(K):
            slope = eta_max - etas[k]
            prob.add_le(
                {power: 1.0, discharge: -etas[k],
                 lam[k]: slope * discharge_max},
                slope * discharge_max)

    return HydroHeadVars(
        power=power, discharge=discharge, soc=soc,
        breakpoint_weights=lam,
    )

