"""Thermal-unit modelling helpers.

Implements the Phase 2 tight UC formulations:
  * 3-bin unit commitment (Morales-España 2013): separate startup `v[t]`
    and shutdown `w[t]` indicators linked to status `u[t]` via
    `u[t] - u[t-1] = v[t] - w[t]` with mutex `v[t] + w[t] <= 1`.
  * Rajan-Takriti tight min-up / min-down (polynomial convex-hull form):
      sum_{s=t-TU+1..t} v[s] <= u[t]        (min-up)
      sum_{s=t-TD+1..t} w[s] <= 1 - u[t]    (min-down)
    These are LP-relaxation tight for the single-unit on/off polytope
    (Rajan & Takriti 2005; Gentile, Morales-España, Ramos 2017).
  * Linearized clustered UC (GenX ``UCommit: 2`` analogue): relax u/v/w
    to continuous in [0, n_units] and lump N identical units together.
  * PWL heat rate / part-load efficiency via segment variables that sum
    to dispatch; convex increasing marginal cost is LP-naturally tight.

All functions mutate the passed-in generator's internal var lists in
place; the caller (`core.EnergySystem.optimise`) uses those later when
building the objective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import Generator


def build_three_bin_uc(model, gen: "Generator", T: int,
                       available_cap_per_t,
                       cap_var=None) -> None:
    """Create u / v / w commitment variables and add state-transition,
    mutex, Rajan-Takriti min-up/down, and p-min/p-max constraints.

    Args:
        model: nexus-opt Model instance.
        gen: Generator dataclass with ``committable=True``.
        T: horizon length in timesteps.
        available_cap_per_t: callable (t: int) -> float giving the
            per-timestep per-unit (clustered) or total (binary) upper
            bound on dispatch. For clustered + extendable this must be
            the per-unit nameplate (``unit_size × carrier_factor``) so
            ``p ≤ cap_t × u`` stays linear in cap_var.
        cap_var: optional cap_var handle (the extendable capacity
            decision variable). When provided AND ``gen.clustered`` is
            True AND ``gen.unit_size > 0``, adds the per-timestep
            commitment-bound constraint ``unit_size × u[t] ≤ cap_var``
            (Phase 14.x — linearises the clustered UC + extendable
            bilinearity by treating the effective capacity as
            per-unit × committed-count). No-op for non-clustered or
            when ``unit_size`` is the default 1.0 fallback.
    """
    # Status upper bound. For clustered + extendable we size against the
    # maximum possible number of built units (``max_capacity / unit_size``)
    # so u[t] can scale with the investment decision; the per-timestep
    # ``unit_size × u[t] ≤ cap_var`` constraint below binds it against
    # what was actually built.
    if gen.clustered:
        if cap_var is not None and gen.unit_size > 0:
            if gen.max_capacity != float("inf"):
                upper = float(gen.max_capacity) / float(gen.unit_size)
            else:
                # No investment limit: pick a loose but finite bound; the
                # ``unit_size × u[t] ≤ cap_var`` constraint added below is
                # what actually binds u[t] to what was built.
                upper = 1.0e6
        else:
            upper = float(gen.n_units)
    else:
        upper = 1.0

    # Create u, v, w.
    gen._status_vars = []
    gen._startup_vars = []
    gen._shutdown_vars = []
    for t in range(T):
        if gen.clustered:
            u = model.variable(f"u_{gen.name}_{t}", lower=0.0, upper=upper)
        else:
            u = model.binary(f"u_{gen.name}_{t}")
        gen._status_vars.append(u)
    # LP-tight inequality form with upper-bound cuts. Clustered uses
    # equality + mutex (Morales-España); non-clustered uses the tight
    # inequality form where v,w are continuous [0,1] but uniquely
    # determined by u at integer solutions.
    for t in range(T):
        if gen.clustered:
            v = model.variable(f"v_{gen.name}_{t}", lower=0.0, upper=upper)
            w = model.variable(f"w_{gen.name}_{t}", lower=0.0, upper=upper)
        else:
            v = model.variable(f"v_{gen.name}_{t}", lower=0.0, upper=1.0)
            w = model.variable(f"w_{gen.name}_{t}", lower=0.0, upper=1.0)
        gen._startup_vars.append(v)
        gen._shutdown_vars.append(w)

    u_init = float(min(gen.initial_status, 1) if not gen.clustered
                   else min(gen.initial_status, gen.n_units))

    if gen.clustered:
        model.add(
            gen._startup_vars[0] - gen._shutdown_vars[0]
            == gen._status_vars[0] - u_init,
            name=f"uc_trans_{gen.name}_0",
        )
        model.add(
            gen._startup_vars[0] + gen._shutdown_vars[0] <= upper,
            name=f"uc_mutex_{gen.name}_0",
        )
        for t in range(1, T):
            model.add(
                gen._startup_vars[t] - gen._shutdown_vars[t]
                == gen._status_vars[t] - gen._status_vars[t - 1],
                name=f"uc_trans_{gen.name}_{t}",
            )
            model.add(
                gen._startup_vars[t] + gen._shutdown_vars[t] <= upper,
                name=f"uc_mutex_{gen.name}_{t}",
            )
    else:
        model.add(
            gen._startup_vars[0] >= gen._status_vars[0] - u_init,
            name=f"uc_trans_up_{gen.name}_0",
        )
        model.add(
            gen._shutdown_vars[0] >= u_init - gen._status_vars[0],
            name=f"uc_trans_dn_{gen.name}_0",
        )

        for t in range(1, T):
            model.add(
                gen._startup_vars[t]
                >= gen._status_vars[t] - gen._status_vars[t - 1],
                name=f"uc_trans_up_{gen.name}_{t}",
            )
            model.add(
                gen._shutdown_vars[t]
                >= gen._status_vars[t - 1] - gen._status_vars[t],
                name=f"uc_trans_dn_{gen.name}_{t}",
            )


    # Initial min-up/min-down enforcement. If the unit was on for
    # up_time_before < min_up_time hours, it must remain on for the
    # remaining hours. Same logic for down_time_before / min_down_time.
    TU = int(gen.min_up_time) if gen.min_up_time else 0
    TD = int(gen.min_down_time) if gen.min_down_time else 0

    if gen.initial_status > 0 and TU > 0 and gen.up_time_before > 0:
        remaining_up = TU - int(gen.up_time_before)
        if remaining_up > 0:
            for t in range(min(remaining_up, T)):
                model.add(gen._status_vars[t] >= u_init,
                          name=f"init_up_{gen.name}_{t}")

    if gen.initial_status == 0 and TD > 0 and gen.down_time_before > 0:
        remaining_down = TD - int(gen.down_time_before)
        if remaining_down > 0:
            for t in range(min(remaining_down, T)):
                model.add(gen._status_vars[t] <= 0,
                          name=f"init_down_{gen.name}_{t}")

    # Rajan-Takriti min-up: sum_{s=t-TU+1..t} v[s] <= u[t]  for t >= TU-1.
    if TU > 1:
        for t in range(TU - 1, T):
            acc = None
            for s in range(t - TU + 1, t + 1):
                term = gen._startup_vars[s]
                acc = term if acc is None else acc + term
            model.add(acc <= gen._status_vars[t],
                      name=f"min_up_{gen.name}_{t}")

    # Rajan-Takriti min-down: sum_{s=t-TD+1..t} w[s] <= upper - u[t].
    if TD > 1:
        for t in range(TD - 1, T):
            acc = None
            for s in range(t - TD + 1, t + 1):
                term = gen._shutdown_vars[s]
                acc = term if acc is None else acc + term
            model.add(acc <= upper - gen._status_vars[t],
                      name=f"min_down_{gen.name}_{t}")

    # p >= p_min * u and p <= cap(t) * u.
    # For clustered: cap(t) is per-unit nameplate × carrier_factor, and u
    # aggregates N units → total on-capacity = cap_t × u. p_min is per-unit
    # minimum, scaling with u similarly.
    for t in range(T):
        cap_t = available_cap_per_t(t)
        model.add(gen._p_vars[t] >= gen.p_min * gen._status_vars[t],
                  name=f"uc_min_{gen.name}_{t}")
        model.add(gen._p_vars[t] <= cap_t * gen._status_vars[t],
                  name=f"uc_max_{gen.name}_{t}")

    # Phase 14.x — commitment-bound linearisation for clustered + extendable:
    # ensure committed units never exceed built units.
    # ``u[t] × unit_size ≤ cap_var`` is linear because unit_size is a
    # constant and both u and cap_var are continuous decision variables.
    # The per-unit ``cap_t`` above (supplied as unit_size × cf[t] by the
    # caller) then makes ``p ≤ cap_t × u`` linear too — no bilinear term.
    if cap_var is not None and gen.clustered and gen.unit_size > 0:
        for t in range(T):
            model.add(
                gen.unit_size * gen._status_vars[t] <= cap_var,
                name=f"uc_commit_built_{gen.name}_{t}",
            )

    # Phase 2.2 — multi-state (hot / warm / cold) startup cost.
    if gen.start_up_segments and not gen.clustered:
        add_multistate_startup(model, gen, T)


def add_multistate_startup(model, gen: "Generator", T: int) -> None:
    """Tight multi-state (hot / warm / cold) startup-cost formulation.

    Morales-España, Latorre & Ramos (2013), "Tight and Compact MILP
    Formulation of Start-Up and Shut-Down Ramping in Unit Commitment".
    A start at ``t`` is assigned exactly one *start type* ``s`` whose cost
    rises with how long the unit has been offline. The temperature is
    inferred from the shutdown indicators ``w``: start type ``s`` (any but
    the coldest) is only usable if the most recent shutdown happened inside
    that type's offline window.

    ``gen.start_up_segments`` is ``[(min_off_0, cost_0), …]`` sorted ascending
    by ``min_off`` with ``min_off_0 == 0`` (hottest) and non-decreasing costs.
    The last entry is the *coldest* (residual) start, always feasible.

    For each ``t`` and non-coldest type ``s`` with offline window
    ``[L_s, L_{s+1})`` (in timesteps):

        δ[t,s] ≤ Σ_{i = L_s}^{L_{s+1}-1} w[t-i]

    and ``Σ_s δ[t,s] == v[t]`` ties the start-type selection to the start
    indicator. The objective term ``Σ_s cost_s · δ[t,s]`` is emitted by the
    caller (``core.optimise``) which reads ``gen._start_type_vars``.
    """
    segs = sorted(gen.start_up_segments, key=lambda x: x[0])
    S = len(segs)
    if S == 0:
        return
    # Offline-window lower thresholds (in timesteps), e.g. [0, 4, 12].
    thresholds = [int(seg[0]) for seg in segs]

    gen._start_type_vars = []
    for t in range(T):
        row = [model.variable(f"startseg_{gen.name}_{t}_{s}", lower=0.0, upper=1.0)
               for s in range(S)]
        gen._start_type_vars.append(row)

    for t in range(T):
        # A start is exactly one type: Σ_s δ[t,s] == v[t].
        acc = None
        for s in range(S):
            acc = gen._start_type_vars[t][s] if acc is None \
                else acc + gen._start_type_vars[t][s]
        model.add(acc == gen._startup_vars[t],
                  name=f"startseg_sum_{gen.name}_{t}")

        # Hot / warm types (every type except the coldest) are gated by the
        # shutdown history inside their offline window [thresholds[s],
        # thresholds[s+1]). The coldest type (s == S-1) is the residual and
        # is always available, so no upper-bound constraint is emitted for it.
        for s in range(S - 1):
            lo = thresholds[s]
            hi = thresholds[s + 1]
            lookback = None
            for i in range(lo, hi):
                ti = t - i
                if ti < 0:
                    continue  # before horizon — unit assumed cold/at initial state
                term = gen._shutdown_vars[ti]
                lookback = term if lookback is None else lookback + term
            if lookback is None:
                # No in-horizon shutdowns can place the unit in this band →
                # this start type is infeasible at t.
                model.add(gen._start_type_vars[t][s] <= 0.0,
                          name=f"startseg_gate_{gen.name}_{t}_{s}")
            else:
                model.add(gen._start_type_vars[t][s] <= lookback,
                          name=f"startseg_gate_{gen.name}_{t}_{s}")


def add_must_run(model, gen: "Generator", T: int) -> None:
    """Force a generator always-on.

    For committable gens: fix u[t] = upper (1 for binary, n_units for
    clustered). For non-committable: force p[t] >= capacity at every t
    (uses the existing gen._p_vars list).
    """
    if gen.committable and gen._status_vars:
        upper = float(gen.n_units) if gen.clustered else 1.0
        for t in range(T):
            model.add(gen._status_vars[t] == upper,
                      name=f"must_run_u_{gen.name}_{t}")
    else:
        # Force dispatch to capacity (allows time-varying availability to
        # temper this via carrier_factor already baked into p's upper bound).
        for t in range(T):
            cap = gen.capacity
            if gen.carrier_factor is not None:
                cap = gen.capacity * float(gen.carrier_factor[t])
            model.add(gen._p_vars[t] >= cap,
                      name=f"must_run_p_{gen.name}_{t}")


def build_pwl_heat_rate(model, gen: "Generator", T: int) -> list:
    """Create per-segment dispatch variables and return the term to add
    to the objective (sum over t of sum_k mc_k * s_k_t).

    ``gen.heat_rate_segments`` must be a list of ``(p_breakpoint_MW,
    marginal_cost_$/MWh)`` pairs sorted ascending in p. The cost function
    is convex piecewise-linear — the LP will always fill the cheapest
    segment first so no ordering binaries are needed.

    Replaces the flat ``gen.marginal_cost * p[t]`` term in the objective
    when set. The variable-OM ``gen.marginal_cost`` is still applied
    (it's a separate $/MWh line item, not fuel).
    """
    breakpoints = list(gen.heat_rate_segments)
    if len(breakpoints) < 2:
        raise ValueError(
            f"heat_rate_segments for {gen.name!r} needs >= 2 breakpoints; "
            f"got {len(breakpoints)}"
        )
    # Segment widths (MW) and per-segment marginal costs ($/MWh).
    widths = []
    seg_mcs = []
    for k in range(1, len(breakpoints)):
        p_lo, _ = breakpoints[k - 1]
        p_hi, mc_hi = breakpoints[k]
        widths.append(p_hi - p_lo)
        seg_mcs.append(mc_hi)

    gen._seg_vars = []
    cost_term = None
    for t in range(T):
        seg_t = []
        sum_seg = None
        for k, w in enumerate(widths):
            s = model.variable(f"seg_{gen.name}_{t}_{k}", lower=0.0, upper=w)
            seg_t.append(s)
            sum_seg = s if sum_seg is None else sum_seg + s
            # Accumulate cost.
            term = seg_mcs[k] * s
            cost_term = term if cost_term is None else cost_term + term
        gen._seg_vars.append(seg_t)
        # Sum of segments equals dispatch.
        model.add(sum_seg == gen._p_vars[t], name=f"pwl_sum_{gen.name}_{t}")

    return cost_term


def add_regulation_reserve_vars(model, gen: "Generator", T: int,
                                cap_fn) -> None:
    """Create per-timestep reg_up / reg_down variables for a generator.

    Args:
        cap_fn: callable (t: int) -> capacity_expr_at_t (a number or a
            variable). Used to upper-bound the reserve by
            ``reg_{up,down}_max * cap``.
    """
    if gen.reg_up_max > 0.0:
        gen._reg_up_vars = []
        for t in range(T):
            r = model.variable(f"reg_up_{gen.name}_{t}", lower=0.0)
            gen._reg_up_vars.append(r)
            # Sized against capacity × participation fraction.
            cap_t = cap_fn(t)
            model.add(r <= gen.reg_up_max * cap_t,
                      name=f"reg_up_cap_{gen.name}_{t}")
            # Headroom: p + reg_up <= capacity * u (if committable) else cap.
            if gen.committable and gen._status_vars:
                model.add(
                    gen._p_vars[t] + r <= cap_t * gen._status_vars[t],
                    name=f"reg_up_hr_{gen.name}_{t}",
                )
            else:
                model.add(gen._p_vars[t] + r <= cap_t,
                          name=f"reg_up_hr_{gen.name}_{t}")
    if gen.reg_down_max > 0.0:
        gen._reg_down_vars = []
        for t in range(T):
            r = model.variable(f"reg_down_{gen.name}_{t}", lower=0.0)
            gen._reg_down_vars.append(r)
            cap_t = cap_fn(t)
            model.add(r <= gen.reg_down_max * cap_t,
                      name=f"reg_down_cap_{gen.name}_{t}")
            # Footroom: p - reg_down >= p_min * u (if committable) else 0.
            if gen.committable and gen._status_vars:
                model.add(
                    gen._p_vars[t] - r >= gen.p_min * gen._status_vars[t],
                    name=f"reg_down_fr_{gen.name}_{t}",
                )
            else:
                model.add(gen._p_vars[t] - r >= 0.0,
                          name=f"reg_down_fr_{gen.name}_{t}")
