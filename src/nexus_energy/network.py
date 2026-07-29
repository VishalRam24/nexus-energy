"""Network-physics helpers — Phase 3.

DC-OPF (linearised KVL), PTDF, transmission switching, and flow-based
N-1 security constraints.

Notes on scope:
  * SOCP AC-OPF is deferred to Phase 10 — nexus-opt's MILP solver
    (HiGHS) does not expose quadratic / conic constraints; that
    capability lands with the multi-solver Clarabel / Mosek backends.
  * HVDC lines are already modelled as controllable Links in nexus
    (no KVL); this module keeps that behaviour by only injecting
    DC-OPF constraints for Links with ``model_type='dc_opf'``.

Typical use (called from ``core.EnergySystem.optimise``):

    from . import network as nwk
    nwk.build_dc_opf_constraints(model, system, T)
    nwk.build_ptdf_constraints(model, system, T)
    nwk.build_transmission_switching(model, system, T)
    nwk.build_n_minus_1_constraints(model, system, T)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .core import EnergySystem, Link


# Big-M for transmission switching / N-1 big-M relaxations. Scaled per
# line by capacity so the relaxation isn't looser than necessary.
_THETA_BOUND = 1e6  # radians — effectively unbounded; slack bus pinned at 0
                    # provides the angle reference. Real systems run at |theta|
                    # < a few radians; large reactances or stress tests can
                    # push higher, so we don't impose a tight π-style cap.


def _dc_opf_lines(system: "EnergySystem") -> list:
    return [l for l in system._links if l.model_type == "dc_opf" and l.reactance > 0]


def _ptdf_lines(system: "EnergySystem") -> list:
    return [l for l in system._links if l.model_type == "ptdf" and l.reactance > 0]


def build_theta_vars(model, system: "EnergySystem", T: int) -> None:
    """Create phase-angle variables on every bus when any DC-OPF line exists.

    Fixes bus 0 as the slack reference (theta_0 = 0 at every t). Creates
    the vars on Bus._theta_vars for later lookup by DC-OPF / PTDF / N-1.
    """
    if not (_dc_opf_lines(system) or _ptdf_lines(system)):
        return
    for bus in system._buses:
        bus._theta_vars = []
        for t in range(T):
            v = model.variable(
                f"theta_{bus.name}_{t}",
                lower=-_THETA_BOUND, upper=_THETA_BOUND,
            )
            bus._theta_vars.append(v)
    # Slack: fix theta of bus 0 to 0.
    slack = system._buses[0]
    for t in range(T):
        model.add(slack._theta_vars[t] == 0.0, name=f"slack_theta_{slack.name}_{t}")


def build_dc_opf_constraints(model, system: "EnergySystem", T: int) -> None:
    """For every Link with ``model_type='dc_opf'`` and ``reactance > 0``,
    create a signed flow variable and link it to the phase-angle
    difference via Kirchhoff's voltage law:

        f_signed[t] = (theta_from[t] - theta_to[t]) / reactance

    The signed flow supersedes the transport fwd / rev variables — the
    DC-OPF block intentionally clears those lists so the bus-balance
    loop in ``core.py`` uses only the signed flow.
    """
    dc_lines = _dc_opf_lines(system)
    if not dc_lines:
        return

    for link in dc_lines:
        cap = link.capacity if link.capacity > 0 else 1e12
        link._flow_signed_vars = []
        for t in range(T):
            fs = model.variable(
                f"fsigned_{link.name}_{t}",
                lower=-cap, upper=cap,
            )
            link._flow_signed_vars.append(fs)
            theta_f = link.bus_from._theta_vars[t]
            theta_t = link.bus_to._theta_vars[t]
            # KVL: x * f_signed = theta_from - theta_to (rearranged to
            # avoid a 1/x constant that can blow up for very small x).
            model.add(
                link.reactance * fs == theta_f - theta_t,
                name=f"kvl_{link.name}_{t}",
            )
        # Clear the transport-model flow vars so the bus-balance loop
        # in core.py does not double-count.
        link._flow_vars = []
        link._flow_rev_vars = []


def build_ptdf_matrix(system: "EnergySystem", ptdf_lines: list) -> np.ndarray:
    """Compute the Power Transfer Distribution Factor matrix.

    Returns an (n_lines × n_buses) dense matrix mapping nodal net
    injections to line flows. For sparse networks we rebuild it via
    pseudo-inverse of the reduced susceptance matrix; warn above 5 000
    buses (O(N³) dense inverse becomes costly).

    PTDF derivation: removing the slack bus row / column, solve
    B_reduced @ theta = P_inj; then flow = (1/x) * (theta_from - theta_to)
    which collapses to a linear map PTDF @ P_inj.
    """
    n_buses = len(system._buses)
    n_lines = len(ptdf_lines)
    if n_buses > 5000:
        import warnings
        warnings.warn(
            f"PTDF build on {n_buses} buses — dense inverse is O(N³); "
            "consider a sparse / factorised solver.",
            RuntimeWarning, stacklevel=2,
        )

    # Build susceptance matrix B (n_buses × n_buses).
    B = np.zeros((n_buses, n_buses), dtype=float)
    # Incidence matrix A (n_lines × n_buses).
    A = np.zeros((n_lines, n_buses), dtype=float)
    # Line susceptance vector (1 / x).
    y = np.zeros(n_lines, dtype=float)
    for li, link in enumerate(ptdf_lines):
        f = link.bus_from._id
        t = link.bus_to._id
        b = 1.0 / link.reactance
        y[li] = b
        A[li, f] = 1.0
        A[li, t] = -1.0
        B[f, f] += b
        B[t, t] += b
        B[f, t] -= b
        B[t, f] -= b

    # Slack bus 0: drop its row/column, pseudo-invert, reinsert as zero row/col.
    reduced = np.delete(np.delete(B, 0, axis=0), 0, axis=1)
    B_inv_reduced = np.linalg.pinv(reduced)
    B_inv = np.zeros((n_buses, n_buses), dtype=float)
    B_inv[1:, 1:] = B_inv_reduced

    # PTDF = diag(y) @ A @ B_inv.
    ptdf = (A @ B_inv) * y[:, None]
    return ptdf


def build_ptdf_constraints(model, system: "EnergySystem", T: int) -> None:
    """For PTDF lines, add ``flow[line, t] = PTDF[line, :] @ net_inj[:, t]``.

    Net injection at bus b at timestep t is computed from the generator
    dispatch, storage net injection, and load; the same quantities the
    bus-balance constraint already uses. The PTDF block does NOT
    replace the bus-balance (that remains a power-conservation identity)
    — it constrains flows to be consistent with bus injections.
    """
    pt_lines = _ptdf_lines(system)
    if not pt_lines:
        return

    ptdf = build_ptdf_matrix(system, pt_lines)
    n_lines = len(pt_lines)

    for li, link in enumerate(pt_lines):
        cap = link.capacity if link.capacity > 0 else 1e12
        link._flow_signed_vars = []
        for t in range(T):
            fs = model.variable(
                f"fsigned_ptdf_{link.name}_{t}",
                lower=-cap, upper=cap,
            )
            link._flow_signed_vars.append(fs)

    # Add per-line per-timestep equality: f_signed = sum_b PTDF[l, b] * inj[b, t].
    for t in range(T):
        # Build injection expression per bus.
        inj = [None] * len(system._buses)
        for bus in system._buses:
            expr = None
            for gen in system._generators:
                if gen.bus is bus:
                    term = gen._p_vars[t]
                    expr = term if expr is None else expr + term
            for sto in system._storages:
                if sto.bus is bus:
                    term = sto._discharge_vars[t] - sto._charge_vars[t]
                    expr = term if expr is None else expr + term
            for load in system._loads:
                if load.bus is bus:
                    d = load.amount
                    if isinstance(d, np.ndarray):
                        d = float(d[t])
                    if expr is None:
                        expr = -d
                    else:
                        expr = expr - d
            inj[bus._id] = expr

        for li, link in enumerate(pt_lines):
            expr = None
            for b in range(len(system._buses)):
                coef = float(ptdf[li, b])
                if abs(coef) < 1e-12 or inj[b] is None:
                    continue
                term = coef * inj[b]
                expr = term if expr is None else expr + term
            if expr is not None:
                model.add(
                    link._flow_signed_vars[t] == expr,
                    name=f"ptdf_{link.name}_{t}",
                )
        # PTDF vars participate in bus balance via
        # core's bus-balance loop — populated below: since we clear
        # fwd/rev, the balance loop in core uses ``_flow_signed_vars``.
        for link in pt_lines:
            link._flow_vars = []
            link._flow_rev_vars = []


def build_transmission_switching(model, system: "EnergySystem", T: int) -> None:
    """Binary per-line per-timestep on/off; when off, flow must be zero.

    For transport-model lines: just ``fwd + rev ≤ cap * z[t]``.
    For DC-OPF lines: use big-M to relax KVL when z=0.
    """
    big_M = 10.0 * _THETA_BOUND  # radians → divided by x to bound the flow
    for link in system._links:
        if not link.switchable:
            continue
        link._switch_vars = []
        for t in range(T):
            z = model.binary(f"switch_{link.name}_{t}")
            link._switch_vars.append(z)

            if link.model_type in ("dc_opf", "ptdf"):
                cap = link.capacity if link.capacity > 0 else 1e12
                fs = link._flow_signed_vars[t]
                # |fs| <= cap * z.
                model.add(fs <= cap * z, name=f"sw_fs_up_{link.name}_{t}")
                model.add(-fs <= cap * z, name=f"sw_fs_lo_{link.name}_{t}")
                if link.model_type == "dc_opf":
                    # Big-M relaxation of KVL when z=0.
                    theta_f = link.bus_from._theta_vars[t]
                    theta_t = link.bus_to._theta_vars[t]
                    model.add(
                        link.reactance * fs - (theta_f - theta_t)
                        <= big_M * (1 - z),
                        name=f"sw_kvl_up_{link.name}_{t}",
                    )
                    model.add(
                        (theta_f - theta_t) - link.reactance * fs
                        <= big_M * (1 - z),
                        name=f"sw_kvl_lo_{link.name}_{t}",
                    )
            else:
                # Transport: fwd + rev <= cap * z.
                if link._flow_vars:
                    if link.bidirectional and link._flow_rev_vars:
                        model.add(
                            link._flow_vars[t] + link._flow_rev_vars[t]
                            <= link.capacity * z,
                            name=f"sw_trans_bidi_{link.name}_{t}",
                        )
                    else:
                        model.add(
                            link._flow_vars[t] <= link.capacity * z,
                            name=f"sw_trans_fwd_{link.name}_{t}",
                        )


def build_n_minus_1_constraints(model, system: "EnergySystem", T: int) -> None:
    """Flow-based N-1 security via per-contingency signed flows.

    For each line listed in ``system._n_minus_1_lines``, we build a
    replica DC-OPF set (theta^c, f^c) where that line's flow is fenced
    to zero. Line thermal limits apply in the contingency state:
    ``|f^c[l, t]| ≤ cap[l]`` for every l != c.

    This is the *preventive* N-1 formulation — generator dispatch is
    the same in base and contingency states. Generation reserve to
    cover contingency ramp is a separate reserves feature (Phase 2).
    """
    contingencies = getattr(system, "_n_minus_1_lines", None)
    if not contingencies:
        return
    dc_lines = _dc_opf_lines(system)
    if not dc_lines:
        return  # N-1 requires DC-OPF physics; transport model has no flow redistribution

    # Map line_name -> Link for O(1) lookup.
    line_by_name = {l.name: l for l in dc_lines}
    n_buses = len(system._buses)

    for c_name in contingencies:
        c_link = line_by_name.get(c_name)
        if c_link is None:
            continue  # user named a non-DC-OPF line; silently skip

        # Create per-contingency theta and f variables.
        theta_c = {}  # bus._id -> [T] variables
        for bus in system._buses:
            theta_c[bus._id] = []
            for t in range(T):
                v = model.variable(
                    f"theta_c_{c_name}_{bus.name}_{t}",
                    lower=-_THETA_BOUND, upper=_THETA_BOUND,
                )
                theta_c[bus._id].append(v)
            if bus._id == 0:  # slack
                for t in range(T):
                    model.add(theta_c[bus._id][t] == 0.0,
                              name=f"slack_c_{c_name}_{bus.name}_{t}")

        # Per-contingency signed flows for every non-contingent line;
        # fenced flow on the contingent line itself.
        f_c = {}  # link.name -> [T] variables
        for link in dc_lines:
            cap = link.capacity if link.capacity > 0 else 1e12
            if link is c_link:
                # Fence the tripped line to zero flow.
                f_c[link.name] = None
                continue
            f_c[link.name] = []
            for t in range(T):
                fs = model.variable(
                    f"fc_{c_name}_{link.name}_{t}",
                    lower=-cap, upper=cap,
                )
                f_c[link.name].append(fs)
                th_f = theta_c[link.bus_from._id][t]
                th_t = theta_c[link.bus_to._id][t]
                model.add(link.reactance * fs == th_f - th_t,
                          name=f"kvl_c_{c_name}_{link.name}_{t}")

        # Per-bus power-balance in the contingency state: net injection
        # from generators/storage/loads (same as base case) must equal
        # net flow outgoing on the non-contingent lines. Because dispatch
        # is preventive (unchanged), we equate injection to flow:
        #     sum_{l incident} sign(l, bus) * f^c[l, t]
        #         = gen_inj(bus, t) - load(bus, t)
        # Written at every non-slack bus for every t.
        for t in range(T):
            for bus in system._buses:
                # Build RHS: generators - loads - storage_net (storage
                # dispatch is preventive too).
                rhs = None
                for gen in system._generators:
                    if gen.bus is bus:
                        term = gen._p_vars[t]
                        rhs = term if rhs is None else rhs + term
                for sto in system._storages:
                    if sto.bus is bus:
                        term = sto._discharge_vars[t] - sto._charge_vars[t]
                        rhs = term if rhs is None else rhs + term
                for load in system._loads:
                    if load.bus is bus:
                        d = load.amount
                        if isinstance(d, np.ndarray):
                            d = float(d[t])
                        if rhs is None:
                            rhs = -d
                        else:
                            rhs = rhs - d
                # LHS: net outgoing flow on all non-contingent DC-OPF lines.
                lhs = None
                for link in dc_lines:
                    fs_list = f_c[link.name]
                    if fs_list is None:
                        continue  # tripped line
                    if link.bus_from is bus:
                        term = fs_list[t]
                        lhs = term if lhs is None else lhs + term
                    elif link.bus_to is bus:
                        term = -fs_list[t]
                        lhs = term if lhs is None else lhs + term
                if lhs is not None and rhs is not None:
                    model.add(lhs == rhs,
                              name=f"bal_c_{c_name}_{bus.name}_{t}")
