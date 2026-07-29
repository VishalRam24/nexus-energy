"""
N_En_Phase 17.3 — polar AC-OPF formulation.

First-class NLP build of the classical polar power-flow equations on top of
``nexus.Model`` (via `nexus-opt`'s first-class nonlinear expression support,
N_Opt_Phase 5.2). Intended as a parity check against pandapower PIPS and
as a template for users who want to write their own transcendental-heavy
energy models.

Scope (mirrors the SOCP relaxation scope):

- **Single snapshot.** Multi-period deferred.
- **Dispatch only.** No UC / capacity expansion.
- **Constant-power loads** (with optional ``load.q_amount``, Mvar).
- **Linear + quadratic generator cost** (``generator.marginal_cost``,
  ``generator.quadratic_cost``).
- **π-line model with transformer turns ratio** (``link.tap``,
  ``link.shift``) and shunts (``link.g_fr/b_fr/g_to/b_to``,
  ``bus.g_shunt/b_shunt``).

Activation: set ``link.model_type`` to ``"ac_opf_polar"`` (or reuse a
``"socp_opf"``-configured system as-is). Then::

    from nexus_energy import solve_ac_opf_polar
    res = solve_ac_opf_polar(system, snapshot=0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import nexus_opt as nx

if TYPE_CHECKING:
    from nexus_energy.core import EnergySystem


@dataclass
class ACOpfPolarResult:
    """Result of :func:`solve_ac_opf_polar`."""
    status: str
    total_cost: float
    voltage_mag: dict[str, float]    # bus.name -> |V| (pu)
    voltage_angle: dict[str, float]  # bus.name -> θ (rad)
    gen_p: dict[str, float]          # gen.name -> P (MW)
    gen_q: dict[str, float]          # gen.name -> Q (Mvar)
    branch_p: dict[str, float]       # link.name -> P_sending (MW)
    branch_q: dict[str, float]       # link.name -> Q_sending (Mvar)
    branch_loss: dict[str, float]    # link.name -> P_loss (MW)
    solve_time: float
    iterations: int


def _polar_opf_lines(system: "EnergySystem") -> list:
    """Return the branches participating in the polar AC-OPF.

    Accepts both the native ``"ac_opf_polar"`` tag and the
    ``"socp_opf"`` tag so an existing SOCP-configured test system round-
    trips through the polar solver as a parity check.
    """
    return [l for l in system._links
            if getattr(l, "model_type", "") in ("ac_opf_polar", "socp_opf")]


def solve_ac_opf_polar(system: "EnergySystem", *, snapshot: int = 0,
                       slack_bus: str | None = None,
                       verbose: bool = False) -> ACOpfPolarResult:
    """Solve the polar AC-OPF for a single snapshot via IPOPT.

    Uses ``nexus_opt.sin`` / ``cos`` to assemble the P/Q balance equations
    directly; the underlying model goes through :mod:`nexus-opt`'s NLP
    dispatch to CasADi+IPOPT.

    ``slack_bus`` selects the reference bus whose angle is pinned to 0.
    Defaults to the first bus in ``system._buses``.
    """
    def _scalar(x):
        if isinstance(x, np.ndarray):
            return float(x[snapshot])
        return float(x)

    buses = list(system._buses)
    if not buses:
        raise ValueError("solve_ac_opf_polar: system has no buses")
    lines = _polar_opf_lines(system)
    if not lines:
        raise ValueError(
            "solve_ac_opf_polar: no Links with model_type='ac_opf_polar' "
            "(or 'socp_opf') present.")

    slack = slack_bus if slack_bus is not None else buses[0].name
    if slack not in {b.name for b in buses}:
        raise ValueError(f"slack_bus {slack!r} not in system buses")

    gens_by_bus: dict[str, list] = {b.name: [] for b in buses}
    for g in system._generators:
        gens_by_bus[g.bus.name].append(g)
    loads_by_bus: dict[str, list] = {b.name: [] for b in buses}
    for ld in system._loads:
        loads_by_bus[ld.bus.name].append(ld)

    m = nx.Model("ac_opf_polar")

    # --- variables --------------------------------------------------------
    V: dict[str, object] = {}
    theta: dict[str, object] = {}
    for b in buses:
        v_min = float(getattr(b, "v_min", 0.95))
        v_max = float(getattr(b, "v_max", 1.05))
        V[b.name] = m.variable(f"V_{b.name}", lower=v_min, upper=v_max)
        if b.name == slack:
            # Pin slack angle to 0 via a zero-width box.
            theta[b.name] = m.variable(f"theta_{b.name}", lower=0.0, upper=0.0)
        else:
            theta[b.name] = m.variable(
                f"theta_{b.name}", lower=-math.pi, upper=math.pi)

    p_gen: dict[str, object] = {}
    q_gen: dict[str, object] = {}
    for g in system._generators:
        p_max = float(g.capacity)
        if g.carrier_factor is not None:
            p_max *= _scalar(g.carrier_factor)
        p_min = float(getattr(g, "p_min", 0.0))
        q_min = float(getattr(g, "q_min", -0.5 * g.capacity))
        q_max = float(getattr(g, "q_max",  0.5 * g.capacity))
        p_gen[g.name] = m.variable(f"p_{g.name}", lower=p_min, upper=p_max)
        q_gen[g.name] = m.variable(f"q_{g.name}", lower=q_min, upper=q_max)

    # Branch flow variables (free, box-limited by thermal limit elsewhere).
    P_ij: dict[str, object] = {}
    Q_ij: dict[str, object] = {}
    P_ji: dict[str, object] = {}
    Q_ji: dict[str, object] = {}
    for l in lines:
        s_max = float(getattr(l, "s_max", l.capacity))
        P_ij[l.name] = m.variable(f"P_{l.name}", lower=-s_max, upper=s_max)
        Q_ij[l.name] = m.variable(f"Q_{l.name}", lower=-s_max, upper=s_max)
        P_ji[l.name] = m.variable(f"Pji_{l.name}", lower=-s_max, upper=s_max)
        Q_ji[l.name] = m.variable(f"Qji_{l.name}", lower=-s_max, upper=s_max)

    # --- objective --------------------------------------------------------
    obj = 0.0 * V[buses[0].name]  # seed as an Expr, not a Python float
    for g in system._generators:
        mc = float(g.marginal_cost)
        if mc != 0.0:
            obj = obj + mc * p_gen[g.name]
        q2 = float(getattr(g, "quadratic_cost", 0.0))
        if q2 != 0.0:
            if q2 < 0.0:
                raise ValueError(
                    f"generator {g.name!r}: quadratic_cost must be ≥ 0; got {q2}")
            obj = obj + q2 * (p_gen[g.name] * p_gen[g.name])
    m.minimize(obj)

    # --- per-branch flow equations ----------------------------------------
    for l in lines:
        if l.reactance <= 0.0:
            raise ValueError(
                f"link {l.name!r}: model_type='ac_opf_polar' requires reactance > 0")
        r = float(getattr(l, "resistance", 0.0))
        x = float(l.reactance)
        denom = r * r + x * x
        g_se = r / denom
        b_se = -x / denom

        g_fr = float(getattr(l, "g_fr", 0.0))
        b_fr = float(getattr(l, "b_fr", 0.0))
        g_to = float(getattr(l, "g_to", 0.0))
        b_to = float(getattr(l, "b_to", 0.0))
        tap = float(getattr(l, "tap", 1.0))
        if tap <= 0.0:
            raise ValueError(f"link {l.name!r}: tap must be > 0 (got {tap})")
        phi = float(getattr(l, "shift", 0.0))
        inv_tap = 1.0 / tap
        inv_tap2 = inv_tap * inv_tap
        cos_phi = float(np.cos(phi))
        sin_phi = float(np.sin(phi))
        g_ff = (g_se + g_fr) * inv_tap2
        b_ff = (b_se + b_fr) * inv_tap2
        g_tt = g_se + g_to
        b_tt = b_se + b_to
        rft = (b_se * sin_phi - g_se * cos_phi) * inv_tap
        ift = -(g_se * sin_phi + b_se * cos_phi) * inv_tap
        rtf = -(g_se * cos_phi + b_se * sin_phi) * inv_tap
        itf = (g_se * sin_phi - b_se * cos_phi) * inv_tap

        V_i = V[l.bus_from.name]
        V_j = V[l.bus_to.name]
        th_i = theta[l.bus_from.name]
        th_j = theta[l.bus_to.name]
        dth = th_i - th_j
        # c_ij = V_i · V_j · cos(θ_i - θ_j); s_ij = V_i · V_j · sin(θ_i - θ_j)
        c_ij = V_i * V_j * nx.cos(dth)
        s_ij = V_i * V_j * nx.sin(dth)
        vi2 = V_i * V_i
        vj2 = V_j * V_j

        # P_ij =  g_ff·V_i² + rft·c_ij + ift·s_ij
        m.add(P_ij[l.name] - g_ff * vi2 - rft * c_ij - ift * s_ij == 0.0)
        # Q_ij = -b_ff·V_i² + rft·s_ij - ift·c_ij
        m.add(Q_ij[l.name] + b_ff * vi2 - rft * s_ij + ift * c_ij == 0.0)
        # P_ji =  g_tt·V_j² + rtf·c_ij - itf·s_ij
        m.add(P_ji[l.name] - g_tt * vj2 - rtf * c_ij + itf * s_ij == 0.0)
        # Q_ji = -b_tt·V_j² - itf·c_ij - rtf·s_ij
        m.add(Q_ji[l.name] + b_tt * vj2 + itf * c_ij + rtf * s_ij == 0.0)

        # Thermal limits: P² + Q² ≤ s_max²
        s_max = float(getattr(l, "s_max", l.capacity))
        m.add(P_ij[l.name] * P_ij[l.name] + Q_ij[l.name] * Q_ij[l.name]
              <= s_max * s_max)
        m.add(P_ji[l.name] * P_ji[l.name] + Q_ji[l.name] * Q_ji[l.name]
              <= s_max * s_max)

    # --- bus power balance ------------------------------------------------
    for b in buses:
        p_load = sum(_scalar(ld.amount) for ld in loads_by_bus[b.name])
        q_load = sum(float(getattr(ld, "q_amount", 0.0))
                     for ld in loads_by_bus[b.name])
        g_sh = float(getattr(b, "g_shunt", 0.0))
        b_sh = float(getattr(b, "b_shunt", 0.0))
        V_i = V[b.name]

        p_expr = 0.0 * V_i
        q_expr = 0.0 * V_i
        for g in gens_by_bus[b.name]:
            p_expr = p_expr + p_gen[g.name]
            q_expr = q_expr + q_gen[g.name]
        for l in lines:
            if l.bus_from.name == b.name:
                p_expr = p_expr - P_ij[l.name]
                q_expr = q_expr - Q_ij[l.name]
            if l.bus_to.name == b.name:
                p_expr = p_expr - P_ji[l.name]
                q_expr = q_expr - Q_ji[l.name]
        if g_sh != 0.0:
            p_expr = p_expr - g_sh * (V_i * V_i)
        if b_sh != 0.0:
            q_expr = q_expr + b_sh * (V_i * V_i)

        m.add(p_expr == p_load)
        m.add(q_expr == q_load)

    # --- solve ------------------------------------------------------------
    result = m.solve(solver="ipopt", verbose=verbose)

    voltage_mag: dict[str, float] = {}
    voltage_angle: dict[str, float] = {}
    gen_p_out: dict[str, float] = {}
    gen_q_out: dict[str, float] = {}
    branch_p: dict[str, float] = {}
    branch_q: dict[str, float] = {}
    branch_loss: dict[str, float] = {}
    if result.status == "optimal":
        values = result.values()
        for b in buses:
            voltage_mag[b.name] = float(values[f"V_{b.name}"])
            voltage_angle[b.name] = float(values[f"theta_{b.name}"])
        for g in system._generators:
            gen_p_out[g.name] = float(values[f"p_{g.name}"])
            gen_q_out[g.name] = float(values[f"q_{g.name}"])
        for l in lines:
            branch_p[l.name] = float(values[f"P_{l.name}"])
            branch_q[l.name] = float(values[f"Q_{l.name}"])
            branch_loss[l.name] = float(
                values[f"P_{l.name}"] + values[f"Pji_{l.name}"])

    return ACOpfPolarResult(
        status=result.status,
        total_cost=float(result.objective) if result.objective is not None else float("nan"),
        voltage_mag=voltage_mag,
        voltage_angle=voltage_angle,
        gen_p=gen_p_out,
        gen_q=gen_q_out,
        branch_p=branch_p,
        branch_q=branch_q,
        branch_loss=branch_loss,
        solve_time=result.solve_time,
        iterations=result.iterations if result.iterations is not None else 0,
    )
