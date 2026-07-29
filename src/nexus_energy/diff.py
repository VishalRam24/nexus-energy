"""
Phase 12 — differentiable dispatch.

Exposes gradients of a single-period economic-dispatch solution with
respect to its three differentiable inputs: marginal costs, capacity
bounds, and total demand. The immediate motivation is *parameter
learning* — fitting demand elasticity, storage bid curves, or
market-maker parameters from observed dispatch data — which falls out
of implicit differentiation at the KKT optimum.

Three layers of API are available, selected by how much you need:

- :func:`solve_dispatch_with_sensitivities` — pure-numpy, closed-form
  solver for single-period single-bus dispatch with a small strictly-
  convex ridge. Returns ``(p, dp_dmc, dp_dcap, dp_ddemand)``. No
  optimiser call; the KKT system is small enough to solve in-head.
- :class:`EconomicDispatchLayer` — stateful OO wrapper that caches the
  last forward pass so gradients can be pulled on demand. Composes
  with a user-provided loss via ``backward(grad_out)``.
- :class:`TorchDispatchLayer` — torch-optional ``cvxpylayers`` hook
  for problems that exceed the pure-numpy path (multi-bus, multi-
  period, storage). Not the default; imports cleanly without torch.

The honest scope here is **parameter learning on the inner LP**, not
end-to-end differentiable capacity expansion (that would require
bilevel-MIP gradients, which remain research — see
``DIFFERENTIATORS.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


__all__ = [
    "EconomicDispatchLayer",
    "TorchDispatchLayer",
    "DispatchJacobian",
    "solve_dispatch_with_sensitivities",
    "numerical_jacobian",
    "torch_available",
    # Phase 12.1 — multi-bus multi-period differentiable dispatch.
    "MultiBusDispatchProblem",
    "MultiBusDispatchSolution",
    "MultiBusDispatchLayer",
    "solve_multibus_dispatch_with_sensitivities",
    # Phase 12.3 — demand-elasticity recovery example.
    "ElasticityFitResult",
    "fit_demand_elasticity",
    # Phase 12.2 — differentiable multi-period storage + smoothed commitment.
    "StorageDispatchProblem",
    "StorageDispatchSolution",
    "StorageDispatchLayer",
    "solve_storage_dispatch_with_sensitivities",
    "SmoothCommitmentLayer",
    "smooth_commitment",
    "fit_commitment_threshold",
    "CommitmentFitResult",
    # Phase 20 — differentiable capacity expansion (design-variable grads).
    "CapacityExpansionProblem",
    "CapacityExpansionSolution",
    "CapacityExpansionLayer",
    "solve_capacity_expansion_with_sensitivities",
    "fit_component_params",
    "ComponentFitResult",
]


# ---------------------------------------------------------------------------
# Torch-optional probe
# ---------------------------------------------------------------------------

try:  # pragma: no cover — probe at import time.
    import torch  # type: ignore  # noqa: F401
    torch_available = True
except Exception:  # pragma: no cover — torch missing is the default.
    torch_available = False


# ---------------------------------------------------------------------------
# Pure-numpy closed-form path
# ---------------------------------------------------------------------------

@dataclass
class DispatchJacobian:
    """Partial derivatives of dispatch wrt its inputs.

    All arrays have shape ``(n_gen, ...)``. ``dp_ddemand`` is 1-D since
    demand is a scalar in this single-bus formulation; the others are
    square (or rectangular, for ``dp_dmc``).
    """
    dp_dmc: np.ndarray
    dp_dcap: np.ndarray
    dp_ddemand: np.ndarray


def solve_dispatch_with_sensitivities(
    marginal_cost: np.ndarray,
    capacity: np.ndarray,
    demand: float,
    ridge: float = 1e-3,
) -> tuple[np.ndarray, DispatchJacobian]:
    """Single-period, single-bus economic dispatch with ridge regulariser.

    Solves the QP::

        min  (ridge/2) Σ pᵢ² + Σ cᵢ pᵢ
        s.t. Σ pᵢ = d,  0 ≤ pᵢ ≤ capᵢ

    The small ridge (default ``1e-3``) is mandatory — a pure LP is
    piecewise constant in ``c`` at non-degenerate optima, so its
    gradient is zero almost everywhere and undefined at the boundary.
    Adding ``(ridge/2)||p||²`` restores strict convexity without
    materially distorting the economic-dispatch answer for dispatch
    parameters in the MW range.

    Returns ``(p, jacobian)`` where ``jacobian`` carries the three
    partials. The gradients are exact (not finite-differenced); the
    active-set KKT system is solved analytically after clamping each
    generator to its bounds.
    """
    c = np.asarray(marginal_cost, dtype=float).copy()
    cap = np.asarray(capacity, dtype=float).copy()
    n = c.shape[0]
    if cap.shape != c.shape:
        raise ValueError("marginal_cost and capacity must have the same shape")
    if ridge <= 0:
        raise ValueError("ridge must be > 0; use the LP solver for ridge=0")
    if demand < 0:
        raise ValueError("demand must be non-negative")
    total_cap = float(np.sum(cap))
    if demand > total_cap + 1e-9:
        raise ValueError(
            f"infeasible: demand={demand:g} exceeds total capacity={total_cap:g}")

    # ---- Forward ----
    # Active-set iteration: start with all unconstrained ("free"), then
    # pin gens whose unconstrained optimum violates a bound, and repeat.
    # At most n outer iterations; in practice 1–3.
    free = np.ones(n, dtype=bool)
    at_zero = np.zeros(n, dtype=bool)
    at_cap = np.zeros(n, dtype=bool)
    p = np.zeros(n, dtype=float)

    for _ in range(n + 2):
        # Closed-form solution on current free set.
        free_idx = np.where(free)[0]
        fixed = np.zeros(n, dtype=float)
        fixed[at_cap] = cap[at_cap]
        residual = demand - fixed.sum()

        if free_idx.size == 0:
            # All gens pinned → feasibility check only.
            p = fixed
            if abs(residual) > 1e-6:
                raise ValueError(
                    "infeasible active set — try loosening the ridge")
            lam = 0.0
            break

        cf = c[free_idx]
        k = free_idx.size
        # From stationarity: ridge * p_i + c_i + lam = 0 for free i.
        # Sum constraint on free set: Σ p_i = residual.
        # => Σ (-c_i - lam)/ridge = residual
        # => lam = -(ridge * residual + Σ c_i) / k
        lam = -(ridge * residual + cf.sum()) / k
        p_free = (-cf - lam) / ridge
        p = np.zeros(n, dtype=float)
        p[at_cap] = cap[at_cap]
        p[free_idx] = p_free

        # Classify violations and re-pin.
        below = (p < -1e-12) & free
        above = (p > cap + 1e-12) & free
        if not below.any() and not above.any():
            at_zero = at_zero | ((p <= 1e-12) & free & (p <= 0 + 1e-12) & False)
            break
        # Pin the most violated first to keep the active set monotone.
        if below.any():
            idx = int(np.argmax((-p) * below))
            free[idx] = False
            at_zero[idx] = True
            p[idx] = 0.0
        if above.any():
            idx = int(np.argmax((p - cap) * above))
            free[idx] = False
            at_cap[idx] = True
            p[idx] = cap[idx]

    # Snap near-zero floats.
    p = np.clip(p, 0.0, cap)

    # ---- Sensitivities on the (now frozen) active set ----
    # Only gens in the free set move infinitesimally. Pinned gens
    # contribute zero to dp_dmc / dp_ddemand (they stay at their bound)
    # but DO contribute to dp_dcap for their own entry on the diagonal.
    free_idx = np.where(free)[0]
    k = free_idx.size
    dp_dmc = np.zeros((n, n), dtype=float)
    dp_dcap = np.zeros((n, n), dtype=float)
    dp_ddemand = np.zeros(n, dtype=float)

    if k > 0:
        # ∂p_i/∂c_j for i,j in free: (δ_ij − 1/k) / ridge
        # Derivation: lam = -(ridge*residual + Σ c_j)/k → ∂lam/∂c_j = -1/k
        # Then ∂p_i/∂c_j = (-δ_ij − ∂lam/∂c_j)/ridge = (1/k − δ_ij)/ridge.
        eye_free = np.eye(k)
        block = (eye_free - 1.0 / k) / ridge
        # We want ∂p_i/∂c_j = -(δ_ij − 1/k)/ridge — the NEGATIVE of the
        # naive formula above because stationarity is ridge*p + c + lam = 0
        # (cost ↑ → dispatch ↓). Sign check: at k=1, interior gen, ∂p/∂c
        # should equal 0 (only one free gen must satisfy p=demand). The
        # expression (1 − 1/1)/ridge = 0 — correct.
        # For k=2 symmetric gens, ∂p_i/∂c_i = -(1 - 1/2)/ridge < 0 — correct.
        dp_dmc[np.ix_(free_idx, free_idx)] = -block

        # ∂p_i/∂demand for i in free: 1/k (equal split among free gens).
        dp_ddemand[free_idx] = 1.0 / k

    # ∂p_i/∂cap_j:
    # • If j is at_cap (p_j = cap_j): ∂p_j/∂cap_j = 1 and the freed
    #   residual is absorbed equally by the free gens with coeff -1/k.
    # • If j is free or at_zero: ∂p_i/∂cap_j = 0 (bound is not active).
    for j in np.where(at_cap)[0]:
        dp_dcap[j, j] = 1.0
        if k > 0:
            dp_dcap[free_idx, j] = -1.0 / k

    jacobian = DispatchJacobian(
        dp_dmc=dp_dmc,
        dp_dcap=dp_dcap,
        dp_ddemand=dp_ddemand,
    )
    return p, jacobian


def numerical_jacobian(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central-difference Jacobian of ``fn`` at ``x``. Verification only."""
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(fn(x), dtype=float)
    jac = np.zeros((y0.size, x.size))
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        jac[:, i] = (fn(xp) - fn(xm)) / (2 * eps)
    return jac


# ---------------------------------------------------------------------------
# Stateful layer
# ---------------------------------------------------------------------------

@dataclass
class EconomicDispatchLayer:
    """Thin OO wrapper around :func:`solve_dispatch_with_sensitivities`.

    Calls to :meth:`forward` cache the active set; :meth:`backward`
    returns gradients of a scalar loss w.r.t. the three inputs without
    re-running the forward pass. Mirrors the PyTorch autograd style
    while staying torch-free — easy to drop into a hand-written SGD
    loop for the parameter-learning workflow.
    """
    ridge: float = 1e-3
    _p: np.ndarray | None = field(default=None, init=False, repr=False)
    _jac: DispatchJacobian | None = field(default=None, init=False, repr=False)

    def forward(
        self,
        marginal_cost: np.ndarray,
        capacity: np.ndarray,
        demand: float,
    ) -> np.ndarray:
        p, jac = solve_dispatch_with_sensitivities(
            marginal_cost=marginal_cost,
            capacity=capacity,
            demand=float(demand),
            ridge=self.ridge,
        )
        self._p = p
        self._jac = jac
        return p

    def backward(
        self,
        grad_out: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        """Return ``dL/d{mc, cap, demand}`` given ``dL/dp``."""
        if self._jac is None:
            raise RuntimeError("call forward() before backward()")
        g = np.asarray(grad_out, dtype=float)
        jac = self._jac
        return {
            "marginal_cost": jac.dp_dmc.T @ g,
            "capacity": jac.dp_dcap.T @ g,
            "demand": float(jac.dp_ddemand @ g),
        }


# ---------------------------------------------------------------------------
# Phase 12.1 — multi-bus, multi-period differentiable dispatch
# ---------------------------------------------------------------------------
#
# We solve the strictly-convex transport-model dispatch QP
#
#     min_p  Σ_t Σ_g  mc_g · p_{g,t} + (ridge/2) · p_{g,t}²
#     s.t.   (balance, per bus b, period t)
#            Σ_{g∈b} p_{g,t} + Σ_l A_{b,l} f_{l,t} = d_{b,t}
#            0 ≤ p_{g,t} ≤ cap_g                       (gen bounds)
#            -F_l ≤ f_{l,t} ≤ F_l                      (line bounds)
#
# where ``f`` are line flows and ``A`` is the bus–line incidence
# (+1 at the "to" bus, -1 at the "from" bus). Stacking p and f into one
# decision vector x per period, this is a standard equality+box QP:
#
#     min  ½ xᵀ H x + qᵀ x   s.t.  C x = d,  lb ≤ x ≤ ub.
#
# H = ridge·I on the generator block, ridge·I (small) on the flow block
# to keep the system strictly convex and the active-set KKT matrix
# invertible. Periods are coupled only through shared parameters here
# (no inter-temporal storage), so each period is an independent QP — but
# we keep the multi-period vectorisation explicit because the *gradients*
# w.r.t. shared parameters (mc_g, cap_g, F_l) accumulate across periods,
# which is exactly what a learning loop needs.
#
# Gradients come from differentiating the KKT system of the QP on its
# *active set* (the implicit-function theorem applied at the optimum):
# Amos & Kolter, "OptNet: Differentiable Optimization as a Layer in
# Neural Networks", ICML 2017; Agrawal et al., "Differentiable Convex
# Optimization Layers" (cvxpylayers), NeurIPS 2019. With the active
# inequalities pinned to equalities, the optimum satisfies a linear KKT
# system; differentiating both sides gives dx/dθ in closed form. The
# ridge guarantees H ≻ 0 so the reduced KKT matrix is nonsingular and
# the LP-degeneracy (piecewise-constant, a.e.-zero gradient) is avoided.


@dataclass
class MultiBusDispatchProblem:
    """A tiny multi-bus, multi-period transport-dispatch instance.

    Generators sit on buses; lines connect bus pairs with a symmetric
    flow limit. Demand is per-bus, per-period. The differentiable
    parameters are ``marginal_cost`` (per gen), ``capacity`` (per gen)
    and ``line_limit`` (per line); ``demand`` is also differentiable.

    Attributes:
        gen_bus: ``(G,)`` int — bus index of each generator.
        marginal_cost: ``(G,)`` — linear cost per unit dispatch.
        capacity: ``(G,)`` — upper bound on each generator's dispatch.
        line_from / line_to: ``(L,)`` int — endpoints of each line.
        line_limit: ``(L,)`` — symmetric flow limit (``|f| ≤ limit``).
        demand: ``(B, T)`` — per-bus per-period demand.
        n_buses: number of buses ``B``.
        ridge: strict-convexity regulariser (must be > 0).
        availability: optional ``(G, T)`` in [0, 1] — per-period derating of
            the upper bound: ``p[g,t] ≤ capacity[g] · availability[g,t]``
            (VRE capacity factors / outages). ``None`` ⇒ all ones. The
            ``d_dispatch_d_capacity`` Jacobian stays w.r.t. nameplate
            ``capacity[g]`` (the availability factor is chained in).
        line_min: optional ``(L,)`` lower flow bound. ``None`` (default)
            keeps the symmetric ``-line_limit``; pass ``0.0`` entries for
            unidirectional links. When given, ``d_dispatch_d_linelimit``
            differentiates only the upper bound (``line_min`` is treated
            as an independent constant).
    """
    gen_bus: np.ndarray
    marginal_cost: np.ndarray
    capacity: np.ndarray
    line_from: np.ndarray
    line_to: np.ndarray
    line_limit: np.ndarray
    demand: np.ndarray
    n_buses: int
    ridge: float = 1e-2
    availability: Optional[np.ndarray] = None
    line_min: Optional[np.ndarray] = None


@dataclass
class MultiBusDispatchSolution:
    """Forward solution plus analytic Jacobians of dispatch.

    ``dispatch`` is ``(G, T)``; ``flows`` is ``(L, T)``. The Jacobians
    are flattened in row-major ``(g, t)`` order over the ``G·T`` dispatch
    outputs:

        d_dispatch_d_mc:        (G·T, G)
        d_dispatch_d_capacity:  (G·T, G)
        d_dispatch_d_demand:    (G·T, B·T)   (demand flattened (b, t))
        d_dispatch_d_linelimit: (G·T, L)
    """
    dispatch: np.ndarray
    flows: np.ndarray
    d_dispatch_d_mc: np.ndarray
    d_dispatch_d_capacity: np.ndarray
    d_dispatch_d_demand: np.ndarray
    d_dispatch_d_linelimit: np.ndarray


def _solve_period_qp(
    H: np.ndarray,
    q: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-9,
    pin_degenerate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Active-set solve of  min ½xᵀHx+qᵀx s.t. Cx=d, lb≤x≤ub.

    Returns ``(x, active)`` where ``active`` is an int array: 0 = free,
    +1 = pinned at upper bound, -1 = pinned at lower bound. ``H`` must be
    SPD (the ridge guarantees this). Small problem ⇒ dense KKT solve
    each iteration; converges in a handful of active-set swaps.

    ``pin_degenerate`` permanently pins variables whose bounds coincide
    (``lb == ub``, e.g. a VRE hour with zero availability) so the
    release test cannot cycle on them. Off by default: pinning can zero
    out equality rows (singular reduced KKT) in problems whose
    constraints touch only degenerate variables — those rely on the
    ridge keeping such vars "free" at their bound instead.
    """
    n = H.shape[0]
    m = C.shape[0]
    active = np.zeros(n, dtype=int)  # 0 free, +1 at ub, -1 at lb
    frozen = ((ub - lb) <= 1e-12 if pin_degenerate
              else np.zeros(n, dtype=bool))
    active[frozen] = -1

    for _ in range(max_iter):
        free = active == 0
        fixed_val = np.where(active > 0, ub, np.where(active < 0, lb, 0.0))
        # Reduced KKT on the free block:
        #   [H_ff  C_fᵀ][x_f ]   [-q_f - H_fx x_x]
        #   [C_f   0   ][ λ  ] = [ d   - C_x x_x ]
        idx_f = np.where(free)[0]
        idx_x = np.where(~free)[0]
        kf = idx_f.size
        Hff = H[np.ix_(idx_f, idx_f)]
        Cf = C[:, idx_f]
        rhs_top = -q[idx_f]
        rhs_bot = d.copy()
        if idx_x.size:
            xx = fixed_val[idx_x]
            rhs_top = rhs_top - H[np.ix_(idx_f, idx_x)] @ xx
            rhs_bot = rhs_bot - C[:, idx_x] @ xx
        KKT = np.zeros((kf + m, kf + m))
        KKT[:kf, :kf] = Hff
        KKT[:kf, kf:] = Cf.T
        KKT[kf:, :kf] = Cf
        sol = np.linalg.solve(KKT, np.concatenate([rhs_top, rhs_bot]))
        x_free = sol[:kf]
        x = fixed_val.copy()
        x[idx_f] = x_free

        # Primal feasibility of the free vars: any bound violated → pin it.
        viol_up = (x[idx_f] > ub[idx_f] + tol)
        viol_lo = (x[idx_f] < lb[idx_f] - tol)
        if viol_up.any() or viol_lo.any():
            # Pin the single most-violated free variable.
            over = np.maximum(x[idx_f] - ub[idx_f], lb[idx_f] - x[idx_f])
            j = idx_f[int(np.argmax(over))]
            if x[j] > ub[j]:
                active[j] = 1
                x[j] = ub[j]
            else:
                active[j] = -1
                x[j] = lb[j]
            continue

        # Dual feasibility: a pinned var with the wrong-signed reduced
        # cost should be released. Reduced gradient r = Hx + q + Cᵀλ.
        lam = sol[kf:]
        r = H @ x + q + C.T @ lam
        released = False
        for j in idx_x:
            if frozen[j]:
                continue
            # At ub, KKT-optimal requires r_j ≤ 0; at lb requires r_j ≥ 0.
            if active[j] > 0 and r[j] > tol:
                active[j] = 0
                released = True
                break
            if active[j] < 0 and r[j] < -tol:
                active[j] = 0
                released = True
                break
        if not released:
            return x, active
    return x, active


def _solve_period_qp_dual(
    h: np.ndarray,
    q: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    lam0: Optional[np.ndarray] = None,
    max_iter: int = 200,
    tol: float = 1e-11,
    strict: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dual semismooth-Newton solve of  min ½xᵀdiag(h)x+qᵀx, Cx=d, lb≤x≤ub.

    For diagonal SPD ``h`` the primal minimiser given multipliers λ is the
    closed-form clip ``x(λ) = clip((Cᵀλ − q)/h, lb, ub)``; the dual residual
    ``r(λ) = Cx(λ) − d`` is piecewise-linear, so a damped semismooth Newton
    on λ (generalised Jacobian ``C_F diag(1/h_F) C_Fᵀ`` over the free set F)
    converges robustly regardless of cost/ridge scaling — unlike the greedy
    primal active-set path, which can over-pin its way into a rank-deficient
    reduced KKT on realistically-scaled data (small ridge, MW-scale bounds).

    Returns ``(x, active, λ)`` with ``active`` coded like
    :func:`_solve_period_qp` (0 free, ±1 at bound; degenerate lb==ub pinned).
    """
    B = C.shape[0]
    lam = np.zeros(B) if lam0 is None else lam0.astype(float).copy()
    scale = max(1.0, float(np.abs(d).max()))
    x = np.clip((C.T @ lam - q) / h, lb, ub)
    for _ in range(max_iter):
        u = (C.T @ lam - q) / h
        x = np.clip(u, lb, ub)
        r = C @ x - d
        if float(np.max(np.abs(r))) < tol * scale:
            break
        # Generalized-Jacobian branch at exact kinks (u == bound) is
        # ambiguous, and either branch can be the right one — try the
        # inclusive branch first, the strict branch on stall.
        r0 = float(np.linalg.norm(r))
        moved = False
        for incl in (True, False):
            free = ((u >= lb) & (u <= ub)) if incl else ((u > lb) & (u < ub))
            Jf = C[:, free]
            J = (Jf * (1.0 / h[free])) @ Jf.T
            # Levenberg damping keeps the step defined when a row has no
            # free incident variable (transiently fully-pinned).
            J[np.diag_indices(B)] += 1e-10 * (1.0 + np.trace(J) / B)
            try:
                dlam = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dlam = np.linalg.lstsq(J, -r, rcond=None)[0]
            # EXACT line search along the ray: u(λ+t·dlam) is linear in t,
            # so r(t) is piecewise-AFFINE with kinks where any u_j crosses
            # a bound. Evaluate r at all breakpoints, then the closed-form
            # interior minimiser of ‖r‖ within every segment (the norm of
            # an affine vector dips mid-segment). Backtracking cannot do
            # this: acceptance windows between flat plateaus can be
            # arbitrarily narrow.
            du = (C.T @ dlam) / h
            with np.errstate(divide="ignore", invalid="ignore"):
                t_cand = np.concatenate([(lb - u) / du, (ub - u) / du])
            t_cand = t_cand[np.isfinite(t_cand) & (t_cand > 1e-16)]
            t_cand = np.unique(np.concatenate([t_cand, [0.0, 1.0]]))
            if t_cand.size > 400:
                keep = np.unique(
                    np.linspace(0, t_cand.size - 1, 400).astype(int))
                t_cand = t_cand[keep]
            X = np.clip(u[None, :] + t_cand[:, None] * du[None, :], lb, ub)
            R = X @ C.T - d[None, :]                     # (K, m)
            norms = np.linalg.norm(R, axis=1)
            # Interior minimisers: within [t_a, t_b], r(t) = r_a + s·(r_b−r_a),
            # s ∈ (0,1); min ‖·‖ at s* = −r_aᵀΔ/‖Δ‖².
            best_t = float(t_cand[int(np.argmin(norms))])
            best_norm = float(norms.min())
            if t_cand.size > 1:
                Dr = R[1:] - R[:-1]
                denom = (Dr * Dr).sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    s_star = -np.einsum("ij,ij->i", R[:-1], Dr) / denom
                ok = np.isfinite(s_star) & (s_star > 0) & (s_star < 1)
                if ok.any():
                    Ri = R[:-1][ok] + s_star[ok, None] * Dr[ok]
                    ni = np.linalg.norm(Ri, axis=1)
                    j = int(np.argmin(ni))
                    if ni[j] < best_norm:
                        best_norm = float(ni[j])
                        ta = t_cand[:-1][ok][j]
                        tb = t_cand[1:][ok][j]
                        best_t = float(ta + s_star[ok][j] * (tb - ta))
            if best_norm < r0 - 1e-12 * max(r0, 1.0) and best_t > 0:
                lam = lam + best_t * dlam
                moved = True
                break
        if not moved:
            break  # genuine stall — the strict residual check decides below

    final_resid = float(np.max(np.abs(C @ x - d)))
    if strict and final_resid > 1e-6 * scale:
        # NEVER silently return an infeasible point — semismooth Newton
        # can stall at kinks on hard (e.g. chained-equality) instances.
        raise np.linalg.LinAlgError(
            f"dual Newton did not converge (residual {final_resid:.3e})")

    bound_tol = 1e-9
    active = np.zeros(x.shape[0], dtype=int)
    active[x >= ub - bound_tol * np.maximum(1.0, np.abs(ub))] = 1
    active[x <= lb + bound_tol * np.maximum(1.0, np.abs(lb))] = -1
    active[(ub - lb) <= 1e-12] = -1  # degenerate bounds count as pinned
    return x, active, lam


def _alternating_projection_feasible(
    C: np.ndarray,
    d: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Find a point in {Cx = d} ∩ {lb ≤ x ≤ ub} by alternating projections.

    POCS converges for nonempty intersections of convex sets; used to
    seed the ratio-test primal QP when no analytic feasible start is
    available. Raises if the residual does not vanish (infeasible or
    pathological geometry).
    """
    x = np.clip(0.5 * (lb + np.minimum(ub, lb + 1.0)), lb, ub)
    CT = C.T
    gram = C @ CT
    gram[np.diag_indices_from(gram)] += 1e-12
    scale = max(1.0, float(np.abs(d).max()))
    for _ in range(max_iter):
        resid = C @ x - d
        if float(np.max(np.abs(resid))) < tol * scale:
            return x
        x = x - CT @ np.linalg.solve(gram, resid)
        x = np.clip(x, lb, ub)
    resid = float(np.max(np.abs(C @ x - d)))
    if resid < 1e-7 * scale:
        return x
    raise np.linalg.LinAlgError(
        f"no feasible point found (POCS residual {resid:.3e})")


def _solve_box_eq_qp_primal(
    H: np.ndarray,
    q: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x0: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Textbook primal active-set QP from a FEASIBLE start ``x0``.

    min ½xᵀHx + qᵀx  s.t.  Cx = d, lb ≤ x ≤ ub. Unlike the greedy
    :func:`_solve_period_qp` (which pins the most-violated variable of an
    unconstrained KKT solve and can over-pin into inconsistency), this
    maintains primal feasibility throughout: each iteration solves the
    equality-constrained QP on the current working set, then moves along
    the direction with a RATIO TEST, adding the first blocking bound.
    Equality rows whose variables are all pinned stay satisfied because
    the iterate never leaves the feasible set. Converges finitely for
    SPD ``H``.

    Returns ``(x, active)`` with the usual coding (0 free, ±1 at bound).
    """
    n = H.shape[0]
    x = np.clip(np.asarray(x0, dtype=float).copy(), lb, ub)
    span = ub - lb
    active = np.zeros(n, dtype=int)
    active[x >= ub - 1e-12 * np.maximum(1.0, np.abs(ub))] = 1
    active[x <= lb + 1e-12 * np.maximum(1.0, np.abs(lb))] = -1
    active[span <= 1e-12] = -1  # degenerate bounds: permanently pinned

    for _ in range(max_iter):
        free = active == 0
        idx_f = np.where(free)[0]
        idx_x = np.where(~free)[0]
        kf = idx_f.size
        # Equality rows with at least one free variable enter the EQP;
        # rows fully on pinned variables are already satisfied (feasible
        # iterate) and constrain nothing that can move.
        row_live = (np.abs(C[:, idx_f]).sum(axis=1) > 1e-14) if kf else \
            np.zeros(C.shape[0], dtype=bool)
        Cf = C[np.ix_(row_live, idx_f)]
        m_live = Cf.shape[0]

        # EQP target on the free block (pinned vars held at x).
        rhs_top = -q[idx_f] - H[np.ix_(idx_f, idx_x)] @ x[idx_x] \
            if idx_x.size else -q[idx_f]
        rhs_bot = d[row_live] - C[np.ix_(row_live, idx_x)] @ x[idx_x] \
            if idx_x.size else d[row_live]
        KKT = np.zeros((kf + m_live, kf + m_live))
        KKT[:kf, :kf] = H[np.ix_(idx_f, idx_f)]
        KKT[:kf, kf:] = Cf.T
        KKT[kf:, :kf] = Cf
        try:
            sol = np.linalg.solve(KKT, np.concatenate([rhs_top, rhs_bot]))
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(
                KKT, np.concatenate([rhs_top, rhs_bot]), rcond=None)[0]
        x_star_f = sol[:kf]
        lam_live = sol[kf:]
        p = np.zeros(n)
        p[idx_f] = x_star_f - x[idx_f]

        if float(np.max(np.abs(p))) > tol:
            # Ratio test: longest step in [0, 1] keeping bounds.
            alpha = 1.0
            blocker = -1
            block_side = 0
            for j in idx_f:
                if p[j] > tol and x[j] + p[j] > ub[j]:
                    a = (ub[j] - x[j]) / p[j]
                    if a < alpha:
                        alpha, blocker, block_side = a, j, 1
                elif p[j] < -tol and x[j] + p[j] < lb[j]:
                    a = (lb[j] - x[j]) / p[j]
                    if a < alpha:
                        alpha, blocker, block_side = a, j, -1
            x = x + max(alpha, 0.0) * p
            if blocker >= 0:
                active[blocker] = block_side
                x[blocker] = ub[blocker] if block_side > 0 else lb[blocker]
                continue
            # Full step taken — fall through to the dual check.

        # Dual feasibility of pinned vars: reduced cost r = Hx + q + Cᵀλ
        # (λ = 0 on dropped rows). At ub optimal needs r ≤ 0; at lb r ≥ 0.
        lam_full = np.zeros(C.shape[0])
        lam_full[row_live] = lam_live
        r = H @ x + q + C.T @ lam_full
        release = -1
        worst = tol
        for j in idx_x:
            if span[j] <= 1e-12:
                continue  # degenerate: never released
            if active[j] > 0 and r[j] > worst:
                worst, release = r[j], j
            elif active[j] < 0 and -r[j] > worst:
                worst, release = -r[j], j
        if release < 0:
            return x, active
        active[release] = 0
    return x, active


def solve_multibus_dispatch_with_sensitivities(
    problem: MultiBusDispatchProblem,
    *,
    jacobians: tuple = ("mc", "capacity", "demand", "line_limit"),
) -> MultiBusDispatchSolution:
    """Multi-bus, multi-period ridge-QP dispatch with analytic gradients.

    Solves each period's transport-model QP by an active-set method, then
    differentiates the KKT system on the frozen active set (implicit
    function theorem; Amos & Kolter 2017, Agrawal et al. 2019) to obtain
    exact ``d dispatch / d {mc, capacity, demand, line_limit}``. Pure
    numpy — no torch required.

    ``jacobians`` selects which sensitivity blocks to compute; skipped
    blocks come back as zeros. Pass ``()`` for a forward-only solve (what
    a sampling baseline pays per draw) or ``("mc",)`` when only cost-side
    gradients are chained (e.g. CO₂-price calibration).
    """
    gen_bus = np.asarray(problem.gen_bus, dtype=int)
    mc = np.asarray(problem.marginal_cost, dtype=float)
    cap = np.asarray(problem.capacity, dtype=float)
    lf = np.asarray(problem.line_from, dtype=int)
    lt = np.asarray(problem.line_to, dtype=int)
    flim = np.asarray(problem.line_limit, dtype=float)
    demand = np.asarray(problem.demand, dtype=float)
    B = int(problem.n_buses)
    ridge = float(problem.ridge)
    if ridge <= 0:
        raise ValueError("ridge must be > 0")

    G = mc.shape[0]
    L = lf.shape[0]
    T = demand.shape[1]
    if demand.shape[0] != B:
        raise ValueError("demand must have shape (n_buses, T)")
    n = G + L  # decision vars per period: [p (G); f (L)]

    # Bus–line incidence A: flow f_l adds to its "to" bus, subtracts at
    # "from" bus. Balance: Σ_{g∈b} p + Σ_l A[b,l] f = d_b.
    A = np.zeros((B, L))
    for l in range(L):
        A[lt[l], l] += 1.0
        A[lf[l], l] -= 1.0
    # Generator–bus incidence.
    GB = np.zeros((B, G))
    for g in range(G):
        GB[gen_bus[g], g] = 1.0
    C = np.hstack([GB, A])  # (B, n)

    # Hessian: ridge on gens; a smaller ridge on flows keeps SPD without
    # distorting dispatch (flows are otherwise cost-free).
    flow_ridge = ridge * 1e-3
    H = np.diag(np.concatenate([np.full(G, ridge), np.full(L, flow_ridge)]))
    q_base = np.concatenate([mc, np.zeros(L)])  # linear cost: only gens

    avail = (np.ones((G, T)) if problem.availability is None
             else np.asarray(problem.availability, dtype=float))
    if avail.shape != (G, T):
        raise ValueError("availability must have shape (G, T)")
    symmetric_lines = problem.line_min is None
    flow_lb = (-flim if symmetric_lines
               else np.asarray(problem.line_min, dtype=float))

    dispatch = np.zeros((G, T))
    flows = np.zeros((L, T))
    # Jacobian accumulators.
    d_mc = np.zeros((G * T, G))
    d_cap = np.zeros((G * T, G))
    d_dem = np.zeros((G * T, B * T))
    d_flim = np.zeros((G * T, L))

    Hinv = np.linalg.inv(H)

    h_diag = np.diag(H).copy()
    lam_prev: Optional[np.ndarray] = None
    for t in range(T):
        d_t = demand[:, t]
        lb = np.concatenate([np.zeros(G), flow_lb])
        ub = np.concatenate([cap * avail[:, t], flim])
        try:
            x, active, lam_prev = _solve_period_qp_dual(
                h_diag, q_base, C, d_t, lb, ub, lam0=lam_prev)
        except np.linalg.LinAlgError:
            try:
                # A warm-started λ from a very different regime (previous
                # period under other parameters) can strand the semismooth
                # Newton at a kink — retry cold first.
                x, active, lam_prev = _solve_period_qp_dual(
                    h_diag, q_base, C, d_t, lb, ub, lam0=None)
            except np.linalg.LinAlgError:
                # Last resort (degenerate kink geometry): feasible point
                # by alternating projections, then the stall-proof
                # ratio-test primal active set.
                x0 = _alternating_projection_feasible(C, d_t, lb, ub)
                x, active = _solve_box_eq_qp_primal(
                    H, q_base, C, d_t, lb, ub, x0)
                lam_prev = None
        x = np.clip(x, lb, ub)
        dispatch[:, t] = x[:G]
        flows[:, t] = x[G:]

        if not jacobians:
            continue  # forward-only solve

        # ---- Sensitivities on the frozen active set ----
        # Free vars move; pinned vars are constants (= their bound). Let
        # F = free indices. The active optimum satisfies
        #   H_FF x_F + q_F + C_Fᵀ λ = -H_FX x_X
        #   C_F x_F                  =  d - C_X x_X
        # Differentiating w.r.t. a parameter θ (with x_X piecewise const,
        # dx_X/dθ = d(bound)/dθ) gives a linear system in (dx_F, dλ).
        free = active == 0
        idx_f = np.where(free)[0]
        idx_x = np.where(~free)[0]
        kf = idx_f.size
        Hff = H[np.ix_(idx_f, idx_f)]
        Cf = C[:, idx_f]
        KKT = np.zeros((kf + B, kf + B))
        KKT[:kf, :kf] = Hff
        KKT[:kf, kf:] = Cf.T
        KKT[kf:, :kf] = Cf
        try:
            KKT_inv = np.linalg.inv(KKT)
        except np.linalg.LinAlgError:
            # Degenerate optimum (a bus row with every incident variable
            # pinned). The group-inverse sensitivities are still the
            # correct one-sided derivatives for consistent perturbations.
            KKT_inv = np.linalg.pinv(KKT)

        # Map a (rhs_top (kf), rhs_bot (B)) → dx over ALL n vars, with the
        # pinned-var derivatives injected separately.
        def solve_sens(rhs_top: np.ndarray, rhs_bot: np.ndarray,
                       dx_fixed: np.ndarray) -> np.ndarray:
            sol = KKT_inv @ np.concatenate([rhs_top, rhs_bot])
            dx = dx_fixed.copy()
            dx[idx_f] = sol[:kf]
            return dx

        # Which gens are pinned at capacity (their dx = dcap), and which at
        # zero (dx = 0). Flows pinned at ±limit similarly.
        at_cap_gen = (active[:G] > 0)
        at_lo_gen = (active[:G] < 0)  # gen at 0
        at_hi_flow = (active[G:] > 0)
        at_lo_flow = (active[G:] < 0)

        # ----- d/d mc_j (gen j cost) -----
        for j in (range(G) if "mc" in jacobians else ()):
            # ∂q/∂mc_j = e_j (gen block). Pinned vars don't move (bounds
            # independent of mc) ⇒ dx_fixed = 0, rhs_bot = 0.
            rt = np.zeros(kf)
            # rhs_top = -(∂q_F) - H_FX dx_X = -∂q_F  (dx_X = 0)
            # ∂q at free index i is 1 if that free var is gen j.
            for ii, fi in enumerate(idx_f):
                if fi == j:
                    rt[ii] = -1.0
            dx = solve_sens(rt, np.zeros(B), np.zeros(n))
            for g in range(G):
                d_mc[g * T + t, j] = dx[g]

        # ----- d/d capacity_j -----
        for j in (range(G) if "capacity" in jacobians else ()):
            dx_fixed = np.zeros(n)
            rt = np.zeros(kf)
            rb = np.zeros(B)
            if at_cap_gen[j]:
                # x_j = cap_j·avail_jt ⇒ dx_j/dcap_j = avail_jt (pinned). Its
                # motion feeds the free system:
                # rhs_top -= H_FX·dx_X ; rhs_bot -= C_X·dx_X.
                a_jt = avail[j, t]
                dx_fixed[j] = a_jt
                rt = -H[np.ix_(idx_f, [j])][:, 0] * a_jt
                rb = -C[:, j] * a_jt
            # else capacity bound inactive ⇒ no effect.
            dx = solve_sens(rt, rb, dx_fixed)
            for g in range(G):
                d_cap[g * T + t, j] = dx[g]

        # ----- d/d demand_{b,t'} ----- (only t' == t couples)
        for b in (range(B) if "demand" in jacobians else ()):
            rb = np.zeros(B)
            rb[b] = 1.0  # ∂(C x = d)/∂d_b
            dx = solve_sens(np.zeros(kf), rb, np.zeros(n))
            col = b * T + t
            for g in range(G):
                d_dem[g * T + t, col] = dx[g]

        # ----- d/d line_limit_j -----
        for j in (range(L) if "line_limit" in jacobians else ()):
            dx_fixed = np.zeros(n)
            rt = np.zeros(kf)
            rb = np.zeros(B)
            fvar = G + j
            if at_hi_flow[j]:
                dx_fixed[fvar] = 1.0   # f_j = +limit_j
                rt = -H[np.ix_(idx_f, [fvar])][:, 0]
                rb = -C[:, fvar]
            elif at_lo_flow[j] and symmetric_lines:
                dx_fixed[fvar] = -1.0  # f_j = -limit_j
                rt = -H[np.ix_(idx_f, [fvar])][:, 0] * (-1.0)
                rb = -C[:, fvar] * (-1.0)
            # explicit line_min: lower bound is an independent constant ⇒
            # a flow pinned there contributes nothing to d/d limit.
            dx = solve_sens(rt, rb, dx_fixed)
            for g in range(G):
                d_flim[g * T + t, j] = dx[g]

    return MultiBusDispatchSolution(
        dispatch=dispatch,
        flows=flows,
        d_dispatch_d_mc=d_mc,
        d_dispatch_d_capacity=d_cap,
        d_dispatch_d_demand=d_dem,
        d_dispatch_d_linelimit=d_flim,
    )


@dataclass
class MultiBusStorageProblem:
    """Multibus dispatch WITH bus-attached storage (N_En_Phase 19.x.2).

    Extends :class:`MultiBusDispatchProblem` with ``S`` storages. Solved
    as ONE stacked QP over all periods (SOC couples them), so keep the
    window small — representative days / weeks (T ≤ ~48): the dense
    stacked KKT is O(((G+L+3S)·T)³).

    Storage arrays, all shape ``(S,)``: ``sto_bus`` (int), ``charge_eff``,
    ``discharge_eff`` ∈ (0,1], ``power_limit``, ``soc_max``, ``soc_init``.
    """
    gen_bus: np.ndarray
    marginal_cost: np.ndarray
    capacity: np.ndarray
    line_from: np.ndarray
    line_to: np.ndarray
    line_limit: np.ndarray
    demand: np.ndarray
    n_buses: int
    sto_bus: np.ndarray = None
    charge_eff: np.ndarray = None
    discharge_eff: np.ndarray = None
    power_limit: np.ndarray = None
    soc_max: np.ndarray = None
    soc_init: np.ndarray = None
    ridge: float = 1e-2
    availability: Optional[np.ndarray] = None
    line_min: Optional[np.ndarray] = None


@dataclass
class MultiBusStorageSolution:
    """Stacked solution + the Jacobian blocks calibration needs.

    Jacobians over the flattened (g, t) dispatch (G·T rows):
      d_dispatch_d_mc            (G·T, G)
      d_dispatch_d_charge_eff    (G·T, S)
      d_dispatch_d_discharge_eff (G·T, S)
      d_dispatch_d_soc_init      (G·T, S)
    SOC-trace blocks (S·T rows, flattened (s, t)):
      d_soc_d_charge_eff / d_soc_d_discharge_eff  (S·T, S)
    """
    dispatch: np.ndarray   # (G, T)
    flows: np.ndarray      # (L, T)
    charge: np.ndarray     # (S, T)
    discharge: np.ndarray  # (S, T)
    soc: np.ndarray        # (S, T)
    d_dispatch_d_mc: np.ndarray
    d_dispatch_d_charge_eff: np.ndarray
    d_dispatch_d_discharge_eff: np.ndarray
    d_dispatch_d_soc_init: np.ndarray
    d_soc_d_charge_eff: np.ndarray
    d_soc_d_discharge_eff: np.ndarray


def solve_multibus_storage_dispatch_with_sensitivities(
    problem: MultiBusStorageProblem,
    *,
    jacobians: tuple = ("mc", "eta", "soc_init"),
) -> MultiBusStorageSolution:
    """Stacked multibus + storage ridge-QP with analytic gradients.

    Composition of the per-period multibus transport QP and the
    inter-temporal SOC continuity used in the single-bus storage layer:

        vars/period: [p (G); f (L); ch (S); dis (S); soc (S)]
        balance_b,t: Σ_{g∈b} p + Σ_l A f + Σ_{s∈b} (dis − ch) = d_{b,t}
        soc_s,t:     soc_t − soc_{t−1} − η_c ch_t + dis_t/η_d = 0

    Forward: feasible start from the STORAGE-FREE per-period multibus
    solve (robust dual Newton), batteries idle at soc_init — then the
    ratio-test primal active set walks to the coupled optimum without
    ever leaving the feasible set. Sensitivities: IFT on the frozen
    active set; η blocks via equality-dual recovery (constraint-matrix
    parameters, same construction as the single-bus layer).
    """
    gen_bus = np.asarray(problem.gen_bus, dtype=int)
    mc = np.asarray(problem.marginal_cost, dtype=float)
    cap = np.asarray(problem.capacity, dtype=float)
    lf = np.asarray(problem.line_from, dtype=int)
    lt = np.asarray(problem.line_to, dtype=int)
    flim = np.asarray(problem.line_limit, dtype=float)
    demand = np.asarray(problem.demand, dtype=float)
    B = int(problem.n_buses)
    sbus = np.asarray(problem.sto_bus, dtype=int)
    eta_c = np.asarray(problem.charge_eff, dtype=float)
    eta_d = np.asarray(problem.discharge_eff, dtype=float)
    plim = np.asarray(problem.power_limit, dtype=float)
    smax = np.asarray(problem.soc_max, dtype=float)
    s0 = np.asarray(problem.soc_init, dtype=float)
    ridge = float(problem.ridge)
    G, L, S = mc.shape[0], lf.shape[0], sbus.shape[0]
    T = demand.shape[1]
    avail = (np.ones((G, T)) if problem.availability is None
             else np.asarray(problem.availability, dtype=float))
    flow_lb_1 = (-flim if problem.line_min is None
                 else np.asarray(problem.line_min, dtype=float))

    blk = G + L + 3 * S
    n = blk * T
    P0, F0, CH0, DIS0, SOC0 = 0, G, G + L, G + L + S, G + L + 2 * S

    def ip(t, k):
        return t * blk + k

    m_eq = (B + S) * T  # per t: B balance rows then S soc rows
    C = np.zeros((m_eq, n))
    d_eq = np.zeros(m_eq)
    for t in range(T):
        rb0 = t * (B + S)
        for g in range(G):
            C[rb0 + gen_bus[g], ip(t, P0 + g)] = 1.0
        for l in range(L):
            C[rb0 + lt[l], ip(t, F0 + l)] += 1.0
            C[rb0 + lf[l], ip(t, F0 + l)] -= 1.0
        for s in range(S):
            C[rb0 + sbus[s], ip(t, DIS0 + s)] = 1.0
            C[rb0 + sbus[s], ip(t, CH0 + s)] = -1.0
        d_eq[rb0:rb0 + B] = demand[:, t]
        for s in range(S):
            r = rb0 + B + s
            C[r, ip(t, SOC0 + s)] = 1.0
            C[r, ip(t, CH0 + s)] = -eta_c[s]
            C[r, ip(t, DIS0 + s)] = 1.0 / eta_d[s]
            if t == 0:
                d_eq[r] = s0[s]
            else:
                C[r, ip(t - 1, SOC0 + s)] = -1.0

    lb = np.zeros(n)
    ub = np.zeros(n)
    for t in range(T):
        for g in range(G):
            ub[ip(t, P0 + g)] = cap[g] * avail[g, t]
        for l in range(L):
            lb[ip(t, F0 + l)] = flow_lb_1[l]
            ub[ip(t, F0 + l)] = flim[l]
        for s in range(S):
            ub[ip(t, CH0 + s)] = plim[s]
            ub[ip(t, DIS0 + s)] = plim[s]
            ub[ip(t, SOC0 + s)] = smax[s]

    flow_ridge = ridge * 1e-3
    h_diag = np.zeros(n)
    q = np.zeros(n)
    for t in range(T):
        h_diag[ip(t, P0):ip(t, P0) + G] = ridge
        h_diag[ip(t, F0):ip(t, F0) + L] = flow_ridge
        h_diag[ip(t, CH0):ip(t, SOC0) + S] = ridge
        q[ip(t, P0):ip(t, P0) + G] = mc
    H = np.diag(h_diag)

    # ---- feasible start: storage-free per-period multibus solve ----
    base = solve_multibus_dispatch_with_sensitivities(
        MultiBusDispatchProblem(
            gen_bus=gen_bus, marginal_cost=mc, capacity=cap,
            line_from=lf, line_to=lt, line_limit=flim, demand=demand,
            n_buses=B, ridge=ridge, availability=avail,
            line_min=problem.line_min),
        jacobians=())
    x0 = np.zeros(n)
    for t in range(T):
        x0[ip(t, P0):ip(t, P0) + G] = base.dispatch[:, t]
        x0[ip(t, F0):ip(t, F0) + L] = base.flows[:, t]
        for s in range(S):
            x0[ip(t, SOC0 + s)] = min(s0[s], smax[s])

    x, active = _solve_box_eq_qp_primal(H, q, C, d_eq, lb, ub, x0)
    x = np.clip(x, lb, ub)

    dispatch = np.zeros((G, T))
    flows = np.zeros((L, T))
    charge = np.zeros((S, T))
    discharge = np.zeros((S, T))
    soc = np.zeros((S, T))
    for t in range(T):
        dispatch[:, t] = x[ip(t, P0):ip(t, P0) + G]
        flows[:, t] = x[ip(t, F0):ip(t, F0) + L]
        charge[:, t] = x[ip(t, CH0):ip(t, CH0) + S]
        discharge[:, t] = x[ip(t, DIS0):ip(t, DIS0) + S]
        soc[:, t] = x[ip(t, SOC0):ip(t, SOC0) + S]

    # ---- IFT sensitivities on the frozen active set ----
    free = active == 0
    idx_f = np.where(free)[0]
    idx_x = np.where(~free)[0]
    kf = idx_f.size
    Cf = C[:, idx_f]
    KKT = np.zeros((kf + m_eq, kf + m_eq))
    KKT[:kf, :kf] = H[np.ix_(idx_f, idx_f)]
    KKT[:kf, kf:] = Cf.T
    KKT[kf:, :kf] = Cf
    try:
        KKT_inv = np.linalg.inv(KKT)
    except np.linalg.LinAlgError:
        KKT_inv = np.linalg.pinv(KKT)

    def solve_sens(rhs_top, rhs_bot):
        sol = KKT_inv @ np.concatenate([rhs_top, rhs_bot])
        dx = np.zeros(n)
        dx[idx_f] = sol[:kf]
        return dx

    disp_idx = np.array([ip(t, P0 + g) for g in range(G) for t in range(T)])
    soc_idx_all = np.array([ip(t, SOC0 + s) for s in range(S)
                            for t in range(T)])
    pos_in_free = {fi: ii for ii, fi in enumerate(idx_f)}

    d_mc = np.zeros((G * T, G))
    d_eta_c = np.zeros((G * T, S))
    d_eta_d = np.zeros((G * T, S))
    d_s0 = np.zeros((G * T, S))
    d_soc_eta_c = np.zeros((S * T, S))
    d_soc_eta_d = np.zeros((S * T, S))

    if "mc" in jacobians:
        for j in range(G):
            rt = np.zeros(kf)
            for t in range(T):
                fi = ip(t, P0 + j)
                if fi in pos_in_free:
                    rt[pos_in_free[fi]] = -1.0
            d_mc[:, j] = solve_sens(rt, np.zeros(m_eq))[disp_idx]

    if "eta" in jacobians:
        # λ recovery (original-rhs matvec; H diagonal ⇒ no cross term).
        rhs0_top = -q[idx_f]
        rhs0_bot = d_eq - (C[:, idx_x] @ x[idx_x] if idx_x.size else 0.0)
        lam = (KKT_inv @ np.concatenate([rhs0_top, rhs0_bot]))[kf:]
        for s in range(S):
            # η_c[s]: ∂C[soc_row, CH] = −1 on storage s's rows.
            rt = np.zeros(kf)
            rb = np.zeros(m_eq)
            for t in range(T):
                r = t * (B + S) + B + s
                fi = ip(t, CH0 + s)
                if fi in pos_in_free:
                    rt[pos_in_free[fi]] = lam[r]
                rb[r] = charge[s, t]
            dx = solve_sens(rt, rb)
            d_eta_c[:, s] = dx[disp_idx]
            d_soc_eta_c[:, s] = dx[soc_idx_all]
            # η_d[s]: ∂C[soc_row, DIS] = −1/η_d².
            rt = np.zeros(kf)
            rb = np.zeros(m_eq)
            for t in range(T):
                r = t * (B + S) + B + s
                fi = ip(t, DIS0 + s)
                if fi in pos_in_free:
                    rt[pos_in_free[fi]] = lam[r] / eta_d[s] ** 2
                rb[r] = discharge[s, t] / eta_d[s] ** 2
            dx = solve_sens(rt, rb)
            d_eta_d[:, s] = dx[disp_idx]
            d_soc_eta_d[:, s] = dx[soc_idx_all]

    if "soc_init" in jacobians:
        for s in range(S):
            rb = np.zeros(m_eq)
            rb[0 * (B + S) + B + s] = 1.0  # t=0 SOC row RHS
            d_s0[:, s] = solve_sens(np.zeros(kf), rb)[disp_idx]

    return MultiBusStorageSolution(
        dispatch=dispatch, flows=flows, charge=charge,
        discharge=discharge, soc=soc,
        d_dispatch_d_mc=d_mc,
        d_dispatch_d_charge_eff=d_eta_c,
        d_dispatch_d_discharge_eff=d_eta_d,
        d_dispatch_d_soc_init=d_s0,
        d_soc_d_charge_eff=d_soc_eta_c,
        d_soc_d_discharge_eff=d_soc_eta_d,
    )


@dataclass
class MultiBusDispatchLayer:
    """Stateful OO wrapper around the multi-bus differentiable dispatch.

    Mirrors :class:`EconomicDispatchLayer`: :meth:`forward` caches the
    solution + Jacobians; :meth:`backward` pulls ``dL/dθ`` from
    ``dL/d dispatch`` without re-solving. Torch-free.
    """
    ridge: float = 1e-2
    _sol: MultiBusDispatchSolution | None = field(
        default=None, init=False, repr=False)

    def forward(self, problem: MultiBusDispatchProblem) -> np.ndarray:
        prob = MultiBusDispatchProblem(
            gen_bus=problem.gen_bus,
            marginal_cost=problem.marginal_cost,
            capacity=problem.capacity,
            line_from=problem.line_from,
            line_to=problem.line_to,
            line_limit=problem.line_limit,
            demand=problem.demand,
            n_buses=problem.n_buses,
            ridge=self.ridge,
        )
        sol = solve_multibus_dispatch_with_sensitivities(prob)
        self._sol = sol
        return sol.dispatch

    def backward(self, grad_dispatch: np.ndarray) -> dict[str, np.ndarray]:
        """Return ``dL/d{mc, capacity, demand, line_limit}`` given dL/ddispatch.

        ``grad_dispatch`` has shape ``(G, T)`` and is flattened row-major
        to match the Jacobian layout.
        """
        if self._sol is None:
            raise RuntimeError("call forward() before backward()")
        g = np.asarray(grad_dispatch, dtype=float).reshape(-1)
        s = self._sol
        return {
            "marginal_cost": s.d_dispatch_d_mc.T @ g,
            "capacity": s.d_dispatch_d_capacity.T @ g,
            "demand": s.d_dispatch_d_demand.T @ g,
            "line_limit": s.d_dispatch_d_linelimit.T @ g,
        }


# ---------------------------------------------------------------------------
# Phase 12.3 — demand-elasticity recovery via differentiable dispatch
# ---------------------------------------------------------------------------
#
# Reference training loop: from observed (price, dispatch) pairs, recover
# a linear demand-elasticity parameter by gradient descent through the
# differentiable single-bus dispatch layer. This is the cvxpylayers-style
# "learn the parameters of an optimisation problem from its solutions"
# workflow (Agrawal et al. 2019), but using the analytic gradients we
# already expose — no torch, no cvxpy.
#
# Model. Demand responds linearly to a reference price p0:
#     d(elast) = d0 - elast · (price - p0)
# Given the marginal costs + capacities, the dispatch layer maps demand
# to generator output p*. We observe p* (and the clearing price) for
# several price points generated under a *true* elasticity, then fit
# ``elast`` by minimising ½‖p_pred(elast) - p_obs‖² with the chain rule
#     dL/d elast = Σ (p_pred - p_obs)ᵀ (dp/d demand) (d demand / d elast),
# where dp/d demand is exactly ``DispatchJacobian.dp_ddemand`` and
# d demand/d elast = -(price - p0).


@dataclass
class ElasticityFitResult:
    """Outcome of :func:`fit_demand_elasticity`."""
    elasticity: float
    history: list[float]            # loss per iteration
    elasticity_history: list[float]  # parameter per iteration
    n_iter: int


def fit_demand_elasticity(
    prices: np.ndarray,
    observed_dispatch: np.ndarray,
    marginal_cost: np.ndarray,
    capacity: np.ndarray,
    base_demand: float,
    reference_price: float,
    *,
    ridge: float = 1.0,
    lr: float = 1e-4,
    n_iter: int = 400,
    elasticity_init: float = 0.0,
    tol: float = 1e-12,
) -> ElasticityFitResult:
    """Recover a linear demand-elasticity by gradient descent.

    ``observed_dispatch`` is ``(K, G)`` — the dispatch observed at each of
    ``K`` price points in ``prices`` ``(K,)``. The demand model is
    ``d_k = base_demand - elast · (prices[k] - reference_price)``. We fit
    ``elast`` so the differentiable dispatch layer reproduces the observed
    dispatch, using the analytic ``dp/d demand`` Jacobian.

    Returns an :class:`ElasticityFitResult`. Pure numpy.
    """
    prices = np.asarray(prices, dtype=float)
    observed = np.asarray(observed_dispatch, dtype=float)
    mc = np.asarray(marginal_cost, dtype=float)
    cap = np.asarray(capacity, dtype=float)
    K = prices.shape[0]
    if observed.shape[0] != K:
        raise ValueError("observed_dispatch must have K rows (one per price)")

    elast = float(elasticity_init)
    loss_hist: list[float] = []
    elast_hist: list[float] = []

    for it in range(n_iter):
        grad = 0.0
        total_loss = 0.0
        for k in range(K):
            dk = base_demand - elast * (prices[k] - reference_price)
            dk = max(dk, 0.0)
            p, jac = solve_dispatch_with_sensitivities(mc, cap, dk, ridge=ridge)
            resid = p - observed[k]
            total_loss += 0.5 * float(resid @ resid)
            # dL/d elast = residᵀ (dp/d demand) (d demand/d elast)
            ddemand_delast = -(prices[k] - reference_price)
            grad += float(resid @ jac.dp_ddemand) * ddemand_delast
        loss_hist.append(total_loss)
        elast_hist.append(elast)
        new_elast = elast - lr * grad
        if abs(new_elast - elast) < tol:
            elast = new_elast
            break
        elast = new_elast

    return ElasticityFitResult(
        elasticity=float(elast),
        history=loss_hist,
        elasticity_history=elast_hist,
        n_iter=len(loss_hist),
    )


# ---------------------------------------------------------------------------
# Phase 12.2 — differentiable MULTI-PERIOD storage dispatch
# ---------------------------------------------------------------------------
#
# The Phase-12 layers above are single-period (or per-period-independent
# multi-bus). Storage is what makes dispatch genuinely *inter-temporal*:
# the state of charge couples every period to the next. We solve the
# strictly-convex storage-dispatch QP as ONE stacked QP over the whole
# horizon (periods can no longer be solved independently) and
# differentiate its KKT system on the active set — the same OptNet /
# implicit-function-theorem machinery (Amos & Kolter 2017; Agrawal et al.
# 2019) used by ``solve_multibus_dispatch_with_sensitivities`` and reusing
# its ``_solve_period_qp`` active-set core verbatim, just over a larger
# decision vector.
#
# Decision vector x stacks, for each period t:
#     p_{g,t}   generator dispatch     (G per period)
#     ch_t      storage charge  ≥ 0    (1 per period)
#     dis_t     storage discharge ≥ 0  (1 per period)
#     soc_t     state of charge        (1 per period)   0 ≤ soc_t ≤ soc_max
# so n = (G + 3)·T.
#
# Equality constraints:
#   (balance, per period t)  Σ_g p_{g,t} + dis_t − ch_t = d_t
#   (SOC continuity, per t)  soc_t − soc_{t-1} − η_c·ch_t + dis_t/η_d = 0
#     with soc_{-1} := soc_init (a differentiable parameter).
#
# Objective: Σ_t Σ_g mc_g·p_{g,t}  +  (ridge/2)·‖x‖²  (ridge makes the QP
# strictly convex so the KKT matrix is nonsingular and gradients are the
# analytic active-set sensitivities, not a.e.-zero LP gradients).
#
# Differentiable parameters exposed: marginal_cost (per gen), capacity
# (per gen), demand (per period), soc_init (scalar). The gradient through
# the SOC coupling — d dispatch / d soc_init and d dispatch / d demand_t'
# for t' ≠ t — is the inter-temporal signal a learning loop needs and is
# exactly what the stacked-KKT differentiation produces.


@dataclass
class StorageDispatchProblem:
    """A tiny single-bus multi-period storage-dispatch instance.

    Attributes:
        marginal_cost: ``(G,)`` linear cost per unit gen dispatch.
        capacity: ``(G,)`` per-generator dispatch upper bound.
        demand: ``(T,)`` per-period demand.
        charge_eff / discharge_eff: storage round-trip split η_c, η_d ∈ (0,1].
        power_limit: max charge AND discharge power per period.
        soc_max: state-of-charge upper bound.
        soc_init: initial SOC carried into period 0 (differentiable).
        ridge: strict-convexity regulariser (must be > 0).
    """
    marginal_cost: np.ndarray
    capacity: np.ndarray
    demand: np.ndarray
    charge_eff: float = 0.95
    discharge_eff: float = 0.95
    power_limit: float = 50.0
    soc_max: float = 100.0
    soc_init: float = 0.0
    ridge: float = 1e-2


@dataclass
class StorageDispatchSolution:
    """Forward storage-dispatch solution plus analytic Jacobians.

    ``dispatch`` is ``(G, T)``; ``charge`` / ``discharge`` / ``soc`` are
    ``(T,)``. Jacobians are of the flattened ``(g, t)`` dispatch outputs
    (``G·T`` rows, row-major):

        d_dispatch_d_mc:       (G·T, G)
        d_dispatch_d_capacity: (G·T, G)
        d_dispatch_d_demand:   (G·T, T)
        d_dispatch_d_soc_init: (G·T,)     ← the inter-temporal gradient
        d_dispatch_d_charge_eff:    (G·T,)  ← constraint-matrix sensitivity
        d_dispatch_d_discharge_eff: (G·T,)  ← constraint-matrix sensitivity

    The efficiency blocks differ from the rest: η_c / η_d live in the SOC
    continuity COEFFICIENTS, so their derivatives need the equality duals
    λ (recovered from the frozen-active-set KKT) — they are what an
    auto-calibration loop fits from battery telemetry.
    """
    dispatch: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    d_dispatch_d_mc: np.ndarray
    d_dispatch_d_capacity: np.ndarray
    d_dispatch_d_demand: np.ndarray
    d_dispatch_d_soc_init: np.ndarray
    d_dispatch_d_charge_eff: np.ndarray
    d_dispatch_d_discharge_eff: np.ndarray
    # Phase 20.x.1 — SOC-trace rows of the same IFT solves. Dispatch-only
    # telemetry identifies only the η_c·η_d PRODUCT; adding observed SOC
    # (battery management systems log it) splits charge from discharge
    # efficiency. Shapes (T,).
    d_soc_d_charge_eff: np.ndarray = None
    d_soc_d_discharge_eff: np.ndarray = None


def solve_storage_dispatch_with_sensitivities(
    problem: StorageDispatchProblem,
) -> StorageDispatchSolution:
    """Multi-period ridge-QP storage dispatch with analytic gradients.

    Builds the stacked equality+box QP (balance + SOC continuity), solves
    it with the active-set core (:func:`_solve_period_qp`), then
    differentiates the KKT system on the frozen active set (implicit
    function theorem) for exact ``d dispatch / d {mc, capacity, demand,
    soc_init}``. Pure numpy — no torch required. Gradients propagate
    through the SOC coupling, so ``d dispatch_{g,t} / d demand_{t'}`` is
    nonzero for ``t' ≠ t`` whenever the storage is operating off a bound.
    """
    mc = np.asarray(problem.marginal_cost, dtype=float)
    cap = np.asarray(problem.capacity, dtype=float)
    demand = np.asarray(problem.demand, dtype=float)
    G = mc.shape[0]
    T = demand.shape[0]
    if cap.shape[0] != G:
        raise ValueError("capacity and marginal_cost must share length G")
    eta_c = float(problem.charge_eff)
    eta_d = float(problem.discharge_eff)
    plim = float(problem.power_limit)
    soc_max = float(problem.soc_max)
    soc_init = float(problem.soc_init)
    ridge = float(problem.ridge)
    if ridge <= 0:
        raise ValueError("ridge must be > 0")
    if not (0 < eta_c <= 1 and 0 < eta_d <= 1):
        raise ValueError("efficiencies must be in (0, 1]")

    # Per-period block layout: [p_0..p_{G-1}, ch, dis, soc]; width blk.
    blk = G + 3
    n = blk * T

    def ip(t: int, k: int) -> int:  # index of var k within period t
        return t * blk + k

    P0 = 0          # gen offset within block
    CH = G          # charge
    DIS = G + 1     # discharge
    SOC = G + 2     # soc

    # ---- Equality constraints: balance (T) then SOC continuity (T) ----
    m_eq = 2 * T
    C = np.zeros((m_eq, n))
    d_eq = np.zeros(m_eq)
    # rows 0..T-1: balance. rows T..2T-1: SOC continuity.
    for t in range(T):
        # balance: Σ_g p + dis − ch = demand_t
        for g in range(G):
            C[t, ip(t, P0 + g)] = 1.0
        C[t, ip(t, DIS)] = 1.0
        C[t, ip(t, CH)] = -1.0
        d_eq[t] = demand[t]
        # SOC continuity: soc_t − soc_{t-1} − η_c·ch_t + dis_t/η_d = 0
        r = T + t
        C[r, ip(t, SOC)] = 1.0
        C[r, ip(t, CH)] = -eta_c
        C[r, ip(t, DIS)] = 1.0 / eta_d
        if t == 0:
            d_eq[r] = soc_init  # soc_0 − η_c ch + dis/η_d = soc_init
        else:
            C[r, ip(t - 1, SOC)] = -1.0
            d_eq[r] = 0.0

    # ---- Box bounds ----
    lb = np.zeros(n)
    ub = np.zeros(n)
    for t in range(T):
        for g in range(G):
            lb[ip(t, P0 + g)] = 0.0
            ub[ip(t, P0 + g)] = cap[g]
        lb[ip(t, CH)] = 0.0;  ub[ip(t, CH)] = plim
        lb[ip(t, DIS)] = 0.0; ub[ip(t, DIS)] = plim
        lb[ip(t, SOC)] = 0.0; ub[ip(t, SOC)] = soc_max

    # ---- Hessian + linear cost ----
    H = ridge * np.eye(n)
    q = np.zeros(n)
    for t in range(T):
        for g in range(G):
            q[ip(t, P0 + g)] = mc[g]

    # ---- Forward solve ----
    # Greedy active-set first (its "degenerate vars stay free at bound"
    # behaviour keeps the sensitivity KKT nonsingular for callers like
    # fit_component_params that zero out the storage). On realistically
    # scaled instances it can over-pin into a singular reduced KKT — then
    # fall back to the feasible-start ratio-test active set, which keeps
    # the iterate feasible by construction and cannot go inconsistent.
    try:
        x, active = _solve_period_qp(H, q, C, d_eq, lb, ub)
    except np.linalg.LinAlgError:
        if np.any(demand < -1e-9) or np.any(demand > cap.sum() + 1e-9):
            raise ValueError(
                "storage dispatch infeasible: demand outside generator range")
        x0 = np.zeros(n)
        share = cap / max(float(cap.sum()), 1e-12)
        soc0 = min(soc_init, soc_max)
        for t in range(T):
            for g in range(G):
                x0[ip(t, P0 + g)] = demand[t] * share[g]
            x0[ip(t, SOC)] = soc0
        x, active = _solve_box_eq_qp_primal(H, q, C, d_eq, lb, ub, x0)
    x = np.clip(x, lb, ub)

    dispatch = np.zeros((G, T))
    charge = np.zeros(T)
    discharge = np.zeros(T)
    soc = np.zeros(T)
    for t in range(T):
        for g in range(G):
            dispatch[g, t] = x[ip(t, P0 + g)]
        charge[t] = x[ip(t, CH)]
        discharge[t] = x[ip(t, DIS)]
        soc[t] = x[ip(t, SOC)]

    # ---- Sensitivities on the frozen active set ----
    free = active == 0
    idx_f = np.where(free)[0]
    kf = idx_f.size
    Hff = H[np.ix_(idx_f, idx_f)]
    Cf = C[:, idx_f]
    KKT = np.zeros((kf + m_eq, kf + m_eq))
    KKT[:kf, :kf] = Hff
    KKT[:kf, kf:] = Cf.T
    KKT[kf:, :kf] = Cf
    try:
        KKT_inv = np.linalg.inv(KKT)
    except np.linalg.LinAlgError:
        # Degenerate optimum (an equality row whose every variable is
        # pinned, e.g. via the dual-solver fallback). Group inverse gives
        # the correct one-sided derivatives for consistent perturbations.
        KKT_inv = np.linalg.pinv(KKT)

    def solve_sens(rhs_top: np.ndarray, rhs_bot: np.ndarray,
                   dx_fixed: np.ndarray) -> np.ndarray:
        sol = KKT_inv @ np.concatenate([rhs_top, rhs_bot])
        dx = dx_fixed.copy()
        dx[idx_f] = sol[:kf]
        return dx

    at_ub = active > 0   # pinned at upper bound (cap / plim / soc_max)
    # Map dispatch flat index (g, t) → x index.
    disp_idx = np.array([ip(t, P0 + g) for g in range(G) for t in range(T)])
    # (Row order matches reshape((G, T)) row-major over (g, t).)

    d_mc = np.zeros((G * T, G))
    d_cap = np.zeros((G * T, G))
    d_dem = np.zeros((G * T, T))
    d_soc0 = np.zeros(G * T)

    # ----- d/d mc_j ----- (∂q at every gen-j var across periods)
    for j in range(G):
        rt = np.zeros(kf)
        for ii, fi in enumerate(idx_f):
            t_blk = fi // blk
            within = fi - t_blk * blk
            if within == P0 + j:
                rt[ii] = -1.0
        dx = solve_sens(rt, np.zeros(m_eq), np.zeros(n))
        d_mc[:, j] = dx[disp_idx]

    # ----- d/d capacity_j ----- (gen-j upper bound, per period if pinned)
    for j in range(G):
        dx_fixed = np.zeros(n)
        rt = np.zeros(kf)
        rb = np.zeros(m_eq)
        any_pin = False
        for t in range(T):
            vj = ip(t, P0 + j)
            if at_ub[vj]:
                any_pin = True
                dx_fixed[vj] = 1.0
                rt = rt - H[np.ix_(idx_f, [vj])][:, 0]
                rb = rb - C[:, vj]
        if any_pin:
            dx = solve_sens(rt, rb, dx_fixed)
        else:
            dx = np.zeros(n)
        d_cap[:, j] = dx[disp_idx]

    # ----- d/d demand_{t'} ----- (RHS of balance row t')
    for tp in range(T):
        rb = np.zeros(m_eq)
        rb[tp] = 1.0
        dx = solve_sens(np.zeros(kf), rb, np.zeros(n))
        d_dem[:, tp] = dx[disp_idx]

    # ----- d/d soc_init ----- (RHS of SOC-continuity row for t=0)
    rb = np.zeros(m_eq)
    rb[T + 0] = 1.0
    dx = solve_sens(np.zeros(kf), rb, np.zeros(n))
    d_soc0 = dx[disp_idx]

    # ----- d/d η_c and d/d η_d (constraint-MATRIX sensitivities) -----
    # η enters the SOC rows' coefficients, so differentiating the KKT
    # gives BOTH a stationarity term −(∂Cᵀ/∂θ)λ (needs the equality
    # duals λ) and an equality term −(∂C/∂θ)x. Recover λ on the frozen
    # active set by re-solving the reduced KKT with the ORIGINAL rhs
    # (H is diagonal ⇒ no H_fx cross term in rhs_top).
    idx_x = np.where(~free)[0]
    rhs0_top = -q[idx_f]
    rhs0_bot = d_eq - (C[:, idx_x] @ x[idx_x] if idx_x.size else 0.0)
    lam = (KKT_inv @ np.concatenate([rhs0_top, rhs0_bot]))[kf:]

    # Locate free CH / DIS vars once: maps free-row index → period t.
    free_ch = [(ii, fi // blk) for ii, fi in enumerate(idx_f)
               if fi - (fi // blk) * blk == CH]
    free_dis = [(ii, fi // blk) for ii, fi in enumerate(idx_f)
                if fi - (fi // blk) * blk == DIS]

    # θ = η_c: ∂C[T+t, ip(t,CH)]/∂η_c = −1
    #   rhs_top|free CH_t = +λ[T+t];  rhs_bot[T+t] = +ch_t  (pinned ch
    #   included — its stationarity row is outside the reduced system).
    soc_idx = np.array([ip(t, SOC) for t in range(T)])
    rt = np.zeros(kf)
    for ii, t_blk in free_ch:
        rt[ii] = lam[T + t_blk]
    rb = np.zeros(m_eq)
    rb[T:] = charge
    dx_eta_c = solve_sens(rt, rb, np.zeros(n))
    d_eta_c = dx_eta_c[disp_idx]
    d_soc_eta_c = dx_eta_c[soc_idx]

    # θ = η_d: ∂C[T+t, ip(t,DIS)]/∂η_d = −1/η_d²
    rt = np.zeros(kf)
    for ii, t_blk in free_dis:
        rt[ii] = lam[T + t_blk] / eta_d**2
    rb = np.zeros(m_eq)
    rb[T:] = discharge / eta_d**2
    dx_eta_d = solve_sens(rt, rb, np.zeros(n))
    d_eta_d = dx_eta_d[disp_idx]
    d_soc_eta_d = dx_eta_d[soc_idx]

    return StorageDispatchSolution(
        dispatch=dispatch,
        charge=charge,
        discharge=discharge,
        soc=soc,
        d_dispatch_d_mc=d_mc,
        d_dispatch_d_capacity=d_cap,
        d_dispatch_d_demand=d_dem,
        d_dispatch_d_soc_init=d_soc0,
        d_dispatch_d_charge_eff=d_eta_c,
        d_dispatch_d_discharge_eff=d_eta_d,
        d_soc_d_charge_eff=d_soc_eta_c,
        d_soc_d_discharge_eff=d_soc_eta_d,
    )


@dataclass
class StorageDispatchLayer:
    """Stateful OO wrapper around the multi-period storage dispatch.

    Mirrors :class:`MultiBusDispatchLayer`: :meth:`forward` caches the
    solution + Jacobians; :meth:`backward` pulls ``dL/dθ`` from
    ``dL/d dispatch`` without re-solving. Torch-free.
    """
    ridge: float = 1e-2
    _sol: StorageDispatchSolution | None = field(
        default=None, init=False, repr=False)

    def forward(self, problem: StorageDispatchProblem) -> np.ndarray:
        prob = StorageDispatchProblem(
            marginal_cost=problem.marginal_cost,
            capacity=problem.capacity,
            demand=problem.demand,
            charge_eff=problem.charge_eff,
            discharge_eff=problem.discharge_eff,
            power_limit=problem.power_limit,
            soc_max=problem.soc_max,
            soc_init=problem.soc_init,
            ridge=self.ridge,
        )
        sol = solve_storage_dispatch_with_sensitivities(prob)
        self._sol = sol
        return sol.dispatch

    def backward(self, grad_dispatch: np.ndarray) -> dict[str, np.ndarray | float]:
        """Return ``dL/d{mc, capacity, demand, soc_init, η_c, η_d}``.

        ``grad_dispatch`` has shape ``(G, T)`` and is flattened row-major
        to match the Jacobian layout.
        """
        if self._sol is None:
            raise RuntimeError("call forward() before backward()")
        g = np.asarray(grad_dispatch, dtype=float).reshape(-1)
        s = self._sol
        return {
            "marginal_cost": s.d_dispatch_d_mc.T @ g,
            "capacity": s.d_dispatch_d_capacity.T @ g,
            "demand": s.d_dispatch_d_demand.T @ g,
            "soc_init": float(s.d_dispatch_d_soc_init @ g),
            "charge_eff": float(s.d_dispatch_d_charge_eff @ g),
            "discharge_eff": float(s.d_dispatch_d_discharge_eff @ g),
        }


# ---------------------------------------------------------------------------
# Phase 12.2 — SMOOTHED differentiable commitment surrogate
# ---------------------------------------------------------------------------
#
# Exact unit commitment is a binary on/off decision u ∈ {0, 1}, which is
# NON-DIFFERENTIABLE — there is no gradient of a step function, so it
# cannot be embedded in a gradient-based learning loop. The standard
# remedy (e.g. soft/relaxed UC in differentiable-OPF and learning-to-
# commit work) is to replace the hard threshold with a SMOOTH surrogate:
#
#     u_soft = σ(k · (signal − τ))            (logistic / sigmoid)
#
# where ``signal`` is the committing driver (here: required dispatch as a
# fraction of capacity, or a price headroom), ``τ`` is a learnable
# threshold, and ``k`` is the sharpness (k→∞ recovers the hard step).
# ``u_soft`` is smooth in (signal, τ, k) so its gradient exists
# everywhere. The committed output is then ``p = u_soft · capacity`` (or
# clamped via softplus), which lets a fitting loss flow gradients back to
# the threshold/sharpness. We document this as the standard smoothed
# relaxation: it is NOT exact binary UC, it is its differentiable
# surrogate, and as k→∞ it converges to the binary decision.


def smooth_commitment(
    signal: np.ndarray,
    threshold: float,
    sharpness: float = 10.0,
) -> np.ndarray:
    """Sigmoid-smoothed on/off commitment surrogate ``σ(k·(signal−τ))``.

    ``signal`` is the per-(unit, period) committing driver; ``threshold``
    (τ) and ``sharpness`` (k > 0) are scalars. Returns values in (0, 1).
    Smooth and differentiable in all arguments — the standard relaxation
    of the non-differentiable binary on/off decision; ``sharpness→∞``
    recovers the hard step.
    """
    s = np.asarray(signal, dtype=float)
    z = float(sharpness) * (s - float(threshold))
    # Clip the logit to avoid exp overflow; ±40 saturates σ to 0/1 in f64.
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SmoothCommitmentLayer:
    """Differentiable smoothed-commitment layer with analytic gradients.

    Forward: ``u = σ(k·(signal − τ))``, committed power ``p = u·capacity``.
    Backward: returns ``dp/dτ`` and ``dp/dk`` (and ``dp/dsignal``) in
    closed form via the logistic derivative ``σ' = σ(1−σ)``. Torch-free;
    mirrors the OO ``forward``/``backward`` idiom of the QP layers.
    """
    sharpness: float = 10.0
    _u: np.ndarray | None = field(default=None, init=False, repr=False)
    _signal: np.ndarray | None = field(default=None, init=False, repr=False)
    _cap: np.ndarray | float | None = field(default=None, init=False, repr=False)
    _threshold: float | None = field(default=None, init=False, repr=False)

    def forward(
        self,
        signal: np.ndarray,
        threshold: float,
        capacity: np.ndarray | float = 1.0,
    ) -> np.ndarray:
        s = np.asarray(signal, dtype=float)
        u = smooth_commitment(s, threshold, self.sharpness)
        self._u = u
        self._signal = s
        self._cap = capacity
        self._threshold = float(threshold)
        return u * np.asarray(capacity, dtype=float)

    def backward(
        self, grad_out: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        """Return ``dL/d{threshold, sharpness, signal}`` given dL/dp.

        Uses ``σ' = σ(1−σ)``:
            dp/dτ      = cap · (−k) · σ'
            dp/dk      = cap · (signal − τ) · σ'
            dp/dsignal = cap · k · σ'
        """
        if self._u is None:
            raise RuntimeError("call forward() before backward()")
        g = np.asarray(grad_out, dtype=float)
        u = self._u
        sigp = u * (1.0 - u)
        cap = np.asarray(self._cap, dtype=float)
        k = self.sharpness
        s = self._signal
        tau = self._threshold
        dp_dtau = cap * (-k) * sigp
        dp_dk = cap * (s - tau) * sigp
        dp_dsignal = cap * k * sigp
        return {
            "threshold": float(np.sum(g * dp_dtau)),
            "sharpness": float(np.sum(g * dp_dk)),
            "signal": g * dp_dsignal,
        }


@dataclass
class CommitmentFitResult:
    """Outcome of :func:`fit_commitment_threshold`."""
    threshold: float
    history: list[float]            # loss per iteration
    threshold_history: list[float]  # parameter per iteration
    n_iter: int


def fit_commitment_threshold(
    signal: np.ndarray,
    observed_committed: np.ndarray,
    capacity: np.ndarray | float = 1.0,
    *,
    sharpness: float = 10.0,
    lr: float = 0.05,
    n_iter: int = 500,
    threshold_init: float = 0.5,
    tol: float = 1e-12,
) -> CommitmentFitResult:
    """Fit the commitment threshold τ by gradient descent on the surrogate.

    Minimises ``½‖p(τ) − observed‖²`` where ``p(τ) = capacity ·
    σ(k·(signal − τ))`` using the analytic ``dp/dτ`` from
    :class:`SmoothCommitmentLayer`. Pure numpy. The smoothed-commitment
    gradient is finite everywhere (logistic), so the loss decreases
    smoothly — verified in the smoke test.
    """
    s = np.asarray(signal, dtype=float)
    obs = np.asarray(observed_committed, dtype=float)
    layer = SmoothCommitmentLayer(sharpness=sharpness)
    tau = float(threshold_init)
    loss_hist: list[float] = []
    tau_hist: list[float] = []

    for _ in range(n_iter):
        p = layer.forward(s, tau, capacity)
        resid = p - obs
        loss = 0.5 * float(np.sum(resid * resid))
        grads = layer.backward(resid)  # dL/d? with dL/dp = resid
        grad_tau = grads["threshold"]
        loss_hist.append(loss)
        tau_hist.append(tau)
        new_tau = tau - lr * grad_tau
        if abs(new_tau - tau) < tol:
            tau = new_tau
            break
        tau = new_tau

    return CommitmentFitResult(
        threshold=float(tau),
        history=loss_hist,
        threshold_history=tau_hist,
        n_iter=len(loss_hist),
    )


# ---------------------------------------------------------------------------
# N_En_Phase 20 — differentiable CAPACITY EXPANSION (Paper 2)
# ---------------------------------------------------------------------------
#
# The layers above differentiate *dispatch* w.r.t. operating parameters
# (mc, demand, soc_init) and treat capacity as a fixed bound. Capacity
# EXPANSION is the bilevel design problem: choose generator capacities to
# minimise operating cost, where the operating cost is itself the optimum
# of a lower-level dispatch QP. This extends Degleris et al., "Gradient
# Methods for Scalable Differentiable Optimal Power Flow" (2024) from
# differentiating dispatch to differentiating the *operational optimum*
# w.r.t. the design (capacity) variables — the gradient a planner needs
# to do gradient-descent grid design or to learn component parameters
# from observed dispatch.
#
# Lower level (for fixed capacities ``cap``):
#
#     p*(cap) = argmin_p  Σ_t Σ_g mc_g·p_{g,t} + (ridge/2)·‖p‖²
#               s.t.  (balance, per period t)  Σ_g p_{g,t} = d_t
#                     0 ≤ p_{g,t} ≤ cap_g                  (cap bound)
#
# Upper-level operating cost:  J(cap) = Σ_t Σ_g mc_g · p*_{g,t}(cap).
#
# Capacity enters ONLY as the per-generator upper bound on dispatch, so
# its gradient flows through the *active upper-bound multipliers*: a unit
# of extra capacity moves a generator that is pinned at its cap, and the
# freed dispatch redistributes through the (frozen) active set via the KKT
# system — exactly the ``d_dispatch_d_capacity`` block already derived in
# ``solve_storage_dispatch_with_sensitivities`` /
# ``solve_multibus_dispatch_with_sensitivities``. We REUSE ``_solve_period_qp``
# (active-set core) for the forward solve and the same frozen-active-set
# KKT differentiation (implicit-function theorem; Amos & Kolter 2017 OptNet;
# Agrawal et al. 2019; Degleris et al. 2024) for the backward.
#
# The total operating-cost gradient is, by the chain rule,
#
#     dJ/dcap_j = Σ_{g,t} mc_g · (d p*_{g,t} / d cap_j),
#
# i.e. the operating-cost gradient is the dispatch-Jacobian contracted
# with the marginal-cost vector. (There is NO explicit ∂J/∂cap term: cap
# does not appear in the objective, only in the constraint set — so the
# entire dependence is through p*.) This is the chief verification target:
# the analytic dJ/dcap must match finite differences on J(cap).
#
# A note on the bound-active kink. cap_j only has gradient where generator
# j is pinned at its cap in the optimum (the upper-bound multiplier is
# active). When j is strictly interior, dp*/dcap_j = 0 (extra headroom is
# unused) — this is correct and matches FD AWAY from the kink where a gen
# is *exactly* at cap. We document this and test at smooth points (a gen
# either strictly pinned or strictly interior, never exactly transitioning).


@dataclass
class CapacityExpansionProblem:
    """A tiny single-bus multi-period capacity-expansion dispatch instance.

    The lower-level dispatch QP for fixed capacities. ``capacity`` is the
    DESIGN variable being differentiated; everything else parameterises
    the operational layer.

    Attributes:
        marginal_cost: ``(G,)`` linear operating cost per unit dispatch.
        capacity: ``(G,)`` per-generator capacity (the design variable;
            enters as the dispatch upper bound ``p ≤ cap``).
        demand: ``(T,)`` per-period demand (must be met each period).
        ridge: strict-convexity regulariser (must be > 0).
    """
    marginal_cost: np.ndarray
    capacity: np.ndarray
    demand: np.ndarray
    ridge: float = 1e-2


@dataclass
class CapacityExpansionSolution:
    """Forward dispatch + analytic design (capacity) gradients.

    ``dispatch`` is ``(G, T)``. ``operating_cost`` is the scalar
    Σ_{g,t} mc_g·p*_{g,t}.

        d_dispatch_d_capacity: (G·T, G)   d p*_{g,t} / d cap_j  (row-major g,t)
        d_cost_d_capacity:     (G,)        dJ/dcap_j  (operating-cost gradient)

    ``d_cost_d_capacity`` is the headline deliverable: the gradient of the
    operational optimum w.r.t. the design variables.
    """
    dispatch: np.ndarray
    operating_cost: float
    d_dispatch_d_capacity: np.ndarray
    d_cost_d_capacity: np.ndarray


def solve_capacity_expansion_with_sensitivities(
    problem: CapacityExpansionProblem,
) -> CapacityExpansionSolution:
    """Lower-level dispatch QP with analytic gradients of the operational
    optimum w.r.t. design (capacity).

    Solves the per-period balanced dispatch QP by the active-set core
    (:func:`_solve_period_qp`), then differentiates the KKT system on the
    frozen active set (implicit-function theorem; Amos & Kolter 2017;
    Degleris et al. 2024) to obtain exact ``d dispatch / d capacity`` and,
    by the chain rule with the cost gradient ``mc``, the operating-cost
    gradient ``dJ/dcapacity``. Pure numpy — no torch required.

    Capacity enters only as the dispatch upper bound, so the gradient flows
    through generators pinned at their capacity (active upper-bound
    multiplier). Interior generators contribute zero, as they must.
    """
    mc = np.asarray(problem.marginal_cost, dtype=float)
    cap = np.asarray(problem.capacity, dtype=float)
    demand = np.asarray(problem.demand, dtype=float)
    G = mc.shape[0]
    T = demand.shape[0]
    if cap.shape[0] != G:
        raise ValueError("capacity and marginal_cost must share length G")
    ridge = float(problem.ridge)
    if ridge <= 0:
        raise ValueError("ridge must be > 0")
    total_cap = float(np.sum(cap))
    if np.any(demand > total_cap + 1e-9):
        raise ValueError("infeasible: a period's demand exceeds total capacity")

    # Per-period block: [p_0..p_{G-1}]; one balance equality per period.
    # Periods decouple (no inter-temporal coupling here) but we keep the
    # stacked form so the capacity gradient ACCUMULATES across periods,
    # which is what an expansion-planning loss needs.
    blk = G
    n = blk * T

    def ip(t: int, g: int) -> int:
        return t * blk + g

    m_eq = T
    C = np.zeros((m_eq, n))
    d_eq = np.zeros(m_eq)
    for t in range(T):
        for g in range(G):
            C[t, ip(t, g)] = 1.0
        d_eq[t] = demand[t]

    lb = np.zeros(n)
    ub = np.zeros(n)
    for t in range(T):
        for g in range(G):
            lb[ip(t, g)] = 0.0
            ub[ip(t, g)] = cap[g]

    H = ridge * np.eye(n)
    q = np.zeros(n)
    for t in range(T):
        for g in range(G):
            q[ip(t, g)] = mc[g]

    # ---- Forward solve (active-set; dual Newton fallback) ----
    # The greedy primal active-set can over-pin its way into a singular
    # reduced KKT on realistically-scaled instances (see the multibus
    # solver's history); the dual semismooth Newton handles those. The
    # primal stays primary because it leaves degenerate (lb==ub) vars
    # "free at bound", which keeps the sensitivity KKT nonsingular for
    # callers like fit_component_params that zero out the storage.
    try:
        x, active = _solve_period_qp(H, q, C, d_eq, lb, ub)
    except np.linalg.LinAlgError:
        x, active, _ = _solve_period_qp_dual(
            np.diag(H).copy(), q, C, d_eq, lb, ub)
    x = np.clip(x, lb, ub)

    dispatch = np.zeros((G, T))
    for t in range(T):
        for g in range(G):
            dispatch[g, t] = x[ip(t, g)]
    operating_cost = float(sum(mc[g] * dispatch[g, t]
                               for g in range(G) for t in range(T)))

    # ---- Sensitivities on the frozen active set ----
    free = active == 0
    idx_f = np.where(free)[0]
    kf = idx_f.size
    Hff = H[np.ix_(idx_f, idx_f)]
    Cf = C[:, idx_f]
    KKT = np.zeros((kf + m_eq, kf + m_eq))
    KKT[:kf, :kf] = Hff
    KKT[:kf, kf:] = Cf.T
    KKT[kf:, :kf] = Cf
    KKT_inv = np.linalg.inv(KKT)

    def solve_sens(rhs_top: np.ndarray, rhs_bot: np.ndarray,
                   dx_fixed: np.ndarray) -> np.ndarray:
        sol = KKT_inv @ np.concatenate([rhs_top, rhs_bot])
        dx = dx_fixed.copy()
        dx[idx_f] = sol[:kf]
        return dx

    at_ub = active > 0  # gen var pinned at its capacity
    disp_idx = np.array([ip(t, g) for g in range(G) for t in range(T)])

    d_cap = np.zeros((G * T, G))
    # ----- d/d capacity_j ----- (gen-j upper bound; per period if pinned)
    for j in range(G):
        dx_fixed = np.zeros(n)
        rt = np.zeros(kf)
        rb = np.zeros(m_eq)
        any_pin = False
        for t in range(T):
            vj = ip(t, j)
            if at_ub[vj]:
                any_pin = True
                # x_{j,t} = cap_j ⇒ d x_{j,t}/d cap_j = 1 (pinned). Its
                # motion feeds the free system through H and C.
                dx_fixed[vj] = 1.0
                rt = rt - H[np.ix_(idx_f, [vj])][:, 0]
                rb = rb - C[:, vj]
        if any_pin:
            dx = solve_sens(rt, rb, dx_fixed)
        else:
            dx = np.zeros(n)
        d_cap[:, j] = dx[disp_idx]

    # ----- Operating-cost gradient via the chain rule -----
    # J = Σ_{g,t} mc_g·p*_{g,t};  dJ/dcap_j = Σ_{g,t} mc_g·(dp*_{g,t}/dcap_j).
    # Build the (G·T,) cost-weight vector aligned to the disp flat layout.
    mc_flat = np.array([mc[g] for g in range(G) for t in range(T)])
    d_cost_d_cap = d_cap.T @ mc_flat  # (G,)

    return CapacityExpansionSolution(
        dispatch=dispatch,
        operating_cost=operating_cost,
        d_dispatch_d_capacity=d_cap,
        d_cost_d_capacity=d_cost_d_cap,
    )


@dataclass
class CapacityExpansionLayer:
    """Differentiable capacity-expansion layer (design-variable gradients).

    Forward solves the lower-level dispatch QP for given capacities and
    returns the dispatch; backward returns ``dOperatingCost/dcapacity``
    and ``d dispatch / d capacity`` via implicit-function-theorem
    differentiation of the KKT system on the active set. Mirrors the
    OO ``forward``/``backward`` idiom of :class:`StorageDispatchLayer`
    and :class:`MultiBusDispatchLayer`. Torch-free.

    The headline output is :meth:`operating_cost_gradient` — the gradient
    of the operational optimum w.r.t. the design (capacity) variables,
    suitable for gradient-descent grid design.
    """
    ridge: float = 1e-2
    _sol: CapacityExpansionSolution | None = field(
        default=None, init=False, repr=False)

    def forward(self, problem: CapacityExpansionProblem) -> np.ndarray:
        prob = CapacityExpansionProblem(
            marginal_cost=problem.marginal_cost,
            capacity=problem.capacity,
            demand=problem.demand,
            ridge=self.ridge,
        )
        sol = solve_capacity_expansion_with_sensitivities(prob)
        self._sol = sol
        return sol.dispatch

    def operating_cost(self) -> float:
        """Scalar operating cost J(cap) of the cached forward solve."""
        if self._sol is None:
            raise RuntimeError("call forward() before operating_cost()")
        return self._sol.operating_cost

    def operating_cost_gradient(self) -> np.ndarray:
        """``dOperatingCost/dcapacity`` ``(G,)`` of the cached forward solve.

        The headline deliverable: gradient of the operational optimum
        w.r.t. the design (capacity) variables.
        """
        if self._sol is None:
            raise RuntimeError("call forward() before operating_cost_gradient()")
        return self._sol.d_cost_d_capacity

    def backward(self, grad_dispatch: np.ndarray) -> dict[str, np.ndarray]:
        """Return ``dL/d capacity`` given ``dL/d dispatch``.

        ``grad_dispatch`` has shape ``(G, T)`` and is flattened row-major
        to match the Jacobian layout. To get the operating-cost gradient
        directly, prefer :meth:`operating_cost_gradient`.
        """
        if self._sol is None:
            raise RuntimeError("call forward() before backward()")
        g = np.asarray(grad_dispatch, dtype=float).reshape(-1)
        s = self._sol
        return {"capacity": s.d_dispatch_d_capacity.T @ g}


# ---------------------------------------------------------------------------
# N_En_Phase 20 — fit component parameters from observed dispatch
# ---------------------------------------------------------------------------
#
# The mirror of fit_demand_elasticity / fit_commitment_threshold, but for
# *component* parameters (efficiency, marginal_cost, capacity): given
# observed dispatch, recover the parameters that best reproduce it by
# backprop through the differentiable dispatch layer. Returns an overrides
# dict consumable by ``add_component(..., **overrides)``.
#
# We currently support fitting ``marginal_cost`` (per gen) and ``capacity``
# (per gen) for the single-bus multi-period dispatch QP, since those are the
# parameters whose analytic Jacobians the capacity-expansion / storage
# layers expose (d dispatch / d mc and d dispatch / d cap). Fitting is a
# hand-written SGD loop using those Jacobians — no torch, no cvxpy.


@dataclass
class ComponentFitResult:
    """Outcome of :func:`fit_component_params`.

    ``overrides`` is the calibrated dict consumable by
    ``add_component(..., **overrides)``. ``params`` is the same content as
    a flat name→array map; ``history`` is the per-iteration fitting loss.
    """
    overrides: dict
    params: dict
    history: list[float]
    n_iter: int


def fit_component_params(
    template_or_overrides: dict,
    observed_dispatch: np.ndarray,
    demand: np.ndarray,
    *,
    fit: tuple[str, ...] = ("marginal_cost",),
    ridge: float = 1.0,
    lr: float = 1e-3,
    n_iter: int = 1000,
    tol: float = 1e-12,
) -> ComponentFitResult:
    """Gradient-fit component parameters by backprop through dispatch.

    Recovers component parameters (``marginal_cost`` and/or ``capacity``)
    that best reproduce ``observed_dispatch`` ``(G, T)`` for the given
    per-period ``demand`` ``(T,)``, by minimising
    ``½‖p_pred(θ) − observed‖²`` with the analytic dispatch Jacobians
    (implicit-function-theorem gradients). Mirrors
    :func:`fit_demand_elasticity` / :func:`fit_commitment_threshold`.

    ``template_or_overrides`` is a dict of starting component parameters
    (e.g. ``{"marginal_cost": [...], "capacity": [...]}``); it provides the
    initial values for the fitted params AND the held-fixed values for the
    rest. ``fit`` names which keys to optimise. Returns a
    :class:`ComponentFitResult` whose ``overrides`` is a dict consumable by
    ``add_component(..., **overrides)``. Pure numpy.
    """
    base = dict(template_or_overrides)
    if "marginal_cost" not in base or "capacity" not in base:
        raise ValueError(
            "template_or_overrides must contain 'marginal_cost' and 'capacity'")
    mc = np.asarray(base["marginal_cost"], dtype=float).copy()
    cap = np.asarray(base["capacity"], dtype=float).copy()
    demand = np.asarray(demand, dtype=float)
    observed = np.asarray(observed_dispatch, dtype=float)
    G = mc.shape[0]
    T = demand.shape[0]
    if observed.shape != (G, T):
        raise ValueError("observed_dispatch must have shape (G, T)")
    for name in fit:
        if name not in ("marginal_cost", "capacity"):
            raise ValueError(f"unsupported fit parameter: {name!r}")

    loss_hist: list[float] = []
    for _ in range(n_iter):
        prob = CapacityExpansionProblem(
            marginal_cost=mc, capacity=cap, demand=demand, ridge=ridge)
        sol = solve_capacity_expansion_with_sensitivities(prob)
        # We need d dispatch / d mc too; reuse the storage solver (no
        # storage activity ⇒ same single-bus dispatch QP) for that block.
        stor = solve_storage_dispatch_with_sensitivities(
            StorageDispatchProblem(
                marginal_cost=mc, capacity=cap, demand=demand,
                charge_eff=1.0, discharge_eff=1.0,
                power_limit=0.0, soc_max=0.0, soc_init=0.0, ridge=ridge))
        resid = sol.dispatch - observed  # (G, T)
        loss = 0.5 * float(np.sum(resid * resid))
        loss_hist.append(loss)
        rflat = resid.reshape(-1)  # (G*T,)

        max_step = 0.0
        if "marginal_cost" in fit:
            # dL/dmc_j = Σ resid · (d disp/d mc_j)
            grad_mc = stor.d_dispatch_d_mc.T @ rflat  # (G,)
            step = lr * grad_mc
            mc = mc - step
            max_step = max(max_step, float(np.max(np.abs(step))))
        if "capacity" in fit:
            grad_cap = sol.d_dispatch_d_capacity.T @ rflat  # (G,)
            step = lr * grad_cap
            cap = cap - step
            max_step = max(max_step, float(np.max(np.abs(step))))
        if max_step < tol:
            break

    overrides: dict = {}
    params: dict = {}
    # Always report the fitted keys; carry through any others from base.
    for name in fit:
        val = mc if name == "marginal_cost" else cap
        overrides[name] = val.tolist()
        params[name] = val
    for k, v in base.items():
        if k not in overrides:
            overrides[k] = v
            params[k] = np.asarray(v) if isinstance(v, (list, np.ndarray)) else v

    return ComponentFitResult(
        overrides=overrides,
        params=params,
        history=loss_hist,
        n_iter=len(loss_hist),
    )


# ---------------------------------------------------------------------------
# Torch-optional cvxpylayers hook
# ---------------------------------------------------------------------------

class TorchDispatchLayer:
    """Torch-optional wrapper around ``cvxpylayers.torch.CvxpyLayer``.

    Imports cleanly without torch so users can conditionally reach for
    it. At construction time we probe for ``torch`` and
    ``cvxpylayers`` and raise :class:`RuntimeError` if either is
    missing — the :class:`EconomicDispatchLayer` is the recommended
    fallback.

    The hook is a hook, not a maintained path — the pure-numpy layer
    covers the single-bus single-period case; multi-bus multi-period
    differentiable dispatch requires a real cvxpylayers integration
    that hasn't been merged yet. See
    ``DEFERRALS.md`` Phase 12 § for status.
    """

    def __init__(self) -> None:
        if not torch_available:
            raise RuntimeError(
                "TorchDispatchLayer requires PyTorch. "
                "Install nexus-energy[diff] or use EconomicDispatchLayer "
                "for the pure-numpy single-period path.")
        try:
            import cvxpylayers  # type: ignore  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "TorchDispatchLayer requires cvxpylayers. "
                "Install nexus-energy[diff] or use EconomicDispatchLayer "
                "for the pure-numpy single-period path.") from e
        raise RuntimeError(
            "TorchDispatchLayer is a hook for a future cvxpylayers path; "
            "the multi-bus multi-period differentiable solver is deferred. "
            "See DEFERRALS.md Phase 12 § for status.")
