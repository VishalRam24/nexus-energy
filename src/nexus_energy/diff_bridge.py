"""N_En_Phase 19 (inverse-calibration flagship, Phase A) — compose the
``from_pypsa`` import bridge with the multibus differentiable dispatch layer.

The diff layer (``diff.py``) was built on hand-rolled toy instances; this
module maps a real :class:`~nexus_energy.core.EnergySystem` — typically one
imported via ``from_pypsa(n, line_model="transport")`` — onto a
:class:`~nexus_energy.diff.MultiBusDispatchProblem`, so analytic
d-dispatch/d-parameter Jacobians become available for *real* networks
(PyPSA-Eur slices included).

Honest scope (fail loudly, never silently approximate):
  * dispatch (operations) only — every capacity must be fixed. Solve or pin
    expansion first; extendable components raise.
  * transport flow model only — DC-OPF links raise (re-import with
    ``from_pypsa(..., line_model="transport")``).
  * no storage — drop StorageUnits/Stores from the PyPSA network first.
  * lossless, cost-free, single-output links only.
  * LP-with-ridge: the ridge regulariser (mandatory for differentiability)
    shifts dispatch vs. the true LP by O(ridge); disclose it in results.

The CO₂-price parameter enters through effective marginal cost
``mc_eff[g] = mc[g] + price · emission[g]`` (emission in tCO₂/MWh_el, as
``from_pypsa`` already computes from carrier emissions / efficiency), so

    d dispatch / d price = d_dispatch_d_mc @ emission        (chain rule)

which :func:`fit_co2_price` uses for scalar Gauss-Newton recovery of a
hidden CO₂ price from observed dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .diff import (
    MultiBusDispatchProblem,
    MultiBusDispatchSolution,
    solve_multibus_dispatch_with_sensitivities,
)


@dataclass
class MultiBusBridge:
    """An EnergySystem rendered as a differentiable multibus problem.

    ``problem`` is ready for
    :func:`~nexus_energy.diff.solve_multibus_dispatch_with_sensitivities`.
    ``emission`` is per-generator tCO₂/MWh_el; ``mc_base`` is the marginal
    cost *without* the CO₂-price component, so
    ``problem.marginal_cost = mc_base + co2_price · emission``.
    """
    problem: MultiBusDispatchProblem
    gen_names: list
    bus_names: list
    line_names: list
    emission: np.ndarray
    mc_base: np.ndarray
    co2_price: float


def _reject(component, name, why):
    raise ValueError(
        f"diff_bridge: {component} {name!r} {why} — outside the multibus "
        f"diff layer's honest scope (dispatch-only, transport, lossless).")


def multibus_problem_from_system(
    system,
    *,
    co2_price: float = 0.0,
    ridge: float = 1e-2,
) -> MultiBusBridge:
    """Map an :class:`EnergySystem` onto a differentiable multibus problem.

    Args:
        system: EnergySystem (e.g. from ``from_pypsa(n, line_model="transport")``).
        co2_price: $/tCO₂ added into effective marginal costs via each
            generator's ``emission_factor``.
        ridge: strict-convexity regulariser passed through to the QP.

    Raises:
        ValueError on any component outside scope (see module docstring).
    """
    T = int(system._timesteps)
    if T < 1:
        raise ValueError("system has no timesteps")

    bus_names = [b.name for b in system._buses]
    bus_idx = {b.name: i for i, b in enumerate(system._buses)}
    B = len(bus_names)

    if system._storages:
        _reject("storage", system._storages[0].name,
                "is present (no storage in the per-period multibus QP); "
                "remove StorageUnits/Stores before from_pypsa")

    # ---- Generators ----
    gen_names, gen_bus, mc_base, cap, emission = [], [], [], [], []
    avail_rows = []
    for g in system._generators:
        if g.extendable:
            _reject("generator", g.name,
                    "is extendable (capacity must be fixed for dispatch "
                    "calibration; solve expansion first or set "
                    "p_nom_extendable=False)")
        if g.committable:
            _reject("generator", g.name, "is committable (MILP UC is not "
                    "differentiable here — future work)")
        if g.p_min > 0 or g.must_run:
            _reject("generator", g.name, "has p_min/must_run > 0")
        if g.heat_rate_segments:
            _reject("generator", g.name, "has PWL heat-rate segments")
        gen_names.append(g.name)
        gen_bus.append(bus_idx[g.bus.name])
        mc_base.append(float(g.marginal_cost))
        cap.append(float(g.capacity))
        emission.append(float(g.emission_factor))
        if g.carrier_factor is not None:
            cf = np.asarray(g.carrier_factor, dtype=float)
            if cf.shape != (T,):
                raise ValueError(
                    f"generator {g.name!r}: carrier_factor length {cf.shape} "
                    f"!= T={T}")
            avail_rows.append(np.clip(cf, 0.0, 1.0))
        else:
            avail_rows.append(np.ones(T))
    G = len(gen_names)
    if G == 0:
        raise ValueError("system has no generators")

    # ---- Links → lines (signed transport flow) ----
    line_names, line_from, line_to, line_limit, line_min = [], [], [], [], []
    for lk in system._links:
        if lk.model_type != "transport":
            _reject("link", lk.name,
                    f"has model_type={lk.model_type!r}; re-import with "
                    "from_pypsa(n, line_model='transport')")
        if lk.extendable:
            _reject("link", lk.name, "is extendable")
        if abs(lk.efficiency - 1.0) > 1e-9 or lk.loss or lk.loss_quadratic:
            _reject("link", lk.name, "is lossy (efficiency != 1)")
        if lk.marginal_cost:
            _reject("link", lk.name, "has a marginal cost")
        if lk.bus_to_2 is not None or lk.committable or lk.linepack_capacity:
            _reject("link", lk.name, "uses multi-output/UC/linepack features")
        line_names.append(lk.name)
        line_from.append(bus_idx[lk.bus_from.name])
        line_to.append(bus_idx[lk.bus_to.name])
        line_limit.append(float(lk.capacity))
        line_min.append(-float(lk.capacity) if lk.bidirectional else 0.0)

    # ---- Demand (B, T): sum of loads per bus ----
    demand = np.zeros((B, T))
    for ld in system._loads:
        amt = ld.amount
        row = (np.full(T, float(amt)) if np.isscalar(amt) or isinstance(amt, (int, float))
               else np.asarray(amt, dtype=float))
        if row.shape != (T,):
            raise ValueError(f"load {ld.name!r}: amount length != T={T}")
        demand[bus_idx[ld.bus.name]] += row

    mc_base = np.asarray(mc_base)
    emission = np.asarray(emission)
    problem = MultiBusDispatchProblem(
        gen_bus=np.asarray(gen_bus, dtype=int),
        marginal_cost=mc_base + co2_price * emission,
        capacity=np.asarray(cap),
        line_from=np.asarray(line_from, dtype=int),
        line_to=np.asarray(line_to, dtype=int),
        line_limit=np.asarray(line_limit),
        demand=demand,
        n_buses=B,
        ridge=ridge,
        availability=np.vstack(avail_rows),
        line_min=np.asarray(line_min),
    )
    return MultiBusBridge(
        problem=problem, gen_names=gen_names, bus_names=bus_names,
        line_names=line_names, emission=emission, mc_base=mc_base,
        co2_price=float(co2_price),
    )


def d_dispatch_d_co2_price(
    sol: MultiBusDispatchSolution, emission: np.ndarray,
) -> np.ndarray:
    """Chain rule ``d dispatch / d price = d_dispatch_d_mc @ emission``.

    Returns a ``(G·T,)`` vector in the same row-major (g, t) flattening as
    the solution Jacobians.
    """
    return sol.d_dispatch_d_mc @ np.asarray(emission, dtype=float)


@dataclass
class CO2FitResult:
    """Outcome of :func:`fit_co2_price`."""
    price: float
    history: list          # per-iteration (price, loss)
    n_solves: int          # forward+gradient solves consumed
    converged: bool


def fit_co2_price(
    system,
    observed_dispatch: np.ndarray,
    *,
    ridge: float = 1e-2,
    n_iter: int = 40,
    tol: float = 1e-8,
    price_bounds: tuple = (0.0, 1000.0),
    n_bracket: int = 7,
    verbose: bool = False,
) -> CO2FitResult:
    """Recover a hidden CO₂ price from observed dispatch.

    Two stages, both counted in ``n_solves``:

    1. **Bracket** — ``n_bracket`` cheap forward-only solves on a coarse
       grid over ``price_bounds``. Dispatch is piecewise-linear in the
       price, so the loss has *flat pieces* (price regimes where no
       emitting generator is marginal) on which any pure gradient method
       is stuck at birth; the bracket lands us on the informative piece.
    2. **Safeguarded Gauss-Newton** — minimise
       ``½‖dispatch(p) − observed‖²`` with the analytic
       ``d dispatch/d price`` Jacobian (one mc-block solve per iteration,
       no extra model evaluations). Steps that leave the bracket, or land
       on a zero-gradient piece, fall back to bisection of the bracket —
       the loss is piecewise-quadratic, so genuine GN steps converge in a
       handful of iterations.
    """
    observed = np.asarray(observed_dispatch, dtype=float)
    lo, hi = (float(price_bounds[0]), float(price_bounds[1]))
    history: list = []
    n_solves = 0

    def forward_loss(p: float) -> float:
        nonlocal n_solves
        b = multibus_problem_from_system(system, co2_price=p, ridge=ridge)
        s = solve_multibus_dispatch_with_sensitivities(
            b.problem, jacobians=())
        n_solves += 1
        r = (s.dispatch - observed).reshape(-1)
        return 0.5 * float(r @ r)

    # ---- stage 1: coarse bracket (forward-only solves) ----
    grid = np.linspace(lo, hi, n_bracket)
    g_loss = []
    for p in grid:
        l = forward_loss(float(p))
        g_loss.append(l)
        history.append((float(p), l))
        if verbose:
            print(f"  bracket p={p:8.3f} loss={l:.6e}")
    i = int(np.argmin(g_loss))
    a = float(grid[max(i - 1, 0)])
    b = float(grid[min(i + 1, n_bracket - 1)])
    price = float(grid[i])

    # ---- stage 2: safeguarded Gauss-Newton inside [a, b] ----
    converged = False
    for it in range(n_iter):
        br = multibus_problem_from_system(system, co2_price=price,
                                          ridge=ridge)
        sol = solve_multibus_dispatch_with_sensitivities(
            br.problem, jacobians=("mc",))
        n_solves += 1
        r = (sol.dispatch - observed).reshape(-1)
        loss = 0.5 * float(r @ r)
        history.append((price, loss))
        J = d_dispatch_d_co2_price(sol, br.emission)
        g = float(J @ r)
        h = float(J @ J)
        gn_ok = h > 1e-12
        new_price = price - g / h if gn_ok else None
        if new_price is None or not (a <= new_price <= b):
            # Flat piece or unstable extrapolation → bisect the bracket
            # toward the better side of the current point.
            new_price = 0.5 * (a + price) if (price - a) > (b - price) \
                else 0.5 * (price + b)
        if verbose:
            print(f"  it={it:02d} price={price:10.5f} loss={loss:.6e} "
                  f"next={new_price:10.5f} (gn={gn_ok})")
        # Shrink the safeguard bracket around the better point.
        if new_price < price:
            b = price
        else:
            a = price
        if abs(new_price - price) < tol or loss <= 1e-18:
            converged = True
            break
        price = float(new_price)

    # Final answer: the best evaluated price.
    price = min(history, key=lambda pl: pl[1])[0]
    return CO2FitResult(price=price, history=history,
                        n_solves=n_solves, converged=converged)
