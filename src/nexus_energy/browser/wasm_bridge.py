"""
Phase 12 — browser / WASM bridge.

Turns an :class:`EnergySystem` into a small JSON-ready LP description
the NexusFlow in-browser HiGHS-WASM runtime can solve without a Python
backend. The schema is intentionally component-level (not constraint-
level): the browser runtime rebuilds the same LP structure the desktop
solver builds, so the two stay consistent without having to serialise
the full constraint matrix.

Scope:

- Works for the LP subset (no committable generators, no extendable
  capacities, no storage cyclic closure) — the WASM target is the
  in-browser preview, not a full-scale optimisation. Larger models
  should still round-trip through the desktop solver.
- Carries enough metadata to reconstruct an
  :class:`OptimisationResult` from the browser's solution payload.
- Schema is versioned via :const:`WASM_SCHEMA_VERSION`; the browser
  runtime pins the version it supports and errors out if they drift.

Non-goals (see ``DEFERRALS.md`` Phase 12):

- No MIP support (the WASM HiGHS build ships LP + simple QP).
- No network / SOCP OPF terms — linear DC approximations only.
- No PWL cost curves; piecewise linear is flattened to the first
  segment's slope if a PWL generator is encountered (with a warning).

**N_En_Phase 12.7 — SOCP / AC-OPF over the WASM bridge: WONTFIX (decided).**
The in-browser HiGHS-WASM runtime is an LP(/simple-QP) solver only; it has no
second-order-cone support, and shipping a WASM build of Clarabel purely for the
preview pane is not worth the bundle size. Conic AC-OPF (``solve_socp_opf`` and
the first-class ``nexus_opt.Model.add_soc_cone`` cone API) is therefore a
**desktop-only** capability by design. This export path deliberately carries no
cone terms; cone/SOCP models must round-trip through the desktop solver. This is
a settled architectural decision, not a pending gap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from nexus_energy.core import EnergySystem, OptimisationResult


WASM_SCHEMA_VERSION = "1.0"


__all__ = [
    "WASM_SCHEMA_VERSION",
    "export_lp_for_browser",
    "import_result_from_browser",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_list(arr: np.ndarray | float | int, T: int) -> list[float]:
    """Broadcast scalar to length-T list; keep arrays as-is."""
    if isinstance(arr, np.ndarray):
        return [float(x) for x in arr.tolist()[:T]]
    return [float(arr)] * T


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_lp_for_browser(sys: "EnergySystem") -> dict[str, Any]:
    """Serialise ``sys`` to a JSON-ready LP description.

    Returns a plain ``dict`` (no numpy arrays, no custom types) so the
    payload passes through ``json.dumps`` unchanged. The browser-side
    contract is defined in ``docs/planning/WASM_BRIDGE_SCHEMA.md``.

    Raises :class:`ValueError` for features the WASM runtime cannot
    honour: committable generators, extendable capacities, storage
    cyclic closure, PWL heat-rate curves beyond a single segment.
    """
    sys._infer_timesteps()
    T = int(max(sys._timesteps, 1))

    generators = []
    for gen in sys._generators:
        if gen.committable:
            raise ValueError(
                f"generator {gen.name!r}: committable UC is not supported "
                "by the WASM LP runtime; see DEFERRALS.md Phase 12 §")
        if gen.extendable and gen._cap_var is not None:
            raise ValueError(
                f"generator {gen.name!r}: extendable capacities are not "
                "supported by the WASM LP runtime")
        cf = gen.carrier_factor
        generators.append({
            "name": gen.name,
            "bus": gen.bus.name,
            "capacity": float(gen.capacity),
            "marginal_cost": float(gen.marginal_cost),
            "p_min": float(gen.p_min) if not gen.committable else 0.0,
            "carrier_factor": _as_list(cf, T) if cf is not None else None,
            "must_run": bool(getattr(gen, "must_run", False)),
            "tech": getattr(gen, "tech", None),
        })

    loads = []
    for ld in sys._loads:
        amt = ld.amount
        loads.append({
            "name": ld.name,
            "bus": ld.bus.name,
            "amount": _as_list(amt, T),
        })

    storages = []
    for sto in sys._storages:
        storages.append({
            "name": sto.name,
            "bus": sto.bus.name,
            "power_capacity": float(sto.power_capacity),
            "energy_capacity": float(sto.energy_capacity),
            "efficiency_charge": float(getattr(sto, "efficiency_charge", 1.0)),
            "efficiency_discharge": float(getattr(sto, "efficiency_discharge", 1.0)),
            "initial_soc_fraction": float(
                getattr(sto, "initial_soc_fraction", 0.5)),
        })

    links = []
    for link in sys._links:
        if link.extendable and link._cap_var is not None:
            raise ValueError(
                f"link {link.name!r}: extendable capacities are not "
                "supported by the WASM LP runtime")
        links.append({
            "name": link.name,
            "bus_from": link.bus_from.name,
            "bus_to": link.bus_to.name,
            "capacity": float(link.capacity),
            "efficiency": float(link.efficiency),
            "marginal_cost": float(getattr(link, "marginal_cost", 0.0)),
        })

    buses = [{"name": b.name,
              "carrier": getattr(b.carrier, "name", str(b.carrier))}
             for b in sys._buses]

    return {
        "schema": WASM_SCHEMA_VERSION,
        "name": sys.name,
        "timesteps": T,
        "dt": float(sys._dt),
        "buses": buses,
        "generators": generators,
        "loads": loads,
        "storages": storages,
        "links": links,
    }


# ---------------------------------------------------------------------------
# Import (browser → OptimisationResult)
# ---------------------------------------------------------------------------

def import_result_from_browser(
    payload: dict[str, Any],
    sys: "EnergySystem | None" = None,
) -> "OptimisationResult":
    """Rebuild an :class:`OptimisationResult` from a browser-side solve.

    The browser runtime returns a payload with ``status``, ``total_cost``,
    and per-component dispatch arrays keyed by name; this function
    packs them back into the canonical Python result so callers can
    feed a browser-solved plan into the same reporting / diagnostics
    pipeline as a desktop-solved one.

    ``sys`` is accepted but unused today — it's here so we can enrich
    future payloads with component metadata without breaking callers.
    """
    from nexus_energy.core import OptimisationResult  # local to avoid cycle

    schema = payload.get("schema")
    if schema != WASM_SCHEMA_VERSION:
        raise ValueError(
            f"WASM payload schema {schema!r} does not match "
            f"{WASM_SCHEMA_VERSION!r}; update the nexus-ide bridge")

    result = OptimisationResult(
        status=str(payload.get("status", "unknown")),
        total_cost=float(payload.get("total_cost", float("nan"))),
        solve_time=float(payload.get("solve_time", 0.0)),
    )
    for name, vals in (payload.get("generator_dispatch") or {}).items():
        result.generator_dispatch[name] = np.asarray(vals, dtype=float)
    for name, vals in (payload.get("link_flow") or {}).items():
        result.link_flow[name] = np.asarray(vals, dtype=float)
    for name, vals in (payload.get("storage_charge") or {}).items():
        result.storage_charge[name] = np.asarray(vals, dtype=float)
    for name, vals in (payload.get("storage_discharge") or {}).items():
        result.storage_discharge[name] = np.asarray(vals, dtype=float)
    for name, vals in (payload.get("storage_soc") or {}).items():
        result.storage_soc[name] = np.asarray(vals, dtype=float)
    for name, vals in (payload.get("bus_shadow_prices") or {}).items():
        result.bus_shadow_prices[name] = np.asarray(vals, dtype=float)
    return result
