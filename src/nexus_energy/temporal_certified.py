"""
N_En_Phase 18.P2 — temporal decomposition with an optimality certificate.

Motivation (loophole-4 probe, 2026-06-10): CINDER-class MILP wall time
scales ~T^2.5 with horizon length, so solving K short blocks is an order
of magnitude cheaper than one monolithic solve. This module exploits that
WITHOUT giving up exactness semantics: like any MIP solver, it returns a
feasible solution plus a valid global lower bound and certifies
``(UB - LB)/UB <= gap`` — the same certificate a monolithic solver
produces when it stops at its gap tolerance.

Construction
------------
* **LB (valid bound, solved FIRST, optionally iterated)** — Lagrangian
  dual decomposition over the interior SOC boundaries. Each block is a
  RELAXATION of the full problem restricted to its window: interior
  initial SOC free in [soc_min, soc_max]·cap (``Storage.soc_initial_free``;
  block 1 keeps the globally pinned start), the t=0 ramp-from-zero charge
  dropped (``_ramp_cost_skip_t0``), and boundary prices λ
  (``soc_start_cost``/``soc_terminal_cost``) whose terms telescope to
  zero on any feasible full trajectory — so Σ_k (block dual bound) ≤
  full optimum for ARBITRARY λ. ``lb_rounds`` > 1 runs projected
  subgradient ascent on λ (the boundary supply/demand mismatch is the
  subgradient); the reported LB is the BEST round (max of valid bounds
  is valid). ``lb_workers`` > 1 solves the independent blocks of each
  round in parallel processes.

* **UB (feasible incumbent)** — solve blocks sequentially, handing each
  block its predecessor's terminal storage SOC as the pinned initial
  SOC. De-myopification, two modes:
  - ``ub_boundary="prices"`` (default): interior blocks PRICE terminal
    energy at -λ (no hard constraint, so no infeasibility risk); the
    λ·soc_end payment is subtracted back out post-solve, so the reported
    UB is the stitched trajectory's TRUE full-model cost.
  - ``ub_boundary="floors"``: hard terminal-SOC floors from the caller's
    guide (``Storage.soc_terminal_min``); a restriction — infeasible
    blocks retry unfloored.
  Boundary corrections make the cost exact: ramp (block t=0 charges
  |f[0]| vs the full model's |f[0]-f_prev|; add the difference) and
  conservative startup/shutdown (full model charges interior-boundary
  transitions; blocks don't — add the cost whenever the commitment
  state flips across a boundary).

Supported systems (guarded; ValueError otherwise): non-cyclic, non-LDS,
non-extendable storages; committable LINKS with min up/down ≤ 1; no hard
ramp limits; no committable generators; uniform snapshot weights and
durations. These cover dispatch-type production cases (e.g. CINDER);
lifting each guard is incremental future work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class TemporalCertifiedResult:
    """Outcome of a certified temporal-decomposition solve."""
    status: str               # "certified" | "gap_not_met" | "block_failed"
    objective: float          # UB: exact full-model cost of stitched solution
    lower_bound: float        # valid global lower bound (best Σ block bounds)
    gap: float                # (objective - lower_bound) / max(1, |objective|)
    gap_target: float
    n_blocks: int
    ub_wall: float
    lb_wall: float
    total_wall: float
    block_objectives: list = field(default_factory=list)
    block_lower_bounds: list = field(default_factory=list)
    lb_round_bounds: list = field(default_factory=list)   # Σ bound per λ round
    lambda_final: dict = field(default_factory=dict)      # name -> array(K-1)
    boundary_correction: float = 0.0
    # Stitched full-horizon trajectories (UB solution)
    generator_dispatch: dict = field(default_factory=dict)
    link_flow: dict = field(default_factory=dict)
    storage_charge: dict = field(default_factory=dict)
    storage_discharge: dict = field(default_factory=dict)
    storage_soc: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"TemporalCertifiedResult(status={self.status!r}, "
                f"objective={self.objective:.4f}, lb={self.lower_bound:.4f}, "
                f"gap={self.gap:.4%}, blocks={self.n_blocks}, "
                f"wall={self.total_wall:.1f}s)")


def _guard_supported(es) -> None:
    """Reject system features whose cross-block coupling v0 does not handle."""
    for sto in es._storages:
        if sto.cyclic:
            raise ValueError(
                f"temporal_certified: cyclic storage {sto.name!r} unsupported")
        if sto.extendable:
            raise ValueError(
                f"temporal_certified: extendable storage {sto.name!r} unsupported")
        if sto.long_duration:
            raise ValueError(
                f"temporal_certified: LDS storage {sto.name!r} unsupported")
    for gen in es._generators:
        if gen.committable:
            raise ValueError(
                f"temporal_certified: committable generator {gen.name!r} "
                "unsupported in v0 (committable links are fine)")
        if gen.extendable:
            raise ValueError(
                f"temporal_certified: extendable generator {gen.name!r} unsupported")
        if getattr(gen, "ramp_up_limit", None) is not None or \
                getattr(gen, "ramp_down_limit", None) is not None:
            raise ValueError(
                f"temporal_certified: hard ramp limit on {gen.name!r} unsupported")
    for link in es._links:
        if link.extendable:
            raise ValueError(
                f"temporal_certified: extendable link {link.name!r} unsupported")
        if getattr(link, "ramp_up_limit", None) is not None or \
                getattr(link, "ramp_down_limit", None) is not None:
            raise ValueError(
                f"temporal_certified: hard ramp limit on {link.name!r} unsupported")
        if link.committable and (link.min_up_time > 1 or link.min_down_time > 1):
            raise ValueError(
                f"temporal_certified: min up/down > 1 on {link.name!r} "
                "couples blocks; unsupported in v0")
    if es._snapshot_weights is not None:
        raise ValueError("temporal_certified: snapshot weights unsupported in v0")
    if es._snapshot_durations is not None:
        raise ValueError("temporal_certified: variable snapshot durations unsupported in v0")


def _link_status_edge(es, res, link):
    """(u[0], u[-1]) of a committable link from the raw solution."""
    raw = res._raw
    return (float(raw.value(link._status_vars[0])),
            float(raw.value(link._status_vars[-1])))


def _lb_block_task(system_factory, t0, t1, k, n_blocks,
                   lam_start, lam_terminal, start_hi, term_hi,
                   flow_prices, solve_kwargs):
    """Solve one relaxed LB block. Top-level so it pickles for workers.

    lam_start / lam_terminal: dict name -> price (or empty dicts).
    start_hi / term_hi: dict name -> reachability-envelope cap (abs MWh) on
    the free start SOC / the terminal SOC — constraints IMPLIED by the full
    problem (energy that could not physically be stored by that boundary),
    so adding them keeps every full solution's restriction feasible.
    flow_prices: linear boundary-flow prices under-approximating the relaxed
    cross-boundary ramp cost (rc·|Δf| ≥ rc·s·Δf, any |s| ≤ 1): keys
    "link_t0"/"link_term"/"sto_t0"/"sto_term" → {name: price}.
    Returns (k, lb_k, donor_soc_end, receiver_soc_start).
    """
    es = system_factory(t0, t1)
    fp = flow_prices or {}
    _skip_t0 = fp.get("_skip_t0", True)
    for link in es._links:
        v = fp.get("link_t0", {}).get(link.name)
        if v is not None:
            link.flow_t0_cost = float(v)
        v = fp.get("link_term", {}).get(link.name)
        if v is not None:
            link.flow_terminal_cost = float(v)
        v = fp.get("link_ramp_ref", {}).get(link.name)
        if v is not None:
            link.ramp_t0_reference = float(v)
        v = fp.get("link_term_vreb", {}).get(link.name)
        if v is not None:
            link.flow_terminal_v_rebate = (float(v[0]), float(v[1]))
    for sto in es._storages:
        if k > 0:
            sto.soc_initial_free = True
            hi = start_hi.get(sto.name)
            if hi is not None:
                sto.soc_initial_free_max = float(hi)
        ls = lam_start.get(sto.name)
        if ls is not None and k >= 1:
            sto.soc_start_cost = float(ls)
        lt = lam_terminal.get(sto.name)
        if lt is not None and k <= n_blocks - 2:
            sto.soc_terminal_cost = -float(lt)
        th = term_hi.get(sto.name)
        if th is not None and k <= n_blocks - 2:
            sto.soc_terminal_max = float(th)
        v = fp.get("sto_t0", {}).get(sto.name)
        if v is not None:
            sto.net_t0_cost = float(v)
        v = fp.get("sto_term", {}).get(sto.name)
        if v is not None:
            sto.net_terminal_cost = float(v)
        v = fp.get("sto_ramp_ref", {}).get(sto.name)
        if v is not None:
            sto.ramp_t0_reference = float(v)
        v = fp.get("sto_net_vreb", {}).get(sto.name)
        if v is not None:
            sto.net_terminal_v_rebate = (float(v[0]), float(v[1]))
        v = fp.get("soc_start_v", {}).get(sto.name)
        if v is not None and k > 0:
            sto.soc_start_v_cost = (float(v[0]), float(v[1]))
        v = fp.get("soc_term_vreb", {}).get(sto.name)
        if v is not None:
            sto.soc_terminal_v_rebate = (float(v[0]), float(v[1]))
    res = es.optimise(_ramp_cost_skip_t0=_skip_t0, **solve_kwargs)
    if res.status not in ("optimal", "time_limit"):
        return (k, None, {}, {})
    raw_gap = getattr(res._raw, "gap", None) or 0.0
    lb_k = res.total_cost - raw_gap * max(1.0, abs(res.total_cost))
    donor = {}
    receiver = {}
    for sto in es._storages:
        if k <= n_blocks - 2 and sto.name in res.storage_soc:
            donor[sto.name] = float(res.storage_soc[sto.name][-1])
        if k >= 1 and sto._soc_start_var is not None:
            receiver[sto.name] = float(res._raw.value(sto._soc_start_var))
    return (k, lb_k, donor, receiver)


def optimise_temporal_certified(
    system_factory: Callable[[int, int], "object"],
    total_steps: int,
    n_blocks: int = 4,
    gap: float = 1e-4,
    solve_kwargs: Optional[dict] = None,
    boundary_soc_min: Optional[dict] = None,
    boundary_soc_guide: Optional[dict] = None,
    boundary_prices: Optional[dict] = None,
    lb_blocks: Optional[int] = None,
    lb_rounds: int = 1,
    lb_workers: int = 1,
    # Proximal boundary V-terms: one-shot (non-iterated) use was MEASURED
    # 2026-06-11 to LOOSEN the CINDER bound (529 vs 629) — donors harvest
    # the concave rebates profitably — and the rebate binaries slow blocks
    # ~30x. Valid but counterproductive without consensus iteration; keep
    # OFF until the iterated version exists.
    lb_proximal: bool = False,
    lb_flow_prices: bool = False,
    lb_step_scale: float = 0.5,
    ub_boundary: str = "both",
    reachability_envelopes: bool = True,
    envelope_inflow: Optional[dict] = None,
    verbose: bool = False,
) -> TemporalCertifiedResult:
    """
    Solve a long-horizon system as K temporal blocks with a global
    optimality certificate.

    Args:
        system_factory: callable(t0, t1) -> EnergySystem over the window
            [t0, t1) (same contract as MPCController). Must slice every
            time-varying input consistently. Must be a picklable
            module-level callable when ``lb_workers`` > 1.
        total_steps: full horizon length T.
        n_blocks: number of (near-)equal blocks K.
        gap: certificate tolerance on (UB - LB)/max(1, |UB|).
        solve_kwargs: forwarded to every block ``optimise()`` call
            (threads, gap, time_limit, ...). The per-block MIP gap also
            loosens the certificate, so keep it well below the target.
        boundary_soc_min: optional hard floors for ``ub_boundary="floors"``
            — per storage name, an array of n_blocks-1 absolute-MWh values.
        boundary_soc_guide: optional guide SOC trajectory z at the UB
            boundaries (per storage name, array of n_blocks-1 absolute
            MWh). Enables the pin-dual λ harvest: each block re-solves as
            an LP with both boundaries pinned to z, and the terminal pin
            row's dual — the marginal cost of delivering z — becomes the
            Lagrangian boundary price. Future-aware (z comes from a
            full-horizon guide), unlike a free-terminal block whose
            terminal dual is the myopic ≈0.
        boundary_prices: initial λ — per storage name, an array of
            lb_blocks-1 prices at the LB pass's internal boundaries (e.g.
            coarse-LP water values). Valid bound for ARBITRARY λ; rounds
            refine it.
        lb_blocks: block count for the LB pass (default: n_blocks). Each
            relaxed boundary leaks ≈ |λ-λ*|·capacity, so FEWER, LONGER LB
            blocks give a tighter bound at higher per-block cost — they
            run in parallel, so wall often barely moves.
        lb_rounds: subgradient ascent rounds on λ. The reported LB is the
            best round; λ from the best round also prices the UB blocks.
        lb_workers: process-parallelism for each LB round's independent
            blocks (factory must be picklable).
        lb_step_scale: multiplicative λ step — each round scales prices by
            (1 + lb_step_scale·ĝ) with ĝ the per-storage normalized boundary
            mismatch in [-1, 1]; halved automatically when a round worsens
            the bound (restarting from the best λ).
        ub_boundary: "prices" (soft terminal-energy pricing at the best λ,
            exact post-hoc cost adjustment), "floors" (hard terminal-SOC
            minima from boundary_soc_min, infeasible blocks retry
            unfloored), "both" (default), or "none".
        reachability_envelopes: cap each LB block's free start SOC and
            terminal SOC by what the full problem could physically have
            stored by that boundary (s0 + η·P_charge·hours, no-inflow
            storages only unless envelope_inflow supplies a max rate).
            Implied by the full constraints → bound-tightening, never
            validity-affecting.
        envelope_inflow: optional dict name -> max inflow rate (MW) to
            include inflow storages in the envelopes. Storages with inflow
            and no entry here are left un-enveloped (still valid).
        verbose: print per-block progress.
    """
    if ub_boundary not in ("prices", "floors", "both", "none"):
        raise ValueError(
            "ub_boundary must be 'prices', 'floors', 'both', or 'none'")
    solve_kwargs = dict(solve_kwargs or {})
    t_start = time.perf_counter()
    edges = np.linspace(0, total_steps, n_blocks + 1).astype(int)
    K_lb = int(lb_blocks) if lb_blocks else n_blocks
    edges_lb = np.linspace(0, total_steps, K_lb + 1).astype(int)
    n_bnd = K_lb - 1

    # Probe block: feature guards + storage capacities for normalization.
    probe = system_factory(int(edges[0]), int(edges[1]))
    _guard_supported(probe)
    e_caps = {s.name: max(float(s.energy_capacity), 1e-9)
              for s in probe._storages}

    # λ state: name -> array(n_bnd). Start from caller prices (or zeros).
    lam: dict[str, np.ndarray] = {}
    for name in e_caps:
        if boundary_prices is not None and boundary_prices.get(name) is not None:
            lam[name] = np.asarray(boundary_prices[name], dtype=float).copy()
        else:
            lam[name] = np.zeros(n_bnd)
    _pos = np.concatenate([v[v > 0] for v in lam.values()]) if lam else np.array([])
    lam_seed = float(np.median(_pos)) if _pos.size else 1.0  # growth floor

    # Reachability envelopes: an upper cap on the SOC the FULL problem can
    # possibly hold at each interior boundary, from the pinned global start
    # plus maximal charging (and supplied inflow rates). Self-discharge only
    # lowers SOC, so ignoring it keeps the cap valid.
    env_hi: dict[str, np.ndarray] = {}
    if reachability_envelopes:
        dt_h = float(getattr(probe, "_dt", 1.0) or 1.0)
        for sto in probe._storages:
            if sto.inflow is not None and (
                    envelope_inflow is None
                    or envelope_inflow.get(sto.name) is None):
                continue  # unknown future inflow → leave un-enveloped
            rate_in = (float(sto.pump_capacity)
                       if getattr(sto, "pump_capacity", None) is not None
                       else float(sto.power_capacity))
            rate_in *= float(sto.efficiency_charge)
            if envelope_inflow is not None and envelope_inflow.get(sto.name):
                rate_in += float(envelope_inflow[sto.name])
            s0 = float(sto.soc_initial) * float(sto.energy_capacity)
            hi_cap = float(sto.soc_max) * float(sto.energy_capacity)
            env_hi[sto.name] = np.minimum(
                hi_cap, s0 + rate_in * dt_h * edges_lb[1:-1].astype(float))

    # ---------------- λ harvest: pin-dual boundary prices -----------------
    # Re-solve each UB block as an LP (integers relaxed, storages forced to
    # full mode) with BOTH boundaries pinned to the guide trajectory z:
    # start via soc_initial, terminal via a soc_fixed equality. The pin
    # row's dual is the marginal cost of delivering z — a FUTURE-AWARE
    # boundary price because z comes from a full-horizon guide. (The
    # free-terminal harvest variant was refuted 2026-06-11: an isolated
    # block's terminal dual is the myopic value ≈ 0.) ANY λ keeps the LB
    # valid; these make it tight when z tracks the optimal trajectory.
    if boundary_soc_guide is not None and n_blocks > 1:
        hv_t0 = time.perf_counter()
        lam_harvest: dict[str, np.ndarray] = {
            nm: np.full(n_blocks - 1, np.nan) for nm in e_caps}
        for k in range(n_blocks - 1):
            t0, t1 = int(edges[k]), int(edges[k + 1])
            es = system_factory(t0, t1)
            blk_T = t1 - t0
            for sto in es._storages:
                sto.storage_model = "full"
                z = boundary_soc_guide.get(sto.name)
                if z is None:
                    continue
                lo = sto.soc_min * sto.energy_capacity
                hi = sto.soc_max * sto.energy_capacity
                if k > 0:
                    zin = min(max(float(z[k - 1]), lo), hi)
                    frac = zin / max(sto.energy_capacity, 1e-12)
                    sto.soc_initial = float(
                        min(max(frac, sto.soc_min), sto.soc_max))
                sto.soc_fixed = {
                    blk_T - 1: float(min(max(float(z[k]), lo), hi))}
            try:
                res = es.optimise(_relax_integers=True,
                                  mip_strategy="mip_only", **solve_kwargs)
            except Exception:
                res = None
            if res is not None and res.status == "optimal":
                for (nm, t_pin), dv in res.soc_fixed_duals.items():
                    if t_pin == blk_T - 1 and np.isfinite(dv):
                        # SIGNED: negative boundary values are real (e.g. a
                        # must-take-inflow storage where delivering more SOC
                        # avoids curtailment cost). abs() here was measured
                        # 2026-06-11 to invert biomass's -5 curtailment price
                        # into a fake hoarding reward (LB -11,284).
                        lam_harvest[nm][k] = float(dv)
            elif verbose:
                print(f"[temporal_certified] pin-dual harvest block {k + 1}"
                      " infeasible/failed — keeping prior λ at that boundary")
        # Harvested λ (nearest UB boundary) overrides the initial prices —
        # but only inside a trust region around the coarse price scale. A
        # pin dual far outside it (e.g. biomass -5 = curtailment economics
        # vs coarse +0.04) signals donor/receiver DISAGREEMENT at an
        # off-optimal guide z; adopting either extreme on a huge-capacity
        # storage collapses the bound (measured ±11k, 2026-06-11). True
        # reconciliation needs consensus iteration; until then, coarse λ is
        # the safer price at disputed boundaries.
        _cap = 5.0 * max(
            (float(np.max(np.abs(v))) for v in lam.values() if v.size),
            default=1.0) + 1e-6
        for nm in lam:
            for i in range(n_bnd):
                jj = int(np.argmin(np.abs(edges[1:-1] - edges_lb[i + 1]))) \
                    if n_blocks > 1 else 0
                v = lam_harvest[nm][min(jj, n_blocks - 2)]
                if np.isfinite(v) and abs(v) <= _cap:
                    lam[nm][i] = v
                elif np.isfinite(v) and verbose:
                    print(f"[temporal_certified] λ harvest: {nm} boundary "
                          f"{i} rejected ({v:+.3f} outside ±{_cap:.3f}) — "
                          "keeping coarse price")
        if verbose:
            print(f"[temporal_certified] pin-dual λ harvest: "
                  f"{time.perf_counter() - hv_t0:.1f}s")

    # ---------------- UB pass: sequential stitch ----------------
    ub_t0 = time.perf_counter()
    obj_sum = 0.0
    correction = 0.0
    prev_soc: dict[str, float] = {}
    prev_flow: dict[str, float] = {}
    prev_net: dict[str, float] = {}
    prev_u: dict[str, float] = {}
    block_objs: list[float] = []
    stitched: dict[str, dict[str, list]] = {
        "generator_dispatch": {}, "link_flow": {}, "storage_charge": {},
        "storage_discharge": {}, "storage_soc": {}}

    for k in range(n_blocks):
        t0, t1 = int(edges[k]), int(edges[k + 1])
        es = system_factory(t0, t1)
        if k > 0:
            for sto in es._storages:
                frac = prev_soc[sto.name] / max(sto.energy_capacity, 1e-12)
                sto.soc_initial = float(min(max(frac, sto.soc_min), sto.soc_max))
                # Ramp continuity handoff: t=0 ramp priced against the
                # previous block's final net flow, so boundary jumps are
                # avoided in-block instead of corrected post-hoc.
                if sto.ramp_cost > 0.0 and sto.name in prev_net:
                    sto.ramp_t0_reference = prev_net[sto.name]
            for link in es._links:
                if link.ramp_cost > 0.0 and link.name in prev_flow:
                    link.ramp_t0_reference = prev_flow[link.name]
        pinned = False
        priced: dict[str, float] = {}
        if k < n_blocks - 1:
            # Caller λ is indexed by LB boundaries; map this UB boundary to
            # the nearest one (identity when lb_blocks == n_blocks).
            j = int(np.argmin(np.abs(
                edges_lb[1:-1] - edges[k + 1]))) if n_bnd else 0
            for sto in es._storages:
                if ub_boundary in ("prices", "both") and n_bnd:
                    lt = lam.get(sto.name)
                    if lt is not None and lt[j] > 0.0:
                        sto.soc_terminal_cost = -float(lt[j])
                        priced[sto.name] = float(lt[j])
                if ub_boundary in ("floors", "both") and boundary_soc_min is not None:
                    tgt = boundary_soc_min.get(sto.name)
                    if tgt is not None:
                        lo = sto.soc_min * sto.energy_capacity
                        hi = sto.soc_max * sto.energy_capacity
                        sto.soc_terminal_min = float(
                            min(max(float(tgt[k]), lo), hi))
                        pinned = True
        res = es.optimise(**solve_kwargs)
        if res.status != "optimal" and pinned:
            if verbose:
                print(f"[temporal_certified] UB block {k + 1} infeasible "
                      "with SOC floors — retrying unfloored")
            for sto in es._storages:
                sto.soc_terminal_min = None
            res = es.optimise(**solve_kwargs)
        if res.status != "optimal":
            return TemporalCertifiedResult(
                status="block_failed", objective=float("nan"),
                lower_bound=float("nan"), gap=float("inf"), gap_target=gap,
                n_blocks=n_blocks, ub_wall=time.perf_counter() - ub_t0,
                lb_wall=0.0, total_wall=time.perf_counter() - t_start)
        # True block cost: undo the soft terminal-energy payment.
        block_cost = res.total_cost
        for nm, price in priced.items():
            block_cost += price * float(res.storage_soc[nm][-1])
        obj_sum += block_cost
        block_objs.append(block_cost)

        # Boundary corrections. Ramp costs need none: the t=0 ramp reference
        # handoff makes each block charge the TRUE boundary ramp in-block.
        # Startup/shutdown remain conservatively corrected.
        for link in es._links:
            if link.ramp_cost > 0.0 and link.name in res.link_flow:
                prev_flow[link.name] = float(res.link_flow[link.name][-1])
            if link.committable and link._status_vars:
                u0, u_end = _link_status_edge(es, res, link)
                if k > 0:
                    pu = prev_u[link.name]
                    if link.startup_cost > 0.0 and u0 > 0.5 and pu < 0.5:
                        correction += link.startup_cost
                    if link.shutdown_cost > 0.0 and u0 < 0.5 and pu > 0.5:
                        correction += link.shutdown_cost
                prev_u[link.name] = u_end
        for sto in es._storages:
            if sto.ramp_cost > 0.0 and sto.name in res.storage_charge:
                net = (res.storage_discharge[sto.name]
                       - res.storage_charge[sto.name])
                prev_net[sto.name] = float(net[-1])
            prev_soc[sto.name] = float(res.storage_soc[sto.name][-1])

        for field_name, store in stitched.items():
            for name, arr in getattr(res, field_name).items():
                store.setdefault(name, []).append(np.asarray(arr))
        if verbose:
            print(f"[temporal_certified] UB block {k + 1}/{n_blocks} "
                  f"[{t0}:{t1}) cost={block_cost:.4f}")

    ub = obj_sum + correction
    ub_wall = time.perf_counter() - ub_t0

    # ---------------- boundary-flow prices from the UB stitch -------------
    # The LB blocks' t=0 ramp rows are dropped (relaxation), letting blocks
    # teleport their flows at boundaries for free. Recover part of that
    # cost with linear prices: rc·|Δf| ≥ rc·s·Δf for ANY |s| ≤ 1, so adding
    # +rc·s·f_first to the receiver and -rc·s·f_last to the donor is valid
    # for arbitrary s — and tight when s matches the optimum's boundary
    # ramp direction, which the stitched UB flows estimate well.
    mu_link: dict[str, np.ndarray] = {}
    mu_sto: dict[str, np.ndarray] = {}
    # Proximal references F̄/Z̄ at each LB boundary (the stitched state at
    # the donor's final step): receivers keep their t=0 ramp rows priced
    # against F̄ and pay κ·|s_in - Z̄|; donors get the mirror V-rebates.
    # Valid for ANY refs/rates (the pairs cancel exactly on a continuous
    # trajectory; the flow pair under-estimates rc·|Δf| by the triangle
    # inequality) — good refs kill the boundary-teleport slack that linear
    # prices cannot touch (they saturate at the |Δf| kink).
    prox_fbar: dict[str, np.ndarray] = {}
    prox_nbar: dict[str, np.ndarray] = {}
    prox_zbar: dict[str, np.ndarray] = {}
    if lb_proximal and n_bnd:
        for nm, parts in stitched["link_flow"].items():
            link = next((l for l in probe._links if l.name == nm), None)
            if link is None or link.ramp_cost <= 0.0:
                continue
            f_full = np.concatenate(parts)
            prox_fbar[nm] = np.array(
                [float(f_full[int(edges_lb[i + 1]) - 1])
                 for i in range(n_bnd)])
        for nm, parts in stitched["storage_charge"].items():
            sto = next((s for s in probe._storages if s.name == nm), None)
            if sto is None or sto.ramp_cost <= 0.0:
                continue
            net_full = (np.concatenate(stitched["storage_discharge"][nm])
                        - np.concatenate(parts))
            prox_nbar[nm] = np.array(
                [float(net_full[int(edges_lb[i + 1]) - 1])
                 for i in range(n_bnd)])
        for nm, parts in stitched["storage_soc"].items():
            s_full = np.concatenate(parts)
            prox_zbar[nm] = np.array(
                [float(s_full[int(edges_lb[i + 1]) - 1])
                 for i in range(n_bnd)])
    elif lb_flow_prices and n_bnd:
        for nm, parts in stitched["link_flow"].items():
            link = next((l for l in probe._links if l.name == nm), None)
            if link is None or link.ramp_cost <= 0.0:
                continue
            f_full = np.concatenate(parts)
            arr = np.zeros(n_bnd)
            for i in range(n_bnd):
                tb = int(edges_lb[i + 1])
                d = float(f_full[tb] - f_full[tb - 1])
                if abs(d) > 1e-6:
                    arr[i] = link.ramp_cost * (1.0 if d > 0 else -1.0)
            mu_link[nm] = arr
        for nm, parts in stitched["storage_charge"].items():
            sto = next((s for s in probe._storages if s.name == nm), None)
            if sto is None or sto.ramp_cost <= 0.0:
                continue
            net_full = (np.concatenate(stitched["storage_discharge"][nm])
                        - np.concatenate(parts))
            arr = np.zeros(n_bnd)
            for i in range(n_bnd):
                tb = int(edges_lb[i + 1])
                d = float(net_full[tb] - net_full[tb - 1])
                if abs(d) > 1e-6:
                    arr[i] = sto.ramp_cost * (1.0 if d > 0 else -1.0)
            mu_sto[nm] = arr
    _rc_link = {l.name: float(l.ramp_cost) for l in probe._links
                if l.ramp_cost > 0.0}
    _rc_sto = {s.name: float(s.ramp_cost) for s in probe._storages
               if s.ramp_cost > 0.0}

    # ---------------- LB rounds: relaxed independent blocks ----------------
    # lb_rounds=0 skips the LB pass entirely (UB-only diagnostic mode:
    # status is always "gap_not_met", lower_bound -inf).
    lb_t0 = time.perf_counter()
    best_lb = -float("inf")
    best_lam = {k_: v.copy() for k_, v in lam.items()}
    best_block_lbs: list[float] = []
    round_bounds: list[float] = []
    pool = None
    if lb_rounds > 0 and lb_workers > 1:
        import concurrent.futures as _cf
        pool = _cf.ProcessPoolExecutor(max_workers=lb_workers)
    try:
        for r in range(lb_rounds):
            args = []
            for k in range(K_lb):
                lam_start = {nm: lam[nm][k - 1] for nm in lam} if k >= 1 else {}
                lam_term = {nm: lam[nm][k] for nm in lam} if k <= K_lb - 2 else {}
                start_hi = ({nm: env_hi[nm][k - 1] for nm in env_hi}
                            if k >= 1 else {})
                term_hi = ({nm: env_hi[nm][k] for nm in env_hi}
                           if k <= K_lb - 2 else {})
                if lb_proximal:
                    fp = {
                        "_skip_t0": False,  # rows kept, priced vs F̄
                        "link_ramp_ref": (
                            {nm: prox_fbar[nm][k - 1] for nm in prox_fbar}
                            if k >= 1 else {}),
                        "sto_ramp_ref": (
                            {nm: prox_nbar[nm][k - 1] for nm in prox_nbar}
                            if k >= 1 else {}),
                        "link_term_vreb": (
                            {nm: (prox_fbar[nm][k], _rc_link[nm])
                             for nm in prox_fbar}
                            if k <= K_lb - 2 else {}),
                        "sto_net_vreb": (
                            {nm: (prox_nbar[nm][k], _rc_sto[nm])
                             for nm in prox_nbar}
                            if k <= K_lb - 2 else {}),
                        "soc_start_v": (
                            {nm: (prox_zbar[nm][k - 1], abs(lam[nm][k - 1]))
                             for nm in prox_zbar if nm in lam}
                            if k >= 1 else {}),
                        "soc_term_vreb": (
                            {nm: (prox_zbar[nm][k], abs(lam[nm][k]))
                             for nm in prox_zbar if nm in lam}
                            if k <= K_lb - 2 else {}),
                    }
                else:
                    fp = {
                        "link_t0": ({nm: mu_link[nm][k - 1] for nm in mu_link}
                                    if k >= 1 else {}),
                        "link_term": ({nm: -mu_link[nm][k] for nm in mu_link}
                                      if k <= K_lb - 2 else {}),
                        "sto_t0": ({nm: mu_sto[nm][k - 1] for nm in mu_sto}
                                   if k >= 1 else {}),
                        "sto_term": ({nm: -mu_sto[nm][k] for nm in mu_sto}
                                     if k <= K_lb - 2 else {}),
                    }
                args.append((system_factory, int(edges_lb[k]),
                             int(edges_lb[k + 1]),
                             k, K_lb, lam_start, lam_term,
                             start_hi, term_hi, fp, solve_kwargs))
            if pool is not None:
                outs = list(pool.map(_lb_block_task, *zip(*args)))
            else:
                outs = [_lb_block_task(*a) for a in args]
            outs.sort(key=lambda o: o[0])
            if any(o[1] is None for o in outs):
                return TemporalCertifiedResult(
                    status="block_failed", objective=ub,
                    lower_bound=float("nan"), gap=float("inf"), gap_target=gap,
                    n_blocks=n_blocks, ub_wall=ub_wall,
                    lb_wall=time.perf_counter() - lb_t0,
                    total_wall=time.perf_counter() - t_start,
                    block_objectives=block_objs)
            block_lbs = [o[1] for o in outs]
            lb_r = float(sum(block_lbs))
            round_bounds.append(lb_r)
            improved = lb_r > best_lb
            if improved:
                best_lb = lb_r
                best_lam = {k_: v.copy() for k_, v in lam.items()}
                best_block_lbs = block_lbs
            if verbose:
                print(f"[temporal_certified] LB round {r + 1}/{lb_rounds}: "
                      f"Σ bound = {lb_r:.4f} (best {best_lb:.4f})")
            if r == max(1, lb_rounds) - 1:
                break
            # Multiplicative price update on the boundary mismatch — λ enters
            # block objectives as ±λ·SOC with SOC up to the (possibly huge)
            # energy capacity, so steps must scale WITH λ, not absolutely.
            # Receiver wants more than the donor delivers → underpriced →
            # grow λ; oversupplied → shrink. A round that worsens the bound
            # restarts from the best λ with a halved step (overshoot guard).
            if improved or r == 0:
                src = lam
            else:
                lb_step_scale *= 0.5
                src = {k_: v.copy() for k_, v in best_lam.items()}
            new_lam = {k_: v.copy() for k_, v in src.items()}
            for i in range(n_bnd):
                donor_i = outs[i][2]        # block i terminal choices
                recv_i = outs[i + 1][3]     # block i+1 start choices
                for nm in new_lam:
                    d = donor_i.get(nm)
                    rv = recv_i.get(nm)
                    if d is None or rv is None:
                        continue
                    g_norm = min(max((rv - d) / e_caps[nm], -1.0), 1.0)
                    base = new_lam[nm][i]
                    if base <= 0.0 and g_norm > 0.0:
                        base = 1e-3 * lam_seed  # let zero prices grow
                    new_lam[nm][i] = max(0.0, base * (1.0 + lb_step_scale * g_norm))
            lam = new_lam
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    lb_wall = time.perf_counter() - lb_t0
    lam = best_lam

    cert_gap = (ub - best_lb) / max(1.0, abs(ub))
    status = "certified" if cert_gap <= gap + 1e-12 else "gap_not_met"
    result = TemporalCertifiedResult(
        status=status, objective=ub, lower_bound=best_lb, gap=cert_gap,
        gap_target=gap, n_blocks=n_blocks, ub_wall=ub_wall, lb_wall=lb_wall,
        total_wall=time.perf_counter() - t_start,
        block_objectives=block_objs, block_lower_bounds=best_block_lbs,
        lb_round_bounds=round_bounds, lambda_final=lam,
        boundary_correction=correction)
    for field_name, store in stitched.items():
        setattr(result, field_name,
                {name: np.concatenate(parts) for name, parts in store.items()})
    if verbose:
        print(f"[temporal_certified] UB={ub:.4f} LB={best_lb:.4f} "
              f"gap={cert_gap:.4%} target={gap:.4%} → {status}")
    return result
