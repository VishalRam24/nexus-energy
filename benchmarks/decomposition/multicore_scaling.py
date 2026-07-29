"""
Milestone 1 — multicore Benders scaling proof.

The Monday pain this targets: "complex optimisations only use one core — an i5
and a Xeon finish in about the same wall-clock." Production solvers parallelise
their internal simplex/IPM poorly, so adding cores rarely helps the user.

The fix is decomposition *above* the solver: a two-stage stochastic
capacity-expansion problem splits into one independent operational subproblem
per scenario, given the master's capacities. Those subproblems are
embarrassingly parallel, so `BendersDecomposer(n_jobs=N)` solves them across N
worker processes (each with its own single-threaded HiGHS) and the subproblem
pass scales with cores — exactly the lever the monolithic solver can't pull.

What this script proves:
  1. SPEED      — wall-clock vs cores (subproblem-phase and total), with
                  speed-up and parallel efficiency.
  2. CORRECTNESS— every core count returns the *same* objective. For
                  decomposition, parity IS correctness, so the scaling proof
                  and the correctness proof are one artifact.

Each subproblem is a full multi-bus operational dispatch over `T` snapshots, so
solve-time per subproblem dwarfs the process/pickle overhead — the regime where
parallelism honestly bites (tiny LPs would be swamped by pool overhead).

Usage:
    python benchmarks/decomposition/multicore_scaling.py
    python benchmarks/decomposition/multicore_scaling.py --T 2920 --n 24 \
        --jobs 1,2,4,8 --json out/scaling.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import nexus_energy as ne
from nexus_energy.decomposition import BendersDecomposer
from nexus_energy.stochastic import Scenario, apply_scenario


def build_heavy_system(T: int, n_buses: int) -> ne.EnergySystem:
    """Multi-bus capacity-expansion system sized so each scenario subproblem is
    a non-trivial LP (seconds, not microseconds).

    Per bus: a time-varying load, a capacity-limited cheap baseload generator,
    an *extendable* solar generator (first-stage), an *extendable* peaker
    (first-stage), an *extendable* battery (first-stage), and an expensive slack
    for guaranteed feasibility. Buses are wired in a ring by bidirectional
    links, so dispatch couples spatially across the whole horizon.
    """
    rng = np.random.default_rng(0)
    sys = ne.EnergySystem("multicore_bench")
    sys.set_timesteps(T)
    hour = np.arange(T) % 24
    buses = []
    for b in range(n_buses):
        bus = sys.add_bus(f"elec{b}", carrier="electricity")
        buses.append(bus)
        # Load: diurnal with a per-bus phase offset + mild noise.
        phase = b * 2.0
        load = 220 + 90 * np.cos((hour - 18 - phase) * np.pi / 12) ** 2
        load = load + rng.normal(0, 2, size=T)
        sys.add_load(f"d{b}", bus=bus, amount=np.clip(load, 1.0, None))
        sys.add_generator(f"base{b}", bus=bus, capacity=120,
                          marginal_cost=12 + b, tech="gas")
        cf = np.clip(np.cos((hour - 12) * np.pi / 12), 0, None)
        cf = cf * (0.85 + 0.05 * b)
        sys.add_generator(
            f"solar{b}", bus=bus, capacity=10, marginal_cost=0,
            carrier_factor=np.clip(cf, 0, 1), extendable=True,
            min_capacity=10, max_capacity=900, capital_cost=42, tech="solar",
        )
        sys.add_generator(
            f"peak{b}", bus=bus, capacity=10, marginal_cost=180 + 5 * b,
            extendable=True, min_capacity=10, max_capacity=900,
            capital_cost=16, tech="peaker",
        )
        # Positive min capacity: an extendable storage pinned to exactly 0 by
        # the master makes the SOC/cyclic constraints degenerate-infeasible, so
        # keep it a genuine [min, max] first-stage decision that never hits 0.
        sys.add_storage(
            f"bat{b}", bus=bus, power_capacity=20, energy_capacity=80,
            efficiency_charge=0.96, efficiency_discharge=0.96,
            extendable=True, min_power_capacity=20, min_energy_capacity=80,
            max_power_capacity=400, max_energy_capacity=1600,
            capital_cost_power=22, capital_cost_energy=4,
        )
        sys.add_generator(f"slack{b}", bus=bus, capacity=10000,
                          marginal_cost=6000)
    # Ring transmission.
    for b in range(n_buses):
        nb = (b + 1) % n_buses
        sys.add_link(f"line_{b}_{nb}", bus_from=buses[b], bus_to=buses[nb],
                     capacity=150, efficiency=0.98)
        sys.add_link(f"line_{nb}_{b}", bus_from=buses[nb], bus_to=buses[b],
                     capacity=150, efficiency=0.98)
    return sys


def build_scenarios(n: int, seed: int) -> list[Scenario]:
    """Correlated demand / renewables / fuel-price scenarios, equal weight."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        demand = float(np.clip(rng.normal(1.0, 0.12), 0.6, 1.4))
        res = float(np.clip(rng.normal(1.0, 0.20), 0.4, 1.5))
        fuel = float(np.clip(rng.normal(1.0, 0.15), 0.6, 1.6))
        out.append(Scenario(
            name=f"sc{i}", probability=1.0 / n,
            demand_factor=demand, carrier_factor_scale=res,
            fuel_cost_factor=fuel,
        ))
    return out


def run_one(base_sys, scenarios, n_jobs: int, max_iter: int) -> dict:
    """One full Benders solve at a given worker count. Fresh subsystems each
    call so no run inherits another's solver-state cache."""
    probs = [s.probability for s in scenarios]
    subs = [apply_scenario(base_sys, s) for s in scenarios]
    decomp = BendersDecomposer(
        system=base_sys, subsystems=subs, period_weights=probs,
        max_iter=max_iter, tol=1e-3, stabilisation="plain", n_jobs=n_jobs,
    )
    t0 = time.perf_counter()
    res = decomp.solve()
    wall = time.perf_counter() - t0
    return {
        "n_jobs": n_jobs,
        "status": res.status,
        "iterations": len(res.iterations),
        "sub_solves": res.sub_solves,
        "objective": float(res.total_cost),
        "total_wall_sec": wall,
        "subproblem_sec": float(res.subproblem_seconds),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2920,
                    help="timesteps per scenario subproblem (3-hourly year)")
    ap.add_argument("--buses", type=int, default=4)
    ap.add_argument("--n", type=int, default=24, help="scenario count")
    ap.add_argument("--max-iter", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=str, default=None,
                    help="comma list of worker counts, e.g. 1,2,4,8")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    n_cpu = os.cpu_count() or 1
    if args.jobs:
        job_counts = [int(j) for j in args.jobs.split(",")]
    else:
        job_counts = [j for j in (1, 2, 4, 8, 16) if j <= n_cpu]
        if n_cpu not in job_counts:
            job_counts.append(n_cpu)
    job_counts = sorted(set(job_counts))

    print(f"host cores: {n_cpu}   scenarios: {args.n}   T: {args.T}   "
          f"buses: {args.buses}   worker counts: {job_counts}")
    print("building system + scenarios ...", flush=True)
    base_sys = build_heavy_system(args.T, args.buses)
    scenarios = build_scenarios(args.n, args.seed)

    rows = [run_one(base_sys, scenarios, j, args.max_iter) for j in job_counts]

    base = next(r for r in rows if r["n_jobs"] == job_counts[0])
    base_total = base["total_wall_sec"]
    base_sub = base["subproblem_sec"]

    head = (f"{'jobs':>5s} {'status':>11s} {'iter':>5s} {'sub':>5s} "
            f"{'total(s)':>10s} {'subprob(s)':>11s} {'speedup':>8s} "
            f"{'effic':>7s} {'objective':>16s}")
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for r in rows:
        speedup = base_total / r["total_wall_sec"] if r["total_wall_sec"] else 0.0
        effic = speedup / r["n_jobs"] if r["n_jobs"] else 0.0
        print(f"{r['n_jobs']:5d} {r['status']:>11s} {r['iterations']:5d} "
              f"{r['sub_solves']:5d} {r['total_wall_sec']:10.2f} "
              f"{r['subproblem_sec']:11.2f} {speedup:7.2f}x {effic:6.0%} "
              f"{r['objective']:16.2f}")
    print("=" * len(head))

    # Sub-phase speed-up (the parallel kernel, Amdahl-free).
    print("\nsubproblem-phase speed-up (the parallelisable kernel):")
    for r in rows:
        sp = base_sub / r["subproblem_sec"] if r["subproblem_sec"] else 0.0
        print(f"  {r['n_jobs']:2d} jobs: {sp:5.2f}x  "
              f"({base_sub:.1f}s -> {r['subproblem_sec']:.1f}s)")

    # CORRECTNESS = PARITY: every core count must agree on the objective. Both
    # "optimal" (converged) and "iteration_limit" (capped but deterministic)
    # rows carry a meaningful objective, so check across both.
    objs = [r["objective"] for r in rows
            if r["status"] in ("optimal", "iteration_limit")
            and np.isfinite(r["objective"])]
    ok = True
    if objs:
        spread = (max(objs) - min(objs)) / max(abs(np.median(objs)), 1.0)
        ok = bool(spread < 1e-6)
        print(f"\nobjective spread across core counts: {spread:.2e}  "
              f"-> {'PARITY HOLDS' if ok else 'MISMATCH!'}")

    if args.json:
        out = {
            "config": {"T": args.T, "buses": args.buses, "n_scenarios": args.n,
                       "max_iter": args.max_iter, "seed": args.seed,
                       "host_cores": n_cpu},
            "rows": rows,
            "parity_ok": ok,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")

    if not ok:
        raise SystemExit("PARITY VIOLATION: core counts disagree on objective")


if __name__ == "__main__":
    main()
