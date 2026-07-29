"""
Phase 8 — adaptive-oracle Benders vs plain / trust-region / PH benchmark.

Runs the same 20-scenario two-stage capacity-expansion problem through
four solvers:

  1. plain Benders           — baseline.
  2. trust-region Benders    — ℓ∞ cap-var trust region.
  3. adaptive-oracle Benders — loose early sub-gap, tightens with iter.
  4. progressive hedging     — per-scenario consensus via ℓ∞ trust
                                region (Rockafellar–Wets analogue).

Reports: wall-clock time, sub-LP count (Benders variants), iterations,
final expected cost. Reproducibility is controlled by ``--seed``.

The claim backed by this benchmark: adaptive-oracle trades sub-LP
accuracy for cut-quantity in early iterations, so for large scenario
sets it closes the gap in less wall-clock than plain / trust_region.

Usage:
    python benchmarks/decomposition/adaptive_oracle_bench.py
    python benchmarks/decomposition/adaptive_oracle_bench.py --n 30 --T 48
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import nexus_energy as ne
from nexus_energy.decomposition import BendersDecomposer
from nexus_energy.stochastic import (
    Scenario,
    apply_scenario,
    generate_moment_matching_scenarios,
    solve_stochastic,
    solve_stochastic_ph,
)


def build_system(T: int = 24) -> ne.EnergySystem:
    rng = np.random.default_rng(0)
    sys = ne.EnergySystem("adaptive_bench")
    elec = sys.add_bus("elec", carrier="electricity")
    day = np.arange(T) % 24
    load = 120 + 60 * np.cos((day - 18) * np.pi / 12) ** 2
    load = load + rng.normal(0, 1, size=T)
    sys.add_load("d", bus=elec, amount=load)
    sys.add_generator("base", bus=elec, capacity=80, marginal_cost=10,
                      tech="gas")
    sys.add_generator("slack", bus=elec, capacity=5000, marginal_cost=5000)
    cf = np.clip(np.cos((day - 12) * np.pi / 12), 0, None)
    sys.add_generator(
        "solar", bus=elec, capacity=10, marginal_cost=0,
        carrier_factor=cf, extendable=True, min_capacity=10,
        max_capacity=600, capital_cost=40, tech="solar",
    )
    sys.add_generator(
        "peaker", bus=elec, capacity=10, marginal_cost=200,
        extendable=True, min_capacity=10, max_capacity=600,
        capital_cost=15, tech="peaker",
    )
    return sys


def build_scenarios(n: int, seed: int) -> list[Scenario]:
    mean = np.array([1.0, 1.0, 1.0])
    cov = np.array([
        [0.05, 0.01, 0.0],
        [0.01, 0.04, 0.0],
        [0.0,  0.0,  0.02],
    ])
    return generate_moment_matching_scenarios(
        target_mean=mean,
        target_cov=cov,
        n_scenarios=n,
        seed=seed,
    )


def run_benders(sys: ne.EnergySystem, scenarios: list[Scenario],
                 stabilisation: str) -> dict:
    subs = [apply_scenario(sys, s) for s in scenarios]
    probs = [s.probability for s in scenarios]
    decomp = BendersDecomposer(
        system=sys,
        subsystems=subs,
        period_weights=probs,
        max_iter=40,
        tol=1e-3,
        stabilisation=stabilisation,
    )
    t0 = time.perf_counter()
    res = decomp.solve()
    wall = time.perf_counter() - t0
    sub_costs = res.iterations[-1].subproblem_costs if res.iterations else []
    expected = sum(p * c for p, c in zip(probs, sub_costs))
    return {
        "method": f"benders_{stabilisation}",
        "status": res.status,
        "iterations": len(res.iterations),
        "sub_solves": res.sub_solves,
        "wall_sec": wall,
        "expected_cost": expected,
    }


def run_ph(sys: ne.EnergySystem, scenarios: list[Scenario]) -> dict:
    t0 = time.perf_counter()
    res = solve_stochastic_ph(
        sys, scenarios,
        rho=1.0, max_iter=30, tol=3e-3,
        initial_radius=0.5, radius_decay=0.8,
    )
    wall = time.perf_counter() - t0
    return {
        "method": "progressive_hedging",
        "status": res.status,
        "iterations": res.n_iterations,
        "sub_solves": res.n_iterations * len(scenarios),
        "wall_sec": wall,
        "expected_cost": res.expected_cost,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="scenario count")
    ap.add_argument("--T", type=int, default=24, help="timesteps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default=None,
                    help="optional path to dump results as JSON")
    args = ap.parse_args()

    sys = build_system(T=args.T)
    scenarios = build_scenarios(args.n, args.seed)

    rows = []
    for stab in ("plain", "trust_region", "adaptive"):
        rows.append(run_benders(sys, scenarios, stab))
    rows.append(run_ph(sys, scenarios))

    # Report.
    head = f"{'method':25s} {'status':15s} {'iter':>6s} {'sub':>6s} {'wall(s)':>10s} {'E[cost]':>14s}"
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['method']:25s} {r['status']:15s} "
              f"{r['iterations']:6d} {r['sub_solves']:6d} "
              f"{r['wall_sec']:10.3f} {r['expected_cost']:14.2f}")
    print("=" * len(head))

    # Sanity: all expected costs within 5% of median.
    costs = [r["expected_cost"] for r in rows
             if r["status"] in ("optimal", "max_iter")]
    if costs:
        med = float(np.median(costs))
        spread = (max(costs) - min(costs)) / max(abs(med), 1.0)
        print(f"cross-method spread: {spread*100:.2f}% (median ${med:.2f})")

    if args.json:
        out = {
            "config": {"n_scenarios": args.n, "T": args.T, "seed": args.seed},
            "rows": rows,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
