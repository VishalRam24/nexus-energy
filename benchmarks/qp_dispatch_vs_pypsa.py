"""
N_En_Phase 17.1 — QP head-to-head vs PyPSA.

Both sides solve an identical economic-dispatch QP:
    min  sum_t sum_g (mc_g * p_g[t] + mq_g * p_g[t]^2)
    s.t. sum_g p_g[t] = load[t]   for all t
         0 <= p_g[t] <= p_nom_g

PyPSA route: Network with `marginal_cost` + `marginal_cost_quadratic`,
solved via `n.optimize(solver_name="highs")` (linopy → HiGHS QP).

Nexus route: low-level `nexus.Model` (the route nexus-energy will use
once `marginal_cost_quadratic` lands on Generator). We measure each
of {highs, osqp, clarabel, ipopt} via the dispatchers added in
N_En_Phase 10.5 + 10.9.

Reports objective + solve wall to JSON; asserts cross-framework parity
at 1e-4 relative.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
for _n in ("pypsa", "linopy", "highspy"):
    logging.getLogger(_n).setLevel(logging.ERROR)


def make_case(T: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    gens = [
        {"name": "coal", "p_nom": 200.0, "mc": 30.0, "mq": 0.02},
        {"name": "ccgt", "p_nom": 150.0, "mc": 50.0, "mq": 0.05},
        {"name": "peak", "p_nom": 100.0, "mc": 80.0, "mq": 0.10},
    ]
    hours = np.arange(T)
    diurnal = 250 + 80 * np.sin((hours - 6) * np.pi / 12)
    noise = rng.normal(0, 5, size=T)
    load = np.clip(diurnal + noise, 50, 400)
    return gens, load


def solve_nexus(gens, load, solver: str):
    import nexus as nx
    T = len(load)
    t0 = time.perf_counter()
    m = nx.Model(f"qp_dispatch_{solver}")
    p = {g["name"]: m.variables(f"p_{g['name']}", T, lower=0, upper=g["p_nom"]) for g in gens}

    for t in range(T):
        m.add(nx.sum([p[g["name"]][t] for g in gens]) == float(load[t]),
              name=f"balance_{t}")

    obj_q = sum(
        float(g["mq"]) * (p[g["name"]][t] * p[g["name"]][t])
        for g in gens for t in range(T)
    )
    obj_l = nx.sum([
        float(g["mc"]) * p[g["name"]][t]
        for g in gens for t in range(T)
    ])
    m.minimize(obj_q + obj_l)

    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()
    r = m.solve(solver=solver)
    t_solve = time.perf_counter() - t1

    return {
        "framework": f"nexus_{solver}",
        "status": r.status,
        "objective": r.objective,
        "build_ms": t_build * 1000,
        "solve_ms": t_solve * 1000,
    }


def solve_pypsa(gens, load):
    # Guarded competitor hook: skip cleanly if PyPSA is not installed.
    try:
        import pandas as pd
        import pypsa
    except ImportError:
        return {"framework": "pypsa_highs_qp", "status": "not_installed"}
    T = len(load)
    snapshots = pd.date_range("2024-01-01", periods=T, freq="h")
    t0 = time.perf_counter()
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.add("Bus", "bus")
    for g in gens:
        n.add(
            "Generator",
            g["name"],
            bus="bus",
            p_nom=g["p_nom"],
            marginal_cost=g["mc"],
            marginal_cost_quadratic=g["mq"],
        )
    n.add("Load", "demand", bus="bus", p_set=pd.Series(load, index=snapshots))
    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()
    n.optimize(solver_name="highs")
    t_solve = time.perf_counter() - t1

    obj = float(n.model.objective.value)
    return {
        "framework": "pypsa_highs_qp",
        "status": "optimal",
        "objective": obj,
        "build_ms": t_build * 1000,
        "solve_ms": t_solve * 1000,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path("benchmarks/results/qp_dispatch_vs_pypsa.json"))
    args = p.parse_args()

    gens, load = make_case(T=args.T, seed=args.seed)

    # --- Nexus side (always runs): try every QP backend, keep the ones present.
    rows = []
    for solver in ["highs", "osqp", "clarabel", "ipopt"]:
        try:
            rows.append(solve_nexus(gens, load, solver))
        except ModuleNotFoundError as e:
            print(f"  nexus_{solver}: SKIPPED (backend not installed: {e})")
    if not rows:
        raise SystemExit("no nexus QP backend available — install highs/clarabel")

    # --- Competitor side (guarded): PyPSA.
    pypsa_row = solve_pypsa(gens, load)
    if pypsa_row.get("status") == "not_installed":
        print("  pypsa: SKIPPED (not installed) — nexus side only")
    else:
        rows.append(pypsa_row)

    objs = [r["objective"] for r in rows]
    omax, omin = max(objs), min(objs)
    rel_spread = (omax - omin) / max(abs(omax), abs(omin), 1.0)

    print(f"{'framework':22} {'objective':>16} {'build_ms':>10} {'solve_ms':>10}")
    for r in rows:
        print(f"{r['framework']:22} {r['objective']:16.4f} {r['build_ms']:10.2f} {r['solve_ms']:10.2f}")
    print(f"\nrel_spread vs max: {rel_spread:.2e}  (parity tol = 1e-4)")
    # Parity is meaningful whether the spread is across frameworks or just
    # across nexus backends; either way all present rows must agree.
    assert rel_spread < 1e-4, f"QP H2H parity failed: {rows}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"T": args.T, "seed": args.seed, "rows": rows,
                                    "rel_spread": rel_spread}, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
