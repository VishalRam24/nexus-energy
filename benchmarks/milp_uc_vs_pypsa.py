"""
N_En_Phase 14.6 — true-MILP unit-commitment head-to-head vs PyPSA.

Both sides solve the *same* small UC MILP (binary on/off status, min-up /
min-down time, startup cost) over a short horizon, with HiGHS as the MIP
backend, and we compare total cost + the committed schedule.

Nexus side (the deliverable — always runs):
    EnergySystem with committable generators
    (`committable=True, min_up_time, min_down_time, startup_cost`) solved via
    `sys.optimise()`. This is a genuine MILP — binary status/startup/shutdown
    variables — not an LP relaxation.

Competitor side (guarded): PyPSA with `committable=True`,
`min_up_time`, `min_down_time`, `start_up_cost`, solved via
`n.optimize(solver_name="highs")`. Skipped with a message if PyPSA is absent.

Reports total cost (and per-snapshot commitment when both run); asserts
cross-framework cost parity at 1e-4 relative when PyPSA is present.

Usage:
    python benchmarks/milp_uc_vs_pypsa.py            # auto-skips PyPSA if absent
    python benchmarks/milp_uc_vs_pypsa.py --nexus-only
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

RESULTS = Path(__file__).parent / "results" / "milp_uc_vs_pypsa.json"


def make_case():
    """Tiny 1-bus UC: one cheap committable unit + an expensive always-on
    slack, with a demand profile that forces a start-up partway through.

    Demand dips below the committable unit's economic point then rises, so
    the MIP must choose whether to keep it committed (min-up cost) or cycle.
    """
    demand = np.array([60.0, 20.0, 20.0, 90.0, 90.0, 30.0], dtype=float)
    units = [
        # name, p_nom, mc, p_min_pu, min_up, min_down, startup_cost
        dict(name="coal", p_nom=80.0, mc=25.0, p_min_pu=0.4,
             min_up=2, min_down=2, startup=200.0),
    ]
    slack = dict(name="slack", p_nom=200.0, mc=300.0)
    return demand, units, slack


# ---------------------------------------------------------------------------
# Nexus side
# ---------------------------------------------------------------------------

def solve_nexus(demand, units, slack):
    import nexus_energy as ne

    t0 = time.perf_counter()
    sys = ne.EnergySystem("uc_milp_h2h")
    T = len(demand)
    sys.set_timesteps(T)
    bus = sys.add_bus("e")
    sys.add_load("ld", bus=bus, amount=demand)
    for u in units:
        sys.add_generator(
            u["name"], bus=bus, capacity=u["p_nom"], marginal_cost=u["mc"],
            committable=True, min_up_time=u["min_up"],
            min_down_time=u["min_down"], startup_cost=u["startup"],
            min_capacity=u["p_min_pu"] * u["p_nom"],
        )
    sys.add_generator(slack["name"], bus=bus, capacity=slack["p_nom"],
                      marginal_cost=slack["mc"])
    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()
    res = sys.optimise(solver="highs")
    t_solve = time.perf_counter() - t1

    # Extract the binary commitment schedule for the committable unit(s).
    schedule = {}
    for gen in sys._generators:
        status_vars = getattr(gen, "_status_vars", None)
        if status_vars:
            schedule[gen.name] = [
                round(float(res._raw.value(v))) for v in status_vars
            ]
    return {
        "framework": "nexus_highs_milp",
        "status": res.status,
        "objective": float(res.total_cost),
        "build_ms": t_build * 1000,
        "solve_ms": t_solve * 1000,
        "commitment": schedule,
    }


# ---------------------------------------------------------------------------
# Competitor side (guarded)
# ---------------------------------------------------------------------------

def solve_pypsa(demand, units, slack):
    try:
        import pandas as pd
        import pypsa
    except ImportError:
        return {"framework": "pypsa_highs_milp", "status": "not_installed"}

    T = len(demand)
    snapshots = pd.date_range("2024-01-01", periods=T, freq="h")
    t0 = time.perf_counter()
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.add("Bus", "e")
    for u in units:
        n.add("Generator", u["name"], bus="e", p_nom=u["p_nom"],
              marginal_cost=u["mc"], committable=True,
              p_min_pu=u["p_min_pu"], min_up_time=u["min_up"],
              min_down_time=u["min_down"], start_up_cost=u["startup"])
    n.add("Generator", slack["name"], bus="e", p_nom=slack["p_nom"],
          marginal_cost=slack["mc"])
    n.add("Load", "ld", bus="e", p_set=pd.Series(demand, index=snapshots))
    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()
    try:
        n.optimize(solver_name="highs")
        status = "optimal"
    except Exception as e:  # noqa: BLE001
        status = f"error: {e}"
    t_solve = time.perf_counter() - t1

    obj = float("nan")
    try:
        obj = float(n.objective)
    except Exception:
        pass
    commitment = {}
    try:
        for u in units:
            commitment[u["name"]] = [
                round(float(x)) for x in n.generators_t.status[u["name"]].values
            ]
    except Exception:
        pass
    return {
        "framework": "pypsa_highs_milp",
        "status": status,
        "objective": obj,
        "build_ms": t_build * 1000,
        "solve_ms": t_solve * 1000,
        "commitment": commitment,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nexus-only", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)

    demand, units, slack = make_case()

    # --- Nexus side always runs. ---
    nx = solve_nexus(demand, units, slack)
    assert nx["status"] == "optimal", f"nexus UC MILP not optimal: {nx['status']}"
    print(f"nexus  MILP: status={nx['status']}  cost={nx['objective']:.4f}  "
          f"build={nx['build_ms']:.2f}ms  solve={nx['solve_ms']:.2f}ms")
    print(f"  commitment: {nx['commitment']}")

    rows = [nx]
    # --- Competitor side (guarded). ---
    if args.nexus_only:
        print("pypsa: SKIPPED (--nexus-only requested) — nexus side only.")
    else:
        pp = solve_pypsa(demand, units, slack)
        if pp["status"] == "not_installed":
            print("pypsa: SKIPPED (not installed) — nexus side only.")
        else:
            rows.append(pp)
            print(f"pypsa  MILP: status={pp['status']}  cost={pp['objective']:.4f}")
            rel = abs(nx["objective"] - pp["objective"]) / max(abs(pp["objective"]), 1.0)
            print(f"  rel cost Δ: {rel:.2e}  (parity tol 1e-4)")
            assert rel < 1e-4, f"UC MILP H2H parity failed: nx={nx} pp={pp}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
