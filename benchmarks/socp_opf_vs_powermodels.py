"""
N_En_Phase 17.5 — Head-to-head: nexus SOCP AC-OPF vs PowerModels.jl SOCWR.

Same 3-bus radial case as `tests/phase_10/test_phase10.py::test_socp_opf_three_bus_radial`:
  b1 (gen) — l12 — b2 — l23 — b3 (load)
  r = 0.01, x = 0.10 pu per branch; rate_a = 2.0 pu
  load = 0.5 pu at b3; gen 0..5 pu at b1; linear MC = 30

Compares total cost, generator dispatch, voltage magnitudes, and branch
losses against `run_socp_3bus.jl` output (which we load from the same
directory). Wall-clock is captured on both sides but Julia's includes one
full warm compile so the Julia *solve* time (from PowerModels' own
`solve_time` field) is the fair number.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import nexus_energy as ne


_REPO_ROOT = Path(__file__).resolve().parents[2]

# Reference PowerModels.jl checkout lives outside the package. Override with
# NEXUS_POWERMODELS_DIR if you keep the clone elsewhere.
JULIA_PROJECT = Path(
    os.environ.get(
        "NEXUS_POWERMODELS_DIR",
        _REPO_ROOT / "test_projects/test_project_1/julia/PowerModels.jl",
    )
)
JULIA_SCRIPT = JULIA_PROJECT / "run_socp_3bus.jl"


def build_nexus_case() -> ne.EnergySystem:
    sys = ne.EnergySystem("socp_3bus_h2h")
    b1 = sys.add_bus("b1", carrier="electricity")
    b2 = sys.add_bus("b2", carrier="electricity")
    b3 = sys.add_bus("b3", carrier="electricity")

    sys.add_generator("slack", bus=b1, capacity=5.0, marginal_cost=30.0)
    sys.add_load("d3", bus=b3, amount=0.5)

    l12 = sys.add_link("l12", bus_from=b1, bus_to=b2, capacity=2.0)
    l23 = sys.add_link("l23", bus_from=b2, bus_to=b3, capacity=2.0)
    for link in (l12, l23):
        link.resistance = 0.01
        link.reactance = 0.10
        link.s_max = 2.0
        link.model_type = "socp_opf"
    return sys


def powermodels_available() -> bool:
    """True only if a Julia binary AND the PowerModels project script exist."""
    import shutil
    return shutil.which("julia") is not None and JULIA_SCRIPT.exists()


def run_powermodels() -> dict:
    out = subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", str(JULIA_SCRIPT)],
        capture_output=True, text=True, check=True,
    )
    # Script prints a JSON object (with some leading banners); extract it.
    stdout = out.stdout
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < 0:
        raise RuntimeError(f"PowerModels JSON not found in stdout:\n{stdout}")
    return json.loads(stdout[start : end + 1])


def run_nexus() -> dict:
    system = build_nexus_case()

    # Warmup (match Julia's warmup — first solve compiles cones)
    _ = ne.solve_socp_opf(system)

    t0 = time.perf_counter()
    res = ne.solve_socp_opf(system)
    wall = time.perf_counter() - t0

    return dict(
        backend="nexus-energy solve_socp_opf (Clarabel)",
        status=res.status,
        total_cost=float(res.total_cost),
        solve_time=wall,
        clarabel_solve_time=float(res.solve_time),
        voltage_mag={k: float(v) for k, v in res.voltage_mag.items()},
        gen_p={k: float(v) for k, v in res.gen_p.items()},
        gen_q={k: float(v) for k, v in res.gen_q.items()},
        branch_p={k: float(v) for k, v in res.branch_p.items()},
        branch_loss={k: float(v) for k, v in res.branch_loss.items()},
    )


def main():
    # --- Nexus side always runs (the deliverable). ---
    print("Running nexus SOCP …", flush=True)
    nx = run_nexus()
    assert nx["status"] == "optimal", f"nexus SOCP not optimal: {nx['status']}"
    print(f"  nexus SOCP optimal: total_cost={nx['total_cost']:.6f}")

    # --- Competitor side (guarded): PowerModels.jl via Julia. ---
    nexus_only = "--nexus-only" in sys.argv
    if nexus_only or not powermodels_available():
        why = ("--nexus-only requested" if nexus_only
               else "julia binary or run_socp_3bus.jl absent")
        print(f"PowerModels.jl SKIPPED ({why}) — nexus side only.")
        out_path = (Path(__file__).resolve().parent
                    / "results/socp_opf_vs_powermodels.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            dict(powermodels=None, nexus=nx, competitor="skipped"), indent=2))
        print(f"wrote {out_path}")
        return

    print("Running PowerModels.jl SOCWR …", flush=True)
    pm = run_powermodels()

    def fmt(x, w=12, p=6):
        return f"{x:>{w}.{p}f}" if isinstance(x, float) else f"{str(x):>{w}}"

    print()
    print("=" * 78)
    print("SOCP AC-OPF H2H — 3-bus radial (load 0.5 pu; r=0.01, x=0.10; rate_a=2.0)")
    print("=" * 78)
    print(f"{'field':<24}{'PowerModels.jl':>18}{'nexus':>18}{'|Δ|':>18}")
    print("-" * 78)

    rows = [
        ("total_cost", pm["total_cost"], nx["total_cost"]),
        ("gen_p[slack]", pm["gen_p"]["slack"], nx["gen_p"]["slack"]),
        ("|V|[b1]", pm["voltage_mag"]["b1"], nx["voltage_mag"]["b1"]),
        ("|V|[b2]", pm["voltage_mag"]["b2"], nx["voltage_mag"]["b2"]),
        ("|V|[b3]", pm["voltage_mag"]["b3"], nx["voltage_mag"]["b3"]),
        ("P[l12]", pm["branch_p"]["l12"], nx["branch_p"]["l12"]),
        ("P[l23]", pm["branch_p"]["l23"], nx["branch_p"]["l23"]),
        ("loss[l12]", pm["branch_loss"]["l12"], nx["branch_loss"]["l12"]),
        ("loss[l23]", pm["branch_loss"]["l23"], nx["branch_loss"]["l23"]),
    ]
    max_abs_delta = 0.0
    max_rel_delta = 0.0
    for name, a, b in rows:
        delta = abs(a - b)
        max_abs_delta = max(max_abs_delta, delta)
        scale = max(abs(a), abs(b), 1e-12)
        max_rel_delta = max(max_rel_delta, delta / scale)
        print(f"{name:<24}{fmt(a, 18, 8)}{fmt(b, 18, 8)}{fmt(delta, 18, 2 if delta == 0 else 8)}")

    print("-" * 78)
    print(f"max |Δ| across all fields : {max_abs_delta:.3e}")
    print(f"max relative Δ            : {max_rel_delta:.3e}")
    print()
    print(f"wall-clock (fair)         : "
          f"PowerModels {pm['pm_solve_time']*1000:.2f} ms  |  "
          f"nexus {nx['clarabel_solve_time']*1000:.2f} ms")
    print(f"speedup (nexus vs PM)     : "
          f"{pm['pm_solve_time'] / max(nx['clarabel_solve_time'], 1e-9):.2f}×")
    print()

    # Persist for the scorecard / COMPARISON doc
    out_path = (Path(__file__).resolve().parent
                / "results/socp_opf_vs_powermodels.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        dict(powermodels=pm, nexus=nx,
             max_abs_delta=max_abs_delta, max_rel_delta=max_rel_delta),
        indent=2))
    print(f"wrote {out_path}")

    # Parity gate
    if max_rel_delta > 1e-4:
        print(f"PARITY FAIL: max relative Δ {max_rel_delta:.3e} > 1e-4")
        sys.exit(1)
    print(f"PARITY PASS: max relative Δ {max_rel_delta:.3e} ≤ 1e-4")


if __name__ == "__main__":
    main()
