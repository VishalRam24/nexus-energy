"""
Head-to-head benchmark: nexus-energy vs PyPSA on identical problems.

Builds the same LP dispatch problem in both frameworks, uses the same
solver (HiGHS), measures: model construction time, solve time, memory,
total cost. Prints a comparison table.
"""

import argparse
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np


def build_test_case(n_buses=10, T=168, seed=42):
    """Build a synthetic dispatch problem."""
    rng = np.random.RandomState(seed)
    hours = np.arange(T)

    # Time series shared across buses (with per-bus variation)
    base_solar = np.maximum(0, np.sin((hours % 24 - 6) * np.pi / 12))
    base_solar[(hours % 24) < 6] = 0
    base_solar[(hours % 24) >= 18] = 0

    bus_data = []
    for i in range(n_buses):
        solar_cf = np.clip(base_solar * (0.8 + 0.4 * rng.random()), 0, 1)
        demand = 50 + 30 * rng.random() + 20 * np.sin(hours * np.pi / 12 + i)
        bus_data.append({
            "name": f"bus_{i}",
            "solar_cf": solar_cf,
            "demand": demand,
            "solar_cap": 50 + 20 * rng.random(),
            "gas_cap": 100,
            "gas_cost": 40 + 10 * rng.random(),
        })
    return bus_data, T


def run_nexus_energy(bus_data, T):
    """Build and solve the problem in nexus-energy.

    Timing and memory are measured in SEPARATE passes: ``tracemalloc`` is an
    allocation-level instrument that inflates wall-clock by ~100× for code that
    does heavy Python-side allocation (nexus's build + result extraction), while
    barely touching C-backed work (PyPSA's HiGHS solve). Timing the solve *under*
    tracemalloc therefore reports instrumentation overhead, not solve time
    (confirmed: 0.10 s untraced vs 10.5 s traced on 30 bus × 168 h). We time
    untraced and take peak memory from a second, untimed traced pass so both
    figures are honest and the nexus-vs-PyPSA comparison is apples-to-apples.
    """
    from nexus_energy import EnergySystem, add_component

    def _build():
        sys = EnergySystem("bench")
        buses = []
        for b in bus_data:
            bus = sys.add_bus(b["name"])
            buses.append(bus)
            add_component(sys, f"solar_{b['name']}", "EC044", bus=bus,
                          capacity=b["solar_cap"], carrier_factor=b["solar_cf"])
            sys.add_generator(f"gas_{b['name']}", bus=bus,
                              capacity=b["gas_cap"],
                              marginal_cost=b["gas_cost"])
            sys.add_load(f"demand_{b['name']}", bus=bus, amount=b["demand"])
        for i in range(len(buses)):
            j = (i + 1) % len(buses)
            sys.add_link(f"line_{i}_{j}", bus_from=buses[i], bus_to=buses[j],
                         capacity=30, efficiency=1.0, bidirectional=True)
        return sys

    # Timing pass — NOT under tracemalloc.
    t0 = time.perf_counter()
    sys = _build()
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = sys.optimise(solver="highs", verbose=False)
    t_solve = time.perf_counter() - t0

    # Memory pass — traced, untimed.
    tracemalloc.start()
    sys2 = _build()
    sys2.optimise(solver="highs", verbose=False)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "framework": "nexus-energy",
        "status": result.status,
        "total_cost": result.total_cost,
        "build_time_ms": t_build * 1000,
        "solve_time_ms": t_solve * 1000,
        "peak_memory_mb": peak_mem / 1024 / 1024,
    }


def run_pypsa(bus_data, T):
    """Build and solve the same problem in PyPSA."""
    try:
        import pandas as pd
        # PyPSA 1.2.x + recent pandas default to Arrow-backed strings, which
        # xarray's .sel() can't index (TypeError: Invalid array type
        # ArrowStringArray). Force classic python string storage before pypsa
        # builds its component frames.
        try:
            pd.set_option("future.infer_string", False)
            pd.set_option("mode.string_storage", "python")
        except Exception:
            pass
        import pypsa
    except ImportError:
        return {"framework": "pypsa", "status": "not_installed"}

    def _build():
        n = pypsa.Network()
        snapshots = pd.date_range("2024-01-01", periods=T, freq="h")
        n.set_snapshots(snapshots)
        for b in bus_data:
            n.add("Bus", b["name"])
            n.add("Generator", f"solar_{b['name']}",
                  bus=b["name"], p_nom=b["solar_cap"],
                  p_max_pu=pd.Series(b["solar_cf"], index=snapshots),
                  marginal_cost=0)
            n.add("Generator", f"gas_{b['name']}",
                  bus=b["name"], p_nom=b["gas_cap"],
                  marginal_cost=b["gas_cost"])
            n.add("Load", f"demand_{b['name']}",
                  bus=b["name"],
                  p_set=pd.Series(b["demand"], index=snapshots))
        for i in range(len(bus_data)):
            j = (i + 1) % len(bus_data)
            n.add("Link", f"line_{i}_{j}",
                  bus0=bus_data[i]["name"], bus1=bus_data[j]["name"],
                  p_nom=30, p_min_pu=-1, efficiency=1.0)
        return n

    # Timing pass — NOT under tracemalloc (same protocol as run_nexus_energy).
    t0 = time.perf_counter()
    n = _build()
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    try:
        status, _ = n.optimize(solver_name="highs")
    except Exception as e:
        status = f"error: {e}"
    t_solve = time.perf_counter() - t0

    total_cost = float("nan")
    try:
        total_cost = float(n.objective)
    except Exception:
        pass

    # Memory pass — traced, untimed.
    tracemalloc.start()
    n2 = _build()
    try:
        n2.optimize(solver_name="highs")
    except Exception:
        pass
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "framework": "pypsa",
        "status": str(status),
        "total_cost": total_cost,
        "build_time_ms": t_build * 1000,
        "solve_time_ms": t_solve * 1000,
        "peak_memory_mb": peak_mem / 1024 / 1024,
    }


def run_comparison(n_buses, T):
    """Run both frameworks and compare. Returns the captured rows."""
    bus_data, _ = build_test_case(n_buses=n_buses, T=T)

    print(f"\n{'='*80}")
    print(f"Test: {n_buses} buses × {T} timesteps")
    print(f"{'='*80}")

    ne_result = run_nexus_energy(bus_data, T)
    print(f"  nexus-energy:  build={ne_result['build_time_ms']:8.1f}ms  "
          f"solve={ne_result['solve_time_ms']:8.1f}ms  "
          f"mem={ne_result['peak_memory_mb']:6.1f}MB  "
          f"cost={ne_result.get('total_cost', 0):.2f}")

    pypsa_result = run_pypsa(bus_data, T)
    case = {"n_buses": n_buses, "T": T,
            "nexus": ne_result, "pypsa": pypsa_result}
    if pypsa_result["status"] == "not_installed":
        print("  pypsa:         SKIPPED (not installed)")
        return case

    print(f"  pypsa:         build={pypsa_result['build_time_ms']:8.1f}ms  "
          f"solve={pypsa_result['solve_time_ms']:8.1f}ms  "
          f"mem={pypsa_result['peak_memory_mb']:6.1f}MB  "
          f"cost={pypsa_result.get('total_cost', 0):.2f}")

    # Comparison
    if pypsa_result["build_time_ms"] > 0 and ne_result["build_time_ms"] > 0:
        build_speedup = pypsa_result["build_time_ms"] / ne_result["build_time_ms"]
        print(f"  ▸ Build speedup:  {build_speedup:.1f}x")
    if pypsa_result["solve_time_ms"] > 0 and ne_result["solve_time_ms"] > 0:
        solve_speedup = pypsa_result["solve_time_ms"] / ne_result["solve_time_ms"]
        print(f"  ▸ Solve speedup:  {solve_speedup:.1f}x")
    if pypsa_result["peak_memory_mb"] > 0 and ne_result["peak_memory_mb"] > 0:
        mem_ratio = pypsa_result["peak_memory_mb"] / ne_result["peak_memory_mb"]
        print(f"  ▸ Memory ratio:   {mem_ratio:.1f}x (higher = nexus uses less)")

    # Sanity: costs should match (within solver tolerance)
    if not np.isnan(pypsa_result.get("total_cost", float("nan"))):
        ne_c = ne_result.get("total_cost", float("nan"))
        pp_c = pypsa_result.get("total_cost", float("nan"))
        if not (np.isnan(ne_c) or np.isnan(pp_c)):
            rel = abs(ne_c - pp_c) / max(abs(pp_c), 1)
            if rel > 0.01:
                print(f"  ⚠ Cost mismatch: {100*rel:.2f}% (should be < 1%)")
            else:
                print(f"  ✓ Costs match within {100*rel:.3f}%")
    return case


def _ndarray_safe(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    raise TypeError(f"unserialisable: {type(o)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow 8760h case")
    args = ap.parse_args()

    print(f"Platform: {platform.system()} {platform.machine()}, "
          f"Python {platform.python_version()}")

    cases = [
        (3, 24),
        (10, 168),
        (30, 168),
    ]
    if not args.quick:
        cases.append((10, 8760))
    cases.append((50, 168))

    rows = []
    for n_buses, T in cases:
        rows.append(run_comparison(n_buses=n_buses, T=T))

    if args.json:
        out = {"platform": platform.platform(),
               "python": platform.python_version(),
               "cases": rows}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(out, indent=2, default=_ndarray_safe))
        print(f"\nwrote {args.json}")
