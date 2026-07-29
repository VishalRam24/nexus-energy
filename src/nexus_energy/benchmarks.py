"""
Phase 12: Performance Benchmarking Harness.

Provides standard test cases and benchmark utilities for:
- Model construction time
- Solve time
- Memory usage
- Scaling behaviour

Each benchmark returns a BenchmarkResult with reproducible timings.

Run all:
    from nexus_energy.benchmarks import run_all
    results = run_all()
    print_benchmark_report(results)
"""

from __future__ import annotations

import time
import platform
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np

from nexus_energy import EnergySystem, add_component


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    description: str
    n_buses: int
    n_components: int
    n_timesteps: int
    construction_time_s: float
    solve_time_s: float
    total_time_s: float
    total_cost: float
    status: str
    n_variables: int = 0
    n_constraints: int = 0
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"BenchmarkResult({self.name!r}, "
                f"build={1000*self.construction_time_s:.1f}ms, "
                f"solve={1000*self.solve_time_s:.1f}ms, "
                f"total={1000*self.total_time_s:.1f}ms)")


# ---------------------------------------------------------------------------
# Standard test cases
# ---------------------------------------------------------------------------

def build_3bus_island(T: int = 8760) -> EnergySystem:
    """Small island: PV + wind + battery + diesel + demand."""
    sys = EnergySystem("island_3bus")
    elec = sys.add_bus("elec")

    hours = np.arange(T)
    solar_cf = np.maximum(0, np.sin((hours % 24 - 6) * np.pi / 12))
    solar_cf[(hours % 24) < 6] = 0
    solar_cf[(hours % 24) >= 18] = 0
    wind_cf = 0.3 + 0.15 * np.sin(hours * np.pi / 36)
    wind_cf = np.clip(wind_cf, 0, 1)

    add_component(sys, "pv", "EC044", bus=elec, capacity=100, carrier_factor=solar_cf)
    add_component(sys, "wind", "EC062", bus=elec, capacity=50, carrier_factor=wind_cf)
    add_component(sys, "bat", "EC019", bus=elec, capacity=30)
    sys.add_generator("diesel", bus=elec, capacity=150, marginal_cost=200)

    demand = 60 + 30 * np.sin((hours - 3) * np.pi / 12)
    demand = np.maximum(demand, 40)
    sys.add_load("demand", bus=elec, amount=demand)

    return sys


def build_regional_system(n_buses: int = 30, T: int = 168) -> EnergySystem:
    """Regional system with N buses connected in a grid."""
    sys = EnergySystem(f"regional_{n_buses}bus")

    hours = np.arange(T)
    solar_cf = np.maximum(0, np.sin((hours % 24 - 6) * np.pi / 12))
    solar_cf[(hours % 24) < 6] = 0
    solar_cf[(hours % 24) >= 18] = 0

    buses = []
    for i in range(n_buses):
        bus = sys.add_bus(f"bus_{i}")
        buses.append(bus)

        # Each bus has: solar + gas backup + demand
        sys.add_generator(f"solar_{i}", bus=bus, capacity=50,
                          marginal_cost=0, carrier_factor=solar_cf)
        sys.add_generator(f"gas_{i}", bus=bus, capacity=100,
                          marginal_cost=50 + 0.5 * i)
        sys.add_load(f"demand_{i}", bus=bus,
                     amount=30 + 10 * np.sin(hours * np.pi / 12 + i))

    # Connect buses in a ring
    for i in range(n_buses):
        j = (i + 1) % n_buses
        sys.add_link(f"line_{i}_{j}", bus_from=buses[i], bus_to=buses[j],
                     capacity=30, efficiency=1.0, bidirectional=True)

    return sys


def build_sector_coupled(T: int = 168) -> EnergySystem:
    """Multi-sector system: electricity + heat + hydrogen."""
    from nexus_energy.sectors import (
        create_multi_carrier_system,
        create_power_to_hydrogen,
        create_heat_system,
    )

    sys, buses = create_multi_carrier_system(
        name="sector_coupled",
        carriers=["electricity", "heat", "hydrogen"],
    )

    hours = np.arange(T)
    solar_cf = np.maximum(0, np.sin((hours % 24 - 6) * np.pi / 12))
    solar_cf[(hours % 24) < 6] = 0
    solar_cf[(hours % 24) >= 18] = 0

    add_component(sys, "pv", "EC044", bus=buses["electricity"],
                  capacity=500, carrier_factor=solar_cf)
    add_component(sys, "wind", "EC062", bus=buses["electricity"],
                  capacity=200)
    sys.add_generator("gas_backup", bus=buses["electricity"],
                      capacity=300, marginal_cost=60,
                      emission_factor=0.4)

    # Links via sector coupling helpers would create extra buses; instead
    # add components directly to the pre-created buses:
    add_component(sys, "hp", "EC068",
                  bus=buses["electricity"], bus_to=buses["heat"],
                  capacity=50)
    add_component(sys, "elz", "EC008",
                  bus=buses["electricity"], bus_to=buses["hydrogen"],
                  capacity=100)
    add_component(sys, "bat", "EC019", bus=buses["electricity"], capacity=100)

    # Demands
    sys.add_load("elec_demand", bus=buses["electricity"],
                 amount=400 + 150 * np.sin((hours - 3) * np.pi / 12))
    sys.add_load("heat_demand", bus=buses["heat"],
                 amount=100 + 30 * np.sin(hours * np.pi / 12))
    sys.add_load("h2_demand", bus=buses["hydrogen"],
                 amount=20 * np.ones(T))

    return sys


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark(name: str, description: str, builder: Callable) -> BenchmarkResult:
    """Run one benchmark and collect timings."""
    t_start = time.perf_counter()
    sys = builder()
    t_build = time.perf_counter() - t_start

    t_solve_start = time.perf_counter()
    result = sys.optimise(verbose=False)
    t_solve = time.perf_counter() - t_solve_start

    t_total = time.perf_counter() - t_start

    n_vars = 0
    n_cons = 0
    if result._raw is not None:
        # nexus-opt result may expose these; best-effort
        n_vars = getattr(result._raw, "n_variables", 0) or 0
        n_cons = getattr(result._raw, "n_constraints", 0) or 0

    return BenchmarkResult(
        name=name,
        description=description,
        n_buses=sys.n_buses,
        n_components=sys.n_components,
        n_timesteps=sys._timesteps,
        construction_time_s=t_build,
        solve_time_s=t_solve,
        total_time_s=t_total,
        total_cost=result.total_cost if result.status == "optimal" else float("nan"),
        status=result.status,
        n_variables=n_vars,
        n_constraints=n_cons,
    )


# ---------------------------------------------------------------------------
# Standard benchmark suite
# ---------------------------------------------------------------------------

BENCHMARKS = [
    ("island_small",
     "3-bus island, 24h",
     lambda: build_3bus_island(T=24)),
    ("island_day",
     "3-bus island, 168h (1 week)",
     lambda: build_3bus_island(T=168)),
    ("island_year",
     "3-bus island, 8760h (full year)",
     lambda: build_3bus_island(T=8760)),
    ("regional_small",
     "30-bus regional, 168h",
     lambda: build_regional_system(n_buses=30, T=168)),
    ("regional_medium",
     "100-bus regional, 168h",
     lambda: build_regional_system(n_buses=100, T=168)),
    ("sector_coupled_week",
     "3-carrier (elec/heat/H2), 168h",
     lambda: build_sector_coupled(T=168)),
]


def run_all(bench_list=None) -> list[BenchmarkResult]:
    """Run all standard benchmarks and return results."""
    if bench_list is None:
        bench_list = BENCHMARKS
    results = []
    for name, desc, builder in bench_list:
        try:
            res = benchmark(name, desc, builder)
            results.append(res)
            print(f"  ✓ {name:25s}  {res}")
        except Exception as e:
            print(f"  ✗ {name:25s}  FAILED: {e}")
    return results


def print_benchmark_report(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Nexus-Energy Benchmark Report")
    lines.append(f"Platform: {platform.system()} {platform.machine()}, Python {platform.python_version()}")
    lines.append("=" * 80)
    lines.append(f"{'Name':<25}  {'Buses':>6}  {'Comps':>6}  {'T':>6}  "
                 f"{'Build':>10}  {'Solve':>10}  {'Total':>10}  {'Status':<10}")
    lines.append("-" * 80)
    for r in results:
        lines.append(
            f"{r.name:<25}  {r.n_buses:>6}  {r.n_components:>6}  {r.n_timesteps:>6}  "
            f"{1000*r.construction_time_s:>8.1f}ms  "
            f"{1000*r.solve_time_s:>8.1f}ms  "
            f"{1000*r.total_time_s:>8.1f}ms  {r.status:<10}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)
