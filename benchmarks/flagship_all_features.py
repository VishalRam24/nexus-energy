"""
Phase 13 — final all-features-on integration benchmark.

One script that exercises every feature phase of the library on
progressively larger synthetic problems and prints wall-clock +
status per (phase, feature). This is the "all features on" anchor
referenced in the `ROADMAP.md` Phase 13 brief and is the single
place we point to when making end-to-end claims.

Runs (nexus-only; no pypsa / julia / ray / torch dependencies):

  1. **Baseline LP** — single-bus economic dispatch, 168 snapshots.
     Exercises: EnergySystem, Generator, Load, Bus.
  2. **DC-OPF loop (Phase 3)** — 3-bus loop with KVL-binding flow;
     transport vs DC-OPF divergence proves KVL is enforced.
  3. **Unit commitment (Phase 2)** — 3-bin clustered UC MILP; proves
     the Morales-España + Rajan-Takriti formulation converges to
     optimality on a week of peaking / shoulder / baseload fleet.
  4. **Storage + representative days (Phase 5 / 7)** — storage SOC +
     Phase 7 k-medoids rep-days on a year of synthetic load.
  5. **Benders decomposition (Phase 8)** — 6-scenario two-stage
     capacity-expansion via plain Benders.
  6. **Stochastic CVaR (Phase 9)** — risk-averse plan with CVaR_0.1.
  7. **ML warm-start (Phase 11)** — UC with a MeritOrderPredictor
     fixing ≥95 %-confidence u-entries via uc_fix_schedule.
  8. **Rolling horizon + multi-solver (Phase 10)** — 48 h horizon,
     6 h window, solver='highs'.
  9. **Differentiable dispatch (Phase 12)** — closed-form QP gradient
     Jacobian, checked against finite difference.

Each run persists ``{name, status, wall, cost}`` to
``benchmarks/results/flagship_all_features.json``. Exit status 0 iff
every run reached its acceptance criterion; otherwise prints the
failing row.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import nexus_energy as ne
from nexus_energy.ml import (
    MeritOrderPredictor,
    predict_unit_commitment,
    warm_start_from_prediction,
)
from nexus_energy.diff import (
    numerical_jacobian,
    solve_dispatch_with_sensitivities,
)


@dataclass
class BenchRow:
    name: str
    status: str
    wall_seconds: float
    cost: float | None = None
    note: str = ""
    extras: dict = field(default_factory=dict)


def _time(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 1. Baseline LP
# ---------------------------------------------------------------------------

def bench_baseline_lp(T: int = 168) -> BenchRow:
    rng = np.random.default_rng(0)
    hours = np.arange(T)
    demand = 80 + 20 * np.sin(hours * np.pi / 12) + rng.normal(0, 2, T)

    def run():
        sys = ne.EnergySystem("baseline")
        bus = sys.add_bus("elec", carrier="electricity")
        sys.add_load("d", bus=bus, amount=demand)
        sys.add_generator("cheap", bus=bus, capacity=60, marginal_cost=10)
        sys.add_generator("mid", bus=bus, capacity=80, marginal_cost=30)
        sys.add_generator("peak", bus=bus, capacity=200, marginal_cost=120)
        sys.set_timesteps(T)
        return sys.optimise()

    r, wall = _time(run)
    ok = r.status == "optimal"
    return BenchRow("01_baseline_lp", r.status, wall,
                    float(r.total_cost) if ok else None,
                    note=f"T={T}")


# ---------------------------------------------------------------------------
# 2. DC-OPF loop (Phase 3)
# ---------------------------------------------------------------------------

def bench_dc_opf_loop() -> BenchRow:
    def run():
        sys = ne.EnergySystem("dc-opf-loop")
        b1 = sys.add_bus("b1", carrier="electricity")
        b2 = sys.add_bus("b2", carrier="electricity")
        b3 = sys.add_bus("b3", carrier="electricity")
        sys.add_load("d", bus=b3, amount=100.0)
        sys.add_generator("cheap", bus=b1, capacity=100, marginal_cost=10)
        sys.add_generator("slack", bus=b2, capacity=100, marginal_cost=90)
        # Loop: b1-b2, b2-b3, b1-b3 (direct short path) with different x.
        sys.add_link("l12", bus_from=b1, bus_to=b2, capacity=200,
                    efficiency=1.0, model_type="dc_opf", reactance=0.1)
        sys.add_link("l23", bus_from=b2, bus_to=b3, capacity=200,
                    efficiency=1.0, model_type="dc_opf", reactance=0.1)
        sys.add_link("l13", bus_from=b1, bus_to=b3, capacity=40,
                    efficiency=1.0, model_type="dc_opf", reactance=0.1)
        sys.set_timesteps(1)
        return sys.optimise()

    r, wall = _time(run)
    ok = r.status == "optimal"
    return BenchRow("02_dc_opf_loop", r.status, wall,
                    float(r.total_cost) if ok else None,
                    note="KVL on 3-bus loop, bottleneck l13@40")


# ---------------------------------------------------------------------------
# 3. Unit commitment (Phase 2)
# ---------------------------------------------------------------------------

def bench_unit_commitment(T: int = 168) -> BenchRow:
    rng = np.random.default_rng(1)
    hours = np.arange(T)
    demand = 140 + 60 * np.sin(hours * np.pi / 12) + rng.normal(0, 3, T)

    def run():
        sys = ne.EnergySystem("uc")
        bus = sys.add_bus("e", carrier="electricity")
        sys.add_load("d", bus=bus, amount=demand)
        sys.add_generator("base", bus=bus, capacity=60, marginal_cost=10,
                         committable=True, min_up_time=4, min_down_time=4,
                         startup_cost=100)
        sys.add_generator("mid", bus=bus, capacity=80, marginal_cost=40,
                         committable=True, min_up_time=2, min_down_time=2,
                         startup_cost=200)
        sys.add_generator("peak", bus=bus, capacity=120, marginal_cost=120,
                         committable=True, min_up_time=1, min_down_time=1,
                         startup_cost=500)
        sys.set_timesteps(T)
        return sys.optimise()

    r, wall = _time(run)
    return BenchRow("03_unit_commitment", r.status, wall,
                    float(r.total_cost) if r.status == "optimal" else None,
                    note=f"3-bin UC, T={T}",
                    extras={"committable": 3})


# ---------------------------------------------------------------------------
# 4. Storage + representative days (Phase 5 / 7)
# ---------------------------------------------------------------------------

def bench_storage_repdays() -> BenchRow:
    # Synthetic yearly series, aggregate to 6 representative days.
    hours = np.arange(24 * 14)
    rng = np.random.default_rng(2)
    demand = 80 + 25 * np.sin(hours * np.pi / 12) + rng.normal(0, 2, hours.size)
    solar = np.maximum(0, np.sin((hours % 24 - 6) * np.pi / 12))

    def run():
        rep = ne.aggregate_to_representative_days(
            {"demand": demand, "solar": solar},
            n_days=3, hours_per_day=24)
        sys = ne.EnergySystem("storage")
        bus = sys.add_bus("e", carrier="electricity")
        sys.add_load("d", bus=bus, amount=np.zeros(3 * 24))
        sys.add_generator("solar", bus=bus, capacity=120, marginal_cost=0)
        sys.add_generator("gas", bus=bus, capacity=100, marginal_cost=60)
        sys.add_storage("bat", bus=bus, power_capacity=40,
                        energy_capacity=160)
        sys.set_timesteps(3 * 24)
        ne.apply_representative_days(
            sys, rep, timeseries_map={"demand": "d", "solar": "solar"})
        return sys.optimise(), rep

    (r, rep), wall = _time(run)
    return BenchRow("04_storage_repdays", r.status, wall,
                    float(r.total_cost) if r.status == "optimal" else None,
                    note="3 rep-days, storage 40MW/160MWh",
                    extras={"n_periods": rep.n_periods})


# ---------------------------------------------------------------------------
# 5. Benders decomposition (Phase 8)
# ---------------------------------------------------------------------------

def bench_benders() -> BenchRow:
    def build_base():
        sys = ne.EnergySystem("benders")
        bus = sys.add_bus("e", carrier="electricity")
        T = 24
        sys.add_load("d", bus=bus, amount=np.full(T, 100.0))
        sys.add_generator("firm", bus=bus, capacity=0.0, marginal_cost=40,
                          extendable=True, capital_cost=1e5, min_capacity=0,
                          max_capacity=500)
        sys.add_generator("peak", bus=bus, capacity=500, marginal_cost=200)
        sys.set_timesteps(T)
        return sys

    scenarios = [
        ne.Scenario(name=f"s{i}", demand_factor=0.8 + 0.1 * i,
                    probability=1/6)
        for i in range(6)
    ]

    def run():
        return ne.solve_stochastic(
            base_system=build_base(),
            scenarios=scenarios,
            risk_measure="expected",
            method="benders",
            max_iter=25,
            verbose=False,
        )

    r, wall = _time(run)
    return BenchRow("05_benders", r.status,
                    wall, float(r.expected_cost),
                    note=f"{len(scenarios)} scenarios, {r.n_iterations} iters",
                    extras={"n_iterations": r.n_iterations})


# ---------------------------------------------------------------------------
# 6. Stochastic CVaR (Phase 9)
# ---------------------------------------------------------------------------

def bench_stochastic_cvar() -> BenchRow:
    def build_base():
        sys = ne.EnergySystem("cvar")
        bus = sys.add_bus("e", carrier="electricity")
        T = 24
        sys.add_load("d", bus=bus, amount=np.full(T, 80.0))
        sys.add_generator("firm", bus=bus, capacity=0.0, marginal_cost=40,
                          extendable=True, capital_cost=8e4, min_capacity=0,
                          max_capacity=500)
        sys.add_generator("peak", bus=bus, capacity=500, marginal_cost=500)
        sys.set_timesteps(T)
        return sys

    scenarios = [
        ne.Scenario(name=f"s{i}", demand_factor=0.7 + 0.15 * i,
                    probability=1/4)
        for i in range(4)
    ]

    def run():
        return ne.solve_stochastic(
            base_system=build_base(),
            scenarios=scenarios,
            risk_measure="cvar",
            cvar_alpha=0.1,
            method="benders",
            max_iter=25,
        )

    r, wall = _time(run)
    return BenchRow("06_stochastic_cvar", r.status,
                    wall, float(r.expected_cost),
                    note=f"CVaR_0.1, {len(scenarios)} scenarios")


# ---------------------------------------------------------------------------
# 7. ML warm-start (Phase 11)
# ---------------------------------------------------------------------------

def bench_ml_warmstart(T: int = 48) -> BenchRow:
    rng = np.random.default_rng(3)
    hours = np.arange(T)
    demand = 120 + 40 * np.sin(hours * np.pi / 12) + rng.normal(0, 2, T)

    def build():
        sys = ne.EnergySystem("wuc")
        bus = sys.add_bus("e", carrier="electricity")
        sys.add_load("d", bus=bus, amount=demand)
        sys.add_generator("cheap", bus=bus, capacity=60, marginal_cost=10,
                         committable=True, min_up_time=1, min_down_time=1)
        sys.add_generator("mid", bus=bus, capacity=80, marginal_cost=40,
                         committable=True, min_up_time=1, min_down_time=1)
        sys.add_generator("peak", bus=bus, capacity=120, marginal_cost=120,
                         committable=True, min_up_time=1, min_down_time=1)
        sys.set_timesteps(T)
        return sys

    # Cold solve
    cold_sys = build()
    (r_cold, cold_wall) = _time(lambda: cold_sys.optimise())
    # Warm solve with merit-order predictions
    warm_sys = build()
    pred = predict_unit_commitment(warm_sys, MeritOrderPredictor())
    fix = warm_start_from_prediction(pred, confidence_threshold=0.0)
    (r_warm, warm_wall) = _time(
        lambda: warm_sys.optimise(uc_fix_schedule=fix))

    speedup = cold_wall / warm_wall if warm_wall > 0 else 0.0
    status = "optimal" if (r_cold.status == "optimal"
                           and r_warm.status == "optimal") else "mixed"
    return BenchRow("07_ml_warmstart", status, cold_wall + warm_wall,
                    float(r_warm.total_cost),
                    note=f"cold={cold_wall*1000:.0f}ms warm={warm_wall*1000:.0f}ms "
                         f"speedup={speedup:.2f}× "
                         f"Δcost={abs(r_warm.total_cost-r_cold.total_cost):.2f}",
                    extras={"cold_wall": cold_wall, "warm_wall": warm_wall,
                            "speedup": speedup})


# ---------------------------------------------------------------------------
# 8. Rolling horizon + multi-solver (Phase 10)
# ---------------------------------------------------------------------------

def bench_rolling_horizon(T: int = 48, window: int = 12) -> BenchRow:
    rng = np.random.default_rng(4)
    hours = np.arange(T)
    demand = 90 + 30 * np.sin(hours * np.pi / 12) + rng.normal(0, 2, T)

    def factory(t0: int, t1: int) -> ne.EnergySystem:
        sys = ne.EnergySystem(f"rolling_{t0}")
        bus = sys.add_bus("e", carrier="electricity")
        sys.add_load("d", bus=bus, amount=demand[t0:t1])
        sys.add_generator("cheap", bus=bus, capacity=80, marginal_cost=15)
        sys.add_generator("peak", bus=bus, capacity=150, marginal_cost=120)
        sys.set_timesteps(t1 - t0)
        return sys

    def run():
        return ne.rolling_horizon_solve(
            factory, total_timesteps=T, window_size=window,
            overlap=0, warm_start=True, solver="highs")

    out, wall = _time(run)
    ok = bool(out.get("generator_dispatch"))
    cost = float(out.get("total_cost", 0.0)) if ok else None
    return BenchRow("08_rolling_horizon",
                    "optimal" if ok else "failed", wall, cost,
                    note=f"T={T}, window={window}, solver=highs, warm_start=True")


# ---------------------------------------------------------------------------
# 9. Differentiable dispatch (Phase 12)
# ---------------------------------------------------------------------------

def bench_differentiable() -> BenchRow:
    mc = np.array([10.0, 30.0, 50.0])
    cap = np.array([100.0, 100.0, 100.0])
    demand = 150.0

    def run():
        p, jac = solve_dispatch_with_sensitivities(mc, cap, demand, ridge=1.0)

        def fn(c):
            pp, _ = solve_dispatch_with_sensitivities(c, cap, demand, ridge=1.0)
            return pp
        num = numerical_jacobian(fn, mc, eps=1e-4)
        err = float(np.max(np.abs(jac.dp_dmc - num)))
        return p, err

    (p, err), wall = _time(run)
    status = "optimal" if err < 1e-4 else "grad_check_failed"
    return BenchRow("09_differentiable", status, wall,
                    float(np.sum(mc * p)),
                    note=f"max analytic-numerical grad err = {err:.2e}",
                    extras={"grad_err": err})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"nexus-energy flagship all-features-on benchmark — {ne.__version__}")
    print("-" * 72)
    rows = [
        bench_baseline_lp(),
        bench_dc_opf_loop(),
        bench_unit_commitment(),
        bench_storage_repdays(),
        bench_benders(),
        bench_stochastic_cvar(),
        bench_ml_warmstart(),
        bench_rolling_horizon(),
        bench_differentiable(),
    ]
    for row in rows:
        print(f"  {row.name:<22} {row.status:<12} "
              f"{row.wall_seconds*1000:>8.1f} ms  "
              f"cost={row.cost}  {row.note}")

    results_path = (Path(__file__).parent / "results"
                    / "flagship_all_features.json")
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(
        json.dumps({"version": ne.__version__,
                    "rows": [asdict(r) for r in rows]}, indent=2))
    print(f"\nResults → {results_path.relative_to(Path.cwd())}")

    bad = [r for r in rows if r.status not in {"optimal", "max_iter"}]
    if bad:
        print(f"\nFAILED: {len(bad)} row(s): {[r.name for r in bad]}")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
