"""Phase 11 acceptance bench — rolling UC, cold-start vs warm-start.

ROADMAP.md acceptance bar (Phase 11):

    On rolling UC for a region (3 zones, 48 h horizon, 1 year of
    weekly solves), warm-start reaches optimality >= 2x faster on
    the re-solve vs a cold start.

This bench builds a 3-zone clustered UC system, slides a 48-h window
across 1 year of synthetic load + VRE profiles in 7-day strides
(~52 windows), and times each window twice: once cold (no fixings)
and once warm (HistoricalNeighborPredictor trained on prior solves).
A MeritOrderPredictor row is also recorded as a no-training baseline.

The torch-backed GNNPredictor is not exercised here; the
HistoricalNeighborPredictor is the pure-numpy proxy that occupies the
same uc_fix_schedule slot. If we hit >= 2x with the numpy proxy, the
GNN slot is upside.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np

import nexus_energy as ne
from nexus_energy.ml import (
    HistoricalNeighborPredictor,
    MeritOrderPredictor,
    predict_unit_commitment,
    solve_with_warm_retry,
    warm_start_from_prediction,
)

THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"


# ---------------------------------------------------------------------------
# Annual profile generation
# ---------------------------------------------------------------------------

def _annual_profiles(seed: int = 11) -> dict[str, np.ndarray]:
    """Synthetic 8760 h load + wind + solar profiles for 3 zones."""
    rng = np.random.default_rng(seed)
    H = 8760
    hour = np.arange(H)
    day = hour / 24.0

    # Load: diurnal + seasonal + zone-specific noise. Tight enough vs
    # the fleet that thermal commitment binds (without it the MIP is
    # trivial to solve).
    diurnal = 280.0 * np.sin(2 * np.pi * (hour % 24) / 24.0 - np.pi / 3.0)
    seasonal = 140.0 * np.sin(2 * np.pi * day / 365.0 - np.pi / 2.0)
    base = 900.0
    loads = []
    for i in range(3):
        noise = 60.0 * rng.standard_normal(H)
        load = base + diurnal + seasonal + noise + 80.0 * i
        loads.append(np.clip(load, 200.0, None))

    # Wind: anti-diurnal-ish, high noise.
    wind = []
    for i in range(3):
        cf = 0.4 + 0.25 * np.sin(2 * np.pi * (hour % 24) / 24.0 + np.pi)
        cf += 0.1 * np.sin(2 * np.pi * day / 365.0)
        cf += 0.15 * rng.standard_normal(H)
        wind.append(np.clip(cf, 0.0, 1.0))

    # Solar: clipped sinusoid daytime only.
    solar = []
    for i in range(3):
        cf = np.maximum(np.sin(2 * np.pi * (hour % 24) / 24.0 - np.pi / 2.0), 0.0)
        cf *= 0.7 + 0.2 * np.sin(2 * np.pi * day / 365.0)
        cf += 0.05 * rng.standard_normal(H)
        solar.append(np.clip(cf, 0.0, 1.0))

    return {
        "load": np.stack(loads, axis=0),  # (3, 8760)
        "wind": np.stack(wind, axis=0),
        "solar": np.stack(solar, axis=0),
    }


def _build_window(
    profiles: dict[str, np.ndarray],
    t0: int,
    horizon: int,
    name: str = "rolling_uc",
) -> ne.EnergySystem:
    """Construct a 3-zone clustered UC system for a 48-h slice."""
    sys = ne.EnergySystem(name)
    sys.set_timesteps(horizon, dt=1.0)

    buses = [sys.add_bus(f"z{i}", carrier="electricity") for i in range(3)]
    for i, b in enumerate(buses):
        sys.add_load(f"ld_{i}", bus=b,
                     amount=profiles["load"][i, t0:t0 + horizon])

    # Per-zone fleet: many individually-committable thermal units so the
    # MIP branch tree at 48h horizon is non-trivial. Each unit is its own
    # binary u[t]; with 13 thermals/zone × 3 zones × 48 t we get ~1872
    # binaries pre-presolve.
    rng_capacity = np.random.default_rng(7)
    for i, b in enumerate(buses):
        for k in range(4):
            sys.add_generator(
                f"coal_{i}_{k}", bus=b,
                capacity=180 + 20 * (k - 1.5),
                marginal_cost=24 + 1.5 * k,
                p_min=70, committable=True,
                min_up_time=6, min_down_time=4,
                startup_cost=5000 + 500 * k, shutdown_cost=1200,
            )
        for k in range(5):
            sys.add_generator(
                f"ccgt_{i}_{k}", bus=b,
                capacity=130 + 20 * (k - 2),
                marginal_cost=42 + 1.0 * k,
                p_min=35, committable=True,
                min_up_time=3, min_down_time=2,
                startup_cost=1300 + 100 * k, shutdown_cost=400,
            )
        for k in range(4):
            sys.add_generator(
                f"oil_st_{i}_{k}", bus=b,
                capacity=110 + 10 * k,
                marginal_cost=80 + 2.0 * k,
                p_min=25, committable=True,
                min_up_time=2, min_down_time=2,
                startup_cost=400, shutdown_cost=200,
            )
        sys.add_generator(
            f"wind_{i}", bus=b, capacity=260, marginal_cost=0,
            carrier_factor=profiles["wind"][i, t0:t0 + horizon],
        )
        sys.add_generator(
            f"solar_{i}", bus=b, capacity=200, marginal_cost=0,
            carrier_factor=profiles["solar"][i, t0:t0 + horizon],
        )
        sys.add_generator(
            f"peaker_{i}", bus=b, capacity=280, marginal_cost=240,
        )

    for a, c in [(0, 1), (1, 2), (2, 0)]:
        sys.add_link(
            f"line_{a}{c}", bus_from=buses[a], bus_to=buses[c],
            capacity=350, efficiency=1.0, bidirectional=True, loss=0.02,
        )
    sys.set_spinning_reserve(0.05)
    return sys


# ---------------------------------------------------------------------------
# Per-window record
# ---------------------------------------------------------------------------

@dataclass
class WindowRecord:
    window: int
    t0: int
    cold_wall: float
    cold_cost: float
    cold_status: str
    warm_wall_hist: float | None = None
    warm_cost_hist: float | None = None
    warm_status_hist: str | None = None
    warm_retry_path_hist: str | None = None   # warm / warm_retry / cold / infeasible
    warm_retries_hist: int = 0
    fixings_hist: int = 0
    warm_wall_merit: float | None = None
    warm_cost_merit: float | None = None
    warm_status_merit: str | None = None
    warm_retry_path_merit: str | None = None
    warm_retries_merit: int = 0
    fixings_merit: int = 0


def _count_fixings(fix: dict[str, np.ndarray]) -> int:
    return int(sum(np.count_nonzero(~np.isnan(v)) for v in fix.values()))


def _solve_cold(profiles: dict[str, np.ndarray], t0: int, horizon: int):
    sys = _build_window(profiles, t0, horizon)
    t = time.perf_counter()
    res = sys.optimise(time_limit=120.0, gap=0.005)
    return sys, res, time.perf_counter() - t


def _solve_warm(profiles, t0, horizon, predictor, threshold: float,
                max_fix_fraction: float = 0.75, max_retries: int = 2):
    """Run the Phase 11.x retry driver: predict, fix_schedule with a global
    cap, retry-on-infeasible by halving the cap, fall back to cold if
    everything fails. Returns (sys, WarmStartOutcome, wall).
    """
    sys = _build_window(profiles, t0, horizon)
    t = time.perf_counter()
    out = solve_with_warm_retry(
        sys, predictor,
        confidence_threshold=threshold,
        max_fix_fraction=max_fix_fraction,
        max_retries=max_retries,
        cold_fallback=True,
        raise_on_failure=False,
        time_limit=120.0, gap=0.005,
    )
    return sys, out, time.perf_counter() - t


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    n_windows: int = 52,
    horizon: int = 48,
    stride: int = 168,
    burn_in: int = 4,
    threshold_hist: float = 0.8,
    threshold_merit: float = 0.99,
    max_fix_fraction: float = 0.75,
    max_retries: int = 2,
    seed: int = 11,
    progress: bool = True,
) -> dict:
    profiles = _annual_profiles(seed=seed)
    H = profiles["load"].shape[1]
    # Cap n_windows so the last window fits.
    max_windows = (H - horizon) // stride + 1
    n_windows = min(n_windows, max_windows)

    history = HistoricalNeighborPredictor(k_sys=8, k_step=5)
    merit = MeritOrderPredictor()

    records: list[WindowRecord] = []
    for w in range(n_windows):
        t0 = w * stride
        rec = WindowRecord(window=w, t0=t0, cold_wall=0.0,
                           cold_cost=0.0, cold_status="")

        # Cold solve.
        sys_cold, r_cold, cold_wall = _solve_cold(profiles, t0, horizon)
        rec.cold_wall = cold_wall
        rec.cold_cost = float(r_cold.total_cost)
        rec.cold_status = r_cold.status

        # Warm — historical neighbour (only after burn-in: bank must be non-empty).
        if w >= burn_in:
            _, out_h, h_wall = _solve_warm(
                profiles, t0, horizon, history, threshold_hist,
                max_fix_fraction=max_fix_fraction, max_retries=max_retries)
            rec.warm_wall_hist = h_wall
            if out_h.result is not None:
                rec.warm_cost_hist = float(out_h.result.total_cost)
                rec.warm_status_hist = out_h.result.status
            rec.warm_retry_path_hist = out_h.status
            rec.warm_retries_hist = out_h.retries
            rec.fixings_hist = out_h.n_pinned

        # Warm — merit order (always available, no training).
        _, out_m, m_wall = _solve_warm(
            profiles, t0, horizon, merit, threshold_merit,
            max_fix_fraction=max_fix_fraction, max_retries=max_retries)
        rec.warm_wall_merit = m_wall
        if out_m.result is not None:
            rec.warm_cost_merit = float(out_m.result.total_cost)
            rec.warm_status_merit = out_m.result.status
        rec.warm_retry_path_merit = out_m.status
        rec.warm_retries_merit = out_m.retries
        rec.fixings_merit = out_m.n_pinned

        # Feed cold solve into the historical bank for the *next* window.
        history.record(sys_cold, r_cold, tag=f"w{w}")
        records.append(rec)

        if progress:
            tag_h = (f"hist={rec.warm_wall_hist:.2f}s"
                     f"[{rec.warm_retry_path_hist},r{rec.warm_retries_hist},"
                     f"{rec.fixings_hist}pin]"
                     if rec.warm_wall_hist is not None else "hist=skip")
            print(
                f"win {w:02d} t0={t0:5d} | "
                f"cold={cold_wall:.2f}s cost={rec.cold_cost:.4e} | "
                f"merit={rec.warm_wall_merit:.2f}s"
                f"[{rec.warm_retry_path_merit},r{rec.warm_retries_merit},"
                f"{rec.fixings_merit}pin] | "
                f"{tag_h}",
                flush=True,
            )

    summary = _summarise(records, n_windows, burn_in, horizon, stride,
                         threshold_hist, threshold_merit)
    summary["config"]["max_fix_fraction"] = max_fix_fraction
    summary["config"]["max_retries"] = max_retries
    return summary


def _summarise(records, n_windows, burn_in, horizon, stride,
               th_hist, th_merit) -> dict:
    def _ok_warm(cost: float | None, status: str | None) -> bool:
        return (status == "optimal" and cost is not None
                and np.isfinite(cost))

    cold = [r.cold_wall for r in records if r.cold_status == "optimal"]
    merit = [r.warm_wall_merit for r in records
             if _ok_warm(r.warm_cost_merit, r.warm_status_merit)]
    hist = [r.warm_wall_hist for r in records
            if _ok_warm(r.warm_cost_hist, r.warm_status_hist)]

    # Cold timings restricted to post-burn-in windows for an apples-to-apples
    # comparison with the warm runs (which only fire after burn-in).
    cold_post_burn = [r.cold_wall for r in records
                      if r.window >= burn_in and r.cold_status == "optimal"]
    n_infeasible_merit = sum(
        1 for r in records if not _ok_warm(r.warm_cost_merit, r.warm_status_merit))
    n_infeasible_hist = sum(
        1 for r in records if r.window >= burn_in
        and not _ok_warm(r.warm_cost_hist, r.warm_status_hist))

    def stats(xs):
        if not xs:
            return None
        arr = np.asarray(xs)
        return {
            "n": int(arr.size),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    cold_med = float(np.median(cold_post_burn)) if cold_post_burn else 0.0
    hist_med = float(np.median(hist)) if hist else 0.0
    merit_med = float(np.median(merit)) if merit else 0.0
    speedup_hist = (cold_med / hist_med) if hist_med > 0 else 0.0
    speedup_merit = (cold_med / merit_med) if merit_med > 0 else 0.0

    cost_drift_hist = []
    cost_drift_merit = []
    for r in records:
        if r.cold_status != "optimal" or r.cold_cost == 0:
            continue
        if _ok_warm(r.warm_cost_hist, r.warm_status_hist):
            cost_drift_hist.append(
                (r.warm_cost_hist - r.cold_cost) / abs(r.cold_cost))
        if _ok_warm(r.warm_cost_merit, r.warm_status_merit):
            cost_drift_merit.append(
                (r.warm_cost_merit - r.cold_cost) / abs(r.cold_cost))

    return {
        "config": {
            "n_windows": n_windows,
            "horizon_hours": horizon,
            "stride_hours": stride,
            "burn_in_windows": burn_in,
            "confidence_threshold_hist": th_hist,
            "confidence_threshold_merit": th_merit,
        },
        "summary": {
            "cold_all": stats(cold),
            "cold_post_burn": stats(cold_post_burn),
            "warm_merit": stats(merit),
            "warm_hist": stats(hist),
            "speedup_warm_hist_vs_cold": speedup_hist,
            "speedup_warm_merit_vs_cold": speedup_merit,
            "cost_drift_hist_max": (float(np.max(np.abs(cost_drift_hist)))
                                    if cost_drift_hist else 0.0),
            "cost_drift_merit_max": (float(np.max(np.abs(cost_drift_merit)))
                                     if cost_drift_merit else 0.0),
            "n_infeasible_warm_hist": n_infeasible_hist,
            "n_infeasible_warm_merit": n_infeasible_merit,
        },
        "windows": [
            {
                "window": r.window, "t0": r.t0,
                "cold_wall": r.cold_wall, "cold_cost": r.cold_cost,
                "cold_status": r.cold_status,
                "warm_wall_merit": r.warm_wall_merit,
                "warm_cost_merit": r.warm_cost_merit,
                "warm_status_merit": r.warm_status_merit,
                "warm_retry_path_merit": r.warm_retry_path_merit,
                "warm_retries_merit": r.warm_retries_merit,
                "fixings_merit": r.fixings_merit,
                "warm_wall_hist": r.warm_wall_hist,
                "warm_cost_hist": r.warm_cost_hist,
                "warm_status_hist": r.warm_status_hist,
                "warm_retry_path_hist": r.warm_retry_path_hist,
                "warm_retries_hist": r.warm_retries_hist,
                "fixings_hist": r.fixings_hist,
            }
            for r in records
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-windows", type=int, default=52)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--stride", type=int, default=168)
    p.add_argument("--burn-in", type=int, default=4)
    p.add_argument("--threshold-hist", type=float, default=0.8)
    p.add_argument("--threshold-merit", type=float, default=0.99)
    p.add_argument("--max-fix-fraction", type=float, default=0.75)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default=str(RESULTS_DIR / "phase11_rolling_uc.json"))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    out = run(
        n_windows=args.n_windows, horizon=args.horizon, stride=args.stride,
        burn_in=args.burn_in,
        threshold_hist=args.threshold_hist,
        threshold_merit=args.threshold_merit,
        max_fix_fraction=args.max_fix_fraction,
        max_retries=args.max_retries,
        seed=args.seed, progress=not args.quiet,
    )
    Path(args.out).write_text(json.dumps(out, indent=2))

    s = out["summary"]
    print()
    print("--- Phase 11 rolling-UC summary ---")
    if s["cold_post_burn"]:
        print(f"cold (post-burn-in): median={s['cold_post_burn']['median']:.3f}s "
              f"p90={s['cold_post_burn']['p90']:.3f}s "
              f"n={s['cold_post_burn']['n']}")
    if s["warm_hist"]:
        print(f"warm hist          : median={s['warm_hist']['median']:.3f}s "
              f"p90={s['warm_hist']['p90']:.3f}s "
              f"n={s['warm_hist']['n']}")
    if s["warm_merit"]:
        print(f"warm merit         : median={s['warm_merit']['median']:.3f}s "
              f"p90={s['warm_merit']['p90']:.3f}s "
              f"n={s['warm_merit']['n']}")
    print(f"speedup warm-hist  vs cold: {s['speedup_warm_hist_vs_cold']:.2f}x")
    print(f"speedup warm-merit vs cold: {s['speedup_warm_merit_vs_cold']:.2f}x")
    print(f"cost drift hist max : {s['cost_drift_hist_max']*100:.3f}%")
    print(f"cost drift merit max: {s['cost_drift_merit_max']*100:.3f}%")
    print(f"infeasible warm-hist : {s['n_infeasible_warm_hist']}")
    print(f"infeasible warm-merit: {s['n_infeasible_warm_merit']}")
    print(f"wrote {args.out}")

    target = 2.0
    ok = s["speedup_warm_hist_vs_cold"] >= target
    print(f"acceptance bar (>= {target}x on hist re-solve): "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
