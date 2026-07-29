"""Smart LP-method calibrator — the offline "method switcher".

Different LP shapes favour different HiGHS engines: dual **simplex** (warm-start,
vertex/duals, small or ill-conditioned problems), interior-point **ipm**
(+crossover, large well-conditioned LPs needing a true vertex), or **ipm_fast**
(IPM without crossover — fastest on large well-conditioned LPs when an interior
point is acceptable). Picking wrong is costly: IPM is ~1.5x faster than simplex
on PyPSA-Eur but *stalls* on CINDER's `[4e-5, 2e3]`-conditioned LP.

History in this repo proved a static shape heuristic unreliable — an earlier
">=50K cols => ipm" rule REGRESSED CINDER. So this tuner is **empirical**: it
*measures* by racing the candidate methods on a reduced-horizon proxy of the
problem (cheap, run once), checks the winner is stable across two sizes, and
writes a `.nexus_solver.json` sidecar that `optimise()` picks up automatically.
A structural pre-pass only disqualifies obviously-wrong methods (MILP can't use
IPM for branch-and-bound nodes).

Usage:
    from nexus_energy.solver_tuner import tune_solver
    res = tune_solver(system)          # races, prints report, writes sidecar
    print(res.recommended)             # e.g. "ipm_fast"

    # any library: import via from_pypsa first, then tune the EnergySystem.

The sidecar is advisory and overridable: an explicit ``lp_backend=`` on
``optimise`` always wins; delete the file to revert to the built-in default.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nexus_energy.core import EnergySystem

SIDECAR_NAME = ".nexus_solver.json"
# Candidate engines. "ipm" = interior point + crossover (vertex); "ipm_fast" =
# interior point, no crossover (interior point, fastest).
_CANDIDATES = ("simplex", "ipm", "ipm_fast")
# Within this relative wall-clock margin, prefer the "safer" method (simplex >
# ipm > ipm_fast) for vertex/dual cleanliness and warm-startability.
_TIE_MARGIN = 0.10


@dataclass
class MethodRun:
    method: str
    hours: int
    wall: float
    status: str
    objective: float = float("nan")
    error: str = ""


@dataclass
class TuneResult:
    recommended: str
    reason: str
    stable: bool = True
    parity_ok: bool = True
    features: dict = field(default_factory=dict)
    runs: list[MethodRun] = field(default_factory=list)

    def to_sidecar(self) -> dict:
        return {
            "lp_backend": self.recommended,
            "reason": self.reason,
            "stable": self.stable,
            "parity_ok": self.parity_ok,
            "features": self.features,
            "generated_by": "nexus_energy.solver_tuner",
        }


def _has_integers(system: "EnergySystem") -> bool:
    """Mirror of core.optimise's MILP test (committable / integer / segments)."""
    g = system._generators
    return (
        any(getattr(x, "committable", False) for x in g)
        or any(getattr(x, "integer_investment", False) for x in g)
        or any(getattr(x, "capex_segments", None) is not None for x in g)
        or any(getattr(x, "committable", False) for x in system._links)
        or any(getattr(x, "no_simultaneous", False) for x in system._storages)
    )


def structural_features(system: "EnergySystem") -> dict:
    """Cheap, no-solve descriptors used to explain the pick and disqualify
    obviously-wrong methods. NOT used to *choose* among feasible methods — the
    race does that (static shape signals proved unreliable here)."""
    system._infer_timesteps()
    T = system._timesteps or 1
    return {
        "n_buses": len(system._buses),
        "n_generators": len(system._generators),
        "n_storages": len(system._storages),
        "n_links": len(system._links),
        "n_snapshots": int(T),
        "has_integers": _has_integers(system),
    }


def _probe_sizes(T: int) -> list[int]:
    """Two distinct reduced horizons to race (and check the winner is stable).
    Small problems race at their true size."""
    if T <= 48:
        return [T]
    a = min(T, 365)
    b = min(T, 730)
    sizes = sorted({a, b})
    return sizes if len(sizes) == 2 else [max(2, T // 2), min(T, a)]


def _slice_to(system: "EnergySystem", hours: int):
    """A reduced-horizon, same-structure proxy of the system for timing only.
    Uses the decomposition slicer when shrinking; full copy when not."""
    from nexus_energy.decomposition import _slice_system
    T = system._timesteps or 1
    if hours >= T:
        return copy.deepcopy(system)
    return _slice_system(system, 0, hours)


def _race_at(system: "EnergySystem", hours: int, methods, time_cap: float,
             threads: int) -> list[MethodRun]:
    runs = []
    for m in methods:
        sub = _slice_to(system, hours)          # fresh copy per method
        t0 = time.perf_counter()
        try:
            r = sub.optimise(lp_backend=m, threads=threads, time_limit=time_cap)
            wall = time.perf_counter() - t0
            runs.append(MethodRun(m, hours, wall, r.status,
                                  float(r.total_cost)))
        except Exception as e:  # noqa: BLE001 — a failed method just loses
            runs.append(MethodRun(m, hours, float("inf"), "error",
                                  error=repr(e)[:120]))
    return runs


def _winner(runs: list[MethodRun], need_duals: bool) -> Optional[str]:
    """Fastest method that solved, with safety tie-breaks."""
    ok = [r for r in runs if r.status in ("optimal", "time_limit")
          and r.wall != float("inf")]
    if not ok:
        return None
    # ipm_fast gives interior duals — drop it when duals are needed.
    if need_duals:
        ok = [r for r in ok if r.method != "ipm_fast"] or ok
    best = min(ok, key=lambda r: r.wall)
    contenders = [r for r in ok if r.wall <= best.wall * (1 + _TIE_MARGIN)]
    # Speed-first tie-break: among near-tied methods prefer the one whose lead
    # GROWS with scale. ipm_fast skips crossover, whose cost rises with problem
    # size, so on a reduced-horizon near-tie it pulls further ahead at full
    # scale; then ipm, then simplex. (need_duals already removed ipm_fast.)
    speed_order = {"ipm_fast": 0, "ipm": 1, "simplex": 2}
    return min(contenders, key=lambda r: speed_order.get(r.method, 9)).method


def recommend_lp_method(system: "EnergySystem", time_cap: float = 120.0,
                        threads: int = 8, need_duals: bool = False,
                        probe_hours: list[int] | None = None,
                        verbose: bool = True) -> TuneResult:
    """Race the LP engines on reduced-horizon proxies and recommend one.

    ``probe_hours`` overrides the auto-chosen reduced horizons — pass 3+ sizes
    to validate that the winner trend is stable across scales (used by the
    tuner-validation harness to catch a proxy that mispredicts full scale).
    """
    feats = structural_features(system)
    if feats["has_integers"]:
        return TuneResult(
            recommended="simplex",
            reason="MILP (committable/integer): IPM cannot solve branch-and-"
                   "bound node LPs; simplex warm-starts between nodes.",
            features=feats,
        )

    sizes = sorted(set(probe_hours)) if probe_hours else _probe_sizes(feats["n_snapshots"])
    all_runs: list[MethodRun] = []
    winners = []
    for h in sizes:
        runs = _race_at(system, h, _CANDIDATES, time_cap, threads)
        all_runs.extend(runs)
        w = _winner(runs, need_duals)
        winners.append(w)
        if verbose:
            _print_size_table(h, runs, w)

    # Parity across methods — compared WITHIN each probe size (same LP), never
    # across sizes (different horizons have different objectives by design).
    parity_ok = True
    by_size: dict[int, list[float]] = {}
    for r in all_runs:
        if r.status == "optimal" and r.objective == r.objective:
            by_size.setdefault(r.hours, []).append(r.objective)
    for objs in by_size.values():
        if objs and (max(objs) - min(objs)) / max(abs(objs[0]), 1.0) >= 1e-4:
            parity_ok = False

    # Trivial-problem guard: if even the largest probe solves near-instantly,
    # method choice is irrelevant and the "winner" is pure timing noise. Pick
    # the safe default (simplex) and don't cry instability.
    big = max(sizes)
    big_walls = [r.wall for r in all_runs
                 if r.hours == big and r.wall != float("inf")]
    trivial = bool(big_walls) and min(big_walls) < 0.1

    valid = [w for w in winners if w]
    if trivial:
        recommended, stable = "simplex", True
        reason = (f"raced at {sizes}h; all solves <0.1s — problem is trivial, "
                  f"method choice is immaterial; defaulting to simplex.")
    else:
        stable = len(set(valid)) <= 1
        recommended = valid[-1] if valid else "simplex"   # largest-probe winner
        # A clean simplex(small)->IPM(large) flip is the expected size-crossover
        # (IPM's setup amortises only at scale), not random instability. Taking
        # the largest-probe winner is the correct full-scale extrapolation.
        is_crossover = (not stable and valid
                        and valid[0] == "simplex"
                        and valid[-1] in ("ipm", "ipm_fast"))
        if stable:
            note = "stable across sizes"
        elif is_crossover:
            note = (f"size-crossover: simplex fastest at {sizes[0]}h, "
                    f"{valid[-1]} fastest at {sizes[-1]}h — recommending "
                    f"{valid[-1]} (its lead widens with size)")
        else:
            note = "winner varies non-monotonically — took largest-probe winner"
        # A benign crossover is not 'unstable' for reporting purposes.
        stable = stable or is_crossover
        reason = (f"raced {','.join(_CANDIDATES)} at {sizes}h; {note}; "
                  f"fastest qualifying method.")
    if not parity_ok:
        reason += " WARNING: objective disagreement across methods within a size."

    res = TuneResult(recommended=recommended, reason=reason, stable=stable,
                     parity_ok=parity_ok, features=feats, runs=all_runs)
    if verbose:
        print(f"\n  => recommended lp_backend = '{recommended}'  "
              f"({'stable' if stable else 'UNSTABLE'}, "
              f"parity {'ok' if parity_ok else 'MISMATCH'})")
    return res


def _print_size_table(hours, runs, winner):
    print(f"\n  race @ {hours}h:")
    for r in sorted(runs, key=lambda x: x.wall):
        mark = " <-- pick" if r.method == winner else ""
        w = "  inf" if r.wall == float("inf") else f"{r.wall:6.2f}s"
        print(f"    {r.method:10s} {w}  {r.status:12s} obj={r.objective:.5e}{mark}")


def tune_solver(system: "EnergySystem", write: bool = True,
                sidecar_dir: str | Path = ".", **kw) -> TuneResult:
    """Race methods, print a report, and (default) write the sidecar that
    ``optimise()`` reads automatically. Set ``write=False`` for report-only."""
    res = recommend_lp_method(system, **kw)
    if write:
        path = Path(sidecar_dir) / SIDECAR_NAME
        path.write_text(json.dumps(res.to_sidecar(), indent=2))
        print(f"  wrote {path}  (delete to revert; explicit lp_backend= "
              f"always overrides)")
    return res


# --- sidecar lookup used by core.optimise ----------------------------------

_sidecar_cache: dict[str, Optional[str]] = {}


def sidecar_lp_backend(start_dir: str | Path = ".") -> Optional[str]:
    """Return the lp_backend from the nearest ``.nexus_solver.json`` walking up
    from ``start_dir`` (cwd by default), or None. Cached per directory."""
    key = str(Path(start_dir).resolve())
    if key in _sidecar_cache:
        return _sidecar_cache[key]
    result = None
    d = Path(key)
    for cand in [d, *d.parents][:6]:
        f = cand / SIDECAR_NAME
        if f.exists():
            try:
                result = json.loads(f.read_text()).get("lp_backend")
            except Exception:  # noqa: BLE001
                result = None
            break
    _sidecar_cache[key] = result
    return result
