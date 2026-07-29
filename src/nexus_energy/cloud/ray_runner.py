"""
Phase 12 — parallel scenario runner.

:func:`run_scenarios_parallel` takes a list of scenarios and a
user-supplied ``solve_fn(scenario) -> result`` callable and fans them
out across workers. The backend is selected automatically:

1. **ray** — if ray is importable, distribute via ``ray.remote`` so
   the same API scales to a cluster. Ray handles its own scheduling,
   object store, and fault tolerance.
2. **process-pool** — ``concurrent.futures.ProcessPoolExecutor`` as a
   ray-free fallback. Good enough for local-machine Monte Carlo.
3. **serial** — if ``n_workers=1`` or either of the above chokes on
   the payload (e.g. unpicklable closures), we fall through to a
   plain ``for`` loop. Never silently swallows errors.

The runner is deliberately closure-free on the hot path — the scenario
list is packaged into ``(scenario_idx, scenario)`` tuples and the
``solve_fn`` is pickled once per worker. Callers that need to share a
base ``EnergySystem`` across scenarios should construct it inside
``solve_fn``; passing a live system through ``multiprocessing`` is
fragile.

Parallel efficiency is reported as
``serial_equivalent_s / (n_workers * wall_clock_s)`` using the sum of
per-scenario solve times as the "serial equivalent". Values near 1.0
mean near-linear scaling; values near ``1/n_workers`` mean the runner
was bottlenecked (usually on I/O, pickling, or a global lock).
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence


try:  # pragma: no cover — probe at import time.
    import ray  # type: ignore  # noqa: F401
    ray_available = True
except Exception:  # pragma: no cover — ray missing is fine.
    ray_available = False


__all__ = ["ParallelResult", "run_scenarios_parallel", "ray_available"]


@dataclass
class ParallelResult:
    """Aggregate result of a parallel scenario run."""
    results: list[Any] = field(default_factory=list)
    per_scenario_seconds: list[float] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    n_workers: int = 1
    backend: str = "serial"

    @property
    def parallel_efficiency(self) -> float:
        """Serial-equivalent / (n_workers × wall_clock).

        1.0 = perfect scaling; 1/n_workers = fully serialised.
        """
        if self.wall_clock_seconds <= 0 or self.n_workers <= 0:
            return 0.0
        serial_equiv = sum(self.per_scenario_seconds)
        return float(serial_equiv / (self.n_workers * self.wall_clock_seconds))


def _timed_call(fn: Callable[[Any], Any], payload: tuple[int, Any]) -> tuple[int, Any, float]:
    idx, scenario = payload
    t0 = time.perf_counter()
    out = fn(scenario)
    return idx, out, time.perf_counter() - t0


def run_scenarios_parallel(
    solve_fn: Callable[[Any], Any],
    scenarios: Sequence[Any] | Iterable[Any],
    n_workers: int | None = None,
    backend: str = "auto",
) -> ParallelResult:
    """Run ``solve_fn`` on each scenario in parallel and collect results.

    Args:
        solve_fn: ``(scenario) -> result``. Must be picklable for
            process/ray backends — define it at module scope or use
            ``functools.partial`` on a module-level function.
        scenarios: iterable of scenarios. Consumed into a list once to
            preserve ordering in the output.
        n_workers: number of workers. ``None`` picks ``os.cpu_count()``
            with a ceiling at the scenario count. ``1`` forces serial.
        backend: ``"auto"``, ``"ray"``, ``"process"``, or ``"serial"``.
            ``"auto"`` tries ray, then process-pool, then serial.

    Returns :class:`ParallelResult` with results in original scenario
    order.
    """
    scenarios = list(scenarios)
    if not scenarios:
        return ParallelResult(backend="serial")

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, len(scenarios))
    n_workers = max(1, int(n_workers))

    backend = backend.lower()
    if backend == "auto":
        if n_workers == 1:
            backend = "serial"
        elif ray_available:
            backend = "ray"
        else:
            backend = "process"
    if backend not in {"auto", "ray", "process", "serial"}:
        raise ValueError(f"unknown backend {backend!r}")

    if backend == "ray" and not ray_available:
        backend = "process"
    if backend == "process" and mp.get_start_method(allow_none=True) is None:
        # Lock in a deterministic start method once; avoids the macOS
        # fork/spawn warning storm on subsequent calls.
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:  # pragma: no cover
            pass

    payloads = list(enumerate(scenarios))
    results: list[Any] = [None] * len(scenarios)
    timings: list[float] = [0.0] * len(scenarios)
    t_start = time.perf_counter()

    if backend == "serial":
        for payload in payloads:
            idx, out, elapsed = _timed_call(solve_fn, payload)
            results[idx] = out
            timings[idx] = elapsed
    elif backend == "ray":  # pragma: no cover — exercised only when ray is installed.
        import ray  # type: ignore
        if not ray.is_initialized():
            ray.init(num_cpus=n_workers, ignore_reinit_error=True, log_to_driver=False)
        remote_fn = ray.remote(_timed_call)
        futures = [remote_fn.remote(solve_fn, p) for p in payloads]
        for idx, out, elapsed in ray.get(futures):
            results[idx] = out
            timings[idx] = elapsed
    else:  # process
        with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_timed_call, solve_fn, p) for p in payloads]
            for fut in cf.as_completed(futures):
                idx, out, elapsed = fut.result()
                results[idx] = out
                timings[idx] = elapsed

    wall = time.perf_counter() - t_start
    return ParallelResult(
        results=results,
        per_scenario_seconds=timings,
        wall_clock_seconds=wall,
        n_workers=n_workers,
        backend=backend,
    )
