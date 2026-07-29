"""
Phase 12 — cloud / parallel execution helpers.

Public surface:

- :func:`run_scenarios_parallel` — embarrassingly-parallel scenario
  runner with auto-selected backend (ray → process-pool → serial).
- :class:`ParallelResult` — wall-clock + per-scenario results +
  parallel-efficiency estimate.
"""

from __future__ import annotations

from nexus_energy.cloud.ray_runner import (
    ParallelResult,
    run_scenarios_parallel,
    ray_available,
)

__all__ = ["ParallelResult", "run_scenarios_parallel", "ray_available"]
