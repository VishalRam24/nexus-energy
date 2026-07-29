"""
Phase 11 — learned variable fixing.

The :class:`LearnedVarFixer` watches a stream of historical solves and
maintains per-(generator, timestep) activation statistics. At predict
time it emits an ``uc_fix_schedule`` that hard-fixes the committable
status variables whose history is overwhelmingly on or off, and leaves
the "swing" variables free. This is the MIP analogue of the "learn
which binaries to fix" literature (Xavier et al. 2021, Bertsimas &
Stellato 2022), specialised to the UC problem.

Why hard-fix instead of warm-hint? Fixing collapses 2^N binaries
wholesale — the dominant cost in clustered UC. Hints only seed the
search. We mitigate the optimality risk by fixing only variables with
``activation_rate ∈ [0, 1-threshold] ∪ [threshold, 1]`` — i.e. the
statistics say "this unit has never turned on at this hour" or "this
unit has always been on at this hour" across the bank.

The module is deliberately solver-agnostic: :func:`apply_varfix` just
returns the fix dict and the caller decides whether to pass it to
``EnergySystem.optimise(uc_fix_schedule=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nexus_energy.core import EnergySystem, OptimisationResult


@dataclass
class VarFixingStats:
    """Running activation statistics for committable generators.

    ``on_count[name]`` is a per-timestep counter of how many solves had
    u[t] > 0.5 (or u[t] / n_units > 0.5 for clustered). ``n_samples``
    counts the number of solves that included that generator. The
    activation rate ``on_count / n_samples`` at each timestep is the
    signal used to decide whether to fix.
    """
    on_count: dict[str, np.ndarray] = field(default_factory=dict)
    n_samples: dict[str, int] = field(default_factory=dict)
    _T: int = 0

    def observe(self, result: "OptimisationResult", sys: "EnergySystem") -> None:
        """Fold one solved result into the running stats."""
        status = getattr(result, "unit_status", {}) or {}
        if not status:
            return
        for gen in sys._generators:
            if not gen.committable:
                continue
            if gen.name not in status:
                continue
            u = np.asarray(status[gen.name], dtype=float)
            # Normalise clustered UC to a 0..1 activation fraction.
            if getattr(gen, "clustered", False):
                denom = max(int(getattr(gen, "n_units", 1) or 1), 1)
                active = (u / denom) > 0.5
            else:
                active = u > 0.5
            T = active.shape[0]
            self._T = max(self._T, T)
            prev = self.on_count.get(gen.name)
            if prev is None or prev.shape[0] != T:
                prev = np.zeros(T, dtype=int)
            prev = prev + active.astype(int)
            self.on_count[gen.name] = prev
            self.n_samples[gen.name] = self.n_samples.get(gen.name, 0) + 1

    def activation_rate(self, gen_name: str) -> np.ndarray | None:
        """Return per-timestep activation rate in [0, 1], or None."""
        n = self.n_samples.get(gen_name, 0)
        if n == 0:
            return None
        return self.on_count[gen_name] / float(n)

    @property
    def total_samples(self) -> int:
        return max(self.n_samples.values(), default=0)


class LearnedVarFixer:
    """Driver that combines :class:`VarFixingStats` with a threshold rule.

    Typical lifecycle::

        fixer = LearnedVarFixer(threshold=0.95, min_samples=5)
        for day in training_days:
            res = sys.optimise()
            fixer.observe(res, sys)
        schedule = fixer.predict(sys)       # uc_fix_schedule-compatible dict
        res = sys.optimise(uc_fix_schedule=schedule)
    """

    def __init__(
        self,
        threshold: float = 0.95,
        min_samples: int = 5,
        max_fix_fraction: float = 0.9,
    ) -> None:
        if not 0.5 < threshold < 1.0:
            raise ValueError("threshold must be in (0.5, 1.0)")
        if not 0.0 < max_fix_fraction <= 1.0:
            raise ValueError("max_fix_fraction must be in (0, 1]")
        self.threshold = float(threshold)
        self.min_samples = int(min_samples)
        self.max_fix_fraction = float(max_fix_fraction)
        self.stats = VarFixingStats()

    # ---- Training ----

    def observe(
        self,
        result: "OptimisationResult",
        sys: "EnergySystem",
    ) -> None:
        self.stats.observe(result, sys)

    # ---- Prediction ----

    def predict(
        self,
        sys: "EnergySystem",
    ) -> dict[str, np.ndarray]:
        """Emit an ``uc_fix_schedule`` dict.

        Entries are fixed to 0 where activation rate ≤ 1-threshold,
        fixed to 1 where activation rate ≥ threshold, and left as
        ``np.nan`` otherwise. A hard cap of ``max_fix_fraction * T``
        cells per generator protects against over-fixing on highly
        unbalanced generators (e.g. a baseload unit that is always on).
        """
        schedule: dict[str, np.ndarray] = {}
        for gen in sys._generators:
            if not gen.committable:
                continue
            n = self.stats.n_samples.get(gen.name, 0)
            if n < self.min_samples:
                continue
            rate = self.stats.activation_rate(gen.name)
            if rate is None:
                continue
            # Cap the on / off bounds separately — keeping a cell free
            # is always safe, so we only filter *fixings*.
            on_mask = rate >= self.threshold
            off_mask = rate <= (1.0 - self.threshold)
            fixings = on_mask.sum() + off_mask.sum()
            if fixings == 0:
                continue
            # Enforce max_fix_fraction: if we'd fix too many cells,
            # keep only the most-confident ones.
            T = rate.shape[0]
            cap = int(self.max_fix_fraction * T)
            if fixings > cap:
                # Rank by |rate - 0.5| (distance from 50/50), keep top cap.
                idx_sorted = np.argsort(-np.abs(rate - 0.5))
                keep = np.zeros(T, dtype=bool)
                keep[idx_sorted[:cap]] = True
                on_mask &= keep
                off_mask &= keep
            if not (on_mask.any() or off_mask.any()):
                continue
            fixed = np.full(T, np.nan)
            if getattr(gen, "clustered", False):
                on_val = float(int(getattr(gen, "n_units", 1) or 1))
            else:
                on_val = 1.0
            fixed[on_mask] = on_val
            fixed[off_mask] = 0.0
            schedule[gen.name] = fixed
        return schedule


def apply_varfix(
    sys: "EnergySystem",
    stats: VarFixingStats | LearnedVarFixer,
    threshold: float = 0.95,
    min_samples: int = 5,
    max_fix_fraction: float = 0.9,
) -> dict[str, np.ndarray]:
    """Build an ``uc_fix_schedule`` from stats without retaining the fixer.

    Accepts either a raw :class:`VarFixingStats` bank or an existing
    :class:`LearnedVarFixer`. The stand-alone form lets callers build a
    one-shot fix without owning the training lifecycle.
    """
    if isinstance(stats, LearnedVarFixer):
        return stats.predict(sys)
    fixer = LearnedVarFixer(
        threshold=threshold,
        min_samples=min_samples,
        max_fix_fraction=max_fix_fraction,
    )
    fixer.stats = stats
    return fixer.predict(sys)
