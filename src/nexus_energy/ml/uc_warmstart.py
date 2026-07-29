"""
Phase 11 — unit-commitment warm-start predictors.

Three predictor flavours are provided, all satisfying
:class:`UCWarmstartPredictor`:

- :class:`MeritOrderPredictor` — deterministic, pure-numpy fallback.
  Ranks committable units by marginal cost, sorts them against per-
  timestep net load, and returns a hard 0/1 schedule with a
  confidence that scales with the distance to the nearest on/off
  boundary. Good default when no historical bank exists.
- :class:`HistoricalNeighborPredictor` — k-NN over a bank of past
  (``SystemFeatures``, ``TimestepFeatures``, ``unit_status``)
  records. The system vector filters to "same-regime" neighbours;
  per-timestep UC decisions are borrowed from the nearest
  time-feature match within that shortlist. Confidence = neighbour
  agreement ratio.
- :class:`GNNPredictor` — torch-optional wrapper around a trained
  model. When torch is not installed (or no model is provided), the
  predictor raises :class:`RuntimeError` at call-time; the module
  still imports so downstream code can gate behind
  ``torch_available``.

The confidence score is the lever for the cold-start fallback:
:func:`warm_start_from_prediction` keeps only entries with
``confidence >= threshold`` and emits NaN elsewhere so the optimiser
solves those timesteps fresh.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from nexus_energy.ml.features import (
    SystemFeatures,
    TimestepFeatures,
    extract_system_features,
    extract_timestep_features,
)

if TYPE_CHECKING:
    from nexus_energy.core import EnergySystem


# ---------------------------------------------------------------------------
# Common data classes
# ---------------------------------------------------------------------------

@dataclass
class UCPrediction:
    """Container for a predicted UC schedule.

    Attributes:
        schedule: ``{gen_name: np.ndarray of shape (T,)}`` with u[t] in
            {0, 1} (or integer counts for clustered UC). May contain
            ``np.nan`` for entries the predictor refuses to commit to.
        confidence: ``{gen_name: np.ndarray of shape (T,)}`` with values
            in ``[0, 1]``. Higher = more confident that ``schedule[t]``
            is correct. Used to mask low-confidence cells before the
            MIP sees them.
        method: free-text tag of the predictor that produced the result.
    """
    schedule: dict[str, np.ndarray] = field(default_factory=dict)
    confidence: dict[str, np.ndarray] = field(default_factory=dict)
    method: str = "unknown"


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class UCWarmstartPredictor(ABC):
    """Abstract UC warm-start predictor.

    Implementations consume the two feature vectors and return a
    :class:`UCPrediction`. Predictors are required to be deterministic
    given the same features so that warm-started solves remain
    reproducible.
    """

    name: str = "abstract"

    @abstractmethod
    def predict(
        self,
        system_features: SystemFeatures,
        timestep_features: TimestepFeatures,
        generator_meta: list["_GenMeta"],
    ) -> UCPrediction:
        ...


# ---------------------------------------------------------------------------
# Per-generator metadata snapshot (avoids carrying EnergySystem into predict)
# ---------------------------------------------------------------------------

@dataclass
class _GenMeta:
    """Lightweight, predictor-friendly snapshot of a Generator."""
    name: str
    committable: bool
    capacity: float
    n_units: int
    clustered: bool
    marginal_cost: float
    min_up: int
    min_down: int
    carrier_factor: np.ndarray | None


def _generator_meta(sys: "EnergySystem") -> list[_GenMeta]:
    out: list[_GenMeta] = []
    for g in sys._generators:
        cf = (np.asarray(g.carrier_factor, dtype=float)
              if g.carrier_factor is not None else None)
        out.append(_GenMeta(
            name=g.name,
            committable=bool(g.committable),
            capacity=float(g.capacity),
            n_units=int(getattr(g, "n_units", 1) or 1),
            clustered=bool(getattr(g, "clustered", False)),
            marginal_cost=float(g.marginal_cost),
            min_up=int(getattr(g, "min_up_time", 1) or 1),
            min_down=int(getattr(g, "min_down_time", 1) or 1),
            carrier_factor=cf,
        ))
    return out


# ---------------------------------------------------------------------------
# Merit-order predictor (pure numpy, no training)
# ---------------------------------------------------------------------------

class MeritOrderPredictor(UCWarmstartPredictor):
    """Deterministic merit-order UC predictor.

    Algorithm:

    1. Restrict to committable generators and sort ascending by
       ``marginal_cost`` (economic dispatch order).
    2. For each timestep ``t``, compute ``residual = net_load[t] -
       vre_available[t]``. Walk the sorted merit order accumulating
       available capacity until ``cumcap >= residual``. Each generator
       committed so far gets ``u[t] = 1`` (or ``n_units`` for
       clustered), each one after gets ``u[t] = 0``.
    3. Confidence at ``t`` is the scaled slack between cumulative
       capacity and the residual: far from the on/off boundary ⇒
       confidence → 1; within one unit of capacity ⇒ confidence → 0.
    """

    name = "merit_order"

    def predict(
        self,
        system_features: SystemFeatures,
        timestep_features: TimestepFeatures,
        generator_meta: list[_GenMeta],
    ) -> UCPrediction:
        committable = [g for g in generator_meta if g.committable]
        if not committable:
            return UCPrediction(method=self.name)

        peak = max(system_features.peak_load_mw, 1.0)
        T = timestep_features.load_norm.shape[0]
        net = timestep_features.net_load_norm * peak  # MW scale
        net = np.maximum(net, 0.0)

        # Deterministic merit order; tie-break on name for stability.
        order = sorted(
            range(len(committable)),
            key=lambda i: (committable[i].marginal_cost, committable[i].name),
        )
        sorted_gens = [committable[i] for i in order]

        sched = {g.name: np.zeros(T, dtype=float) for g in sorted_gens}
        conf = {g.name: np.zeros(T, dtype=float) for g in sorted_gens}

        total_cap = sum(
            g.capacity * (g.n_units if g.clustered else 1) for g in sorted_gens)
        total_cap = max(total_cap, 1.0)

        for t in range(T):
            cumcap = 0.0
            residual = float(net[t])
            for g in sorted_gens:
                unit_cap = g.capacity
                if g.clustered:
                    # Fractional integer ramp-in — switch on as many units
                    # as needed, clipped to the fleet size.
                    needed = max(residual - cumcap, 0.0)
                    units_on = int(np.clip(np.ceil(needed / max(unit_cap, 1e-9)),
                                           0, g.n_units))
                    sched[g.name][t] = float(units_on)
                    cumcap += units_on * unit_cap
                    # Confidence: fraction of the fleet that is
                    # "uncontested" by the marginal unit.
                    edge = min(units_on, g.n_units - units_on)
                    conf[g.name][t] = 1.0 - (edge / max(g.n_units, 1))
                else:
                    if cumcap + unit_cap <= residual + 1e-9:
                        sched[g.name][t] = 1.0
                        cumcap += unit_cap
                        slack = max(residual - cumcap, 0.0)
                    elif cumcap >= residual - 1e-9:
                        sched[g.name][t] = 0.0
                        slack = max(cumcap - residual, 0.0)
                    else:
                        # Marginal unit — close call.
                        sched[g.name][t] = 1.0
                        cumcap += unit_cap
                        slack = abs(cumcap - unit_cap / 2 - residual)
                    conf[g.name][t] = float(
                        np.clip(slack / max(unit_cap, 1.0), 0.0, 1.0))

        return UCPrediction(schedule=sched, confidence=conf, method=self.name)


# ---------------------------------------------------------------------------
# Historical-neighbour predictor (k-NN over solve bank)
# ---------------------------------------------------------------------------

@dataclass
class HistoricalRecord:
    """One entry in the historical solve bank."""
    system_vec: np.ndarray
    timestep_mat: np.ndarray
    unit_status: dict[str, np.ndarray]
    tag: str = ""


class HistoricalNeighborPredictor(UCWarmstartPredictor):
    """k-NN warm-start predictor over a bank of past solves.

    Two-stage lookup:

    1. System-level shortlist: find the ``k_sys`` records with the
       smallest cosine distance to the current ``SystemFeatures``. This
       filters for systems in the same "regime" (topology / tech mix).
    2. Per-timestep match: within the shortlist, for each timestep ``t``
       find the ``k_step`` nearest rows in feature space and majority-
       vote their ``unit_status[t]``.

    Confidence at ``t`` is the fraction of the ``k_step`` neighbours
    that agreed with the majority. An empty bank yields all-NaN
    schedules with zero confidence; :func:`warm_start_from_prediction`
    will then trigger the cold-start fallback.
    """

    name = "historical_neighbour"

    def __init__(self, k_sys: int = 5, k_step: int = 5) -> None:
        self.k_sys = int(k_sys)
        self.k_step = int(k_step)
        self._bank: list[HistoricalRecord] = []

    # ---- Bank management ----

    def record(
        self,
        sys: "EnergySystem",
        result: "object",
        tag: str = "",
    ) -> None:
        """Append one solved system + UC schedule to the bank."""
        sfs = extract_system_features(sys)
        tfs = extract_timestep_features(sys)
        status = getattr(result, "unit_status", {}) or {}
        if not status:
            return
        self._bank.append(HistoricalRecord(
            system_vec=sfs.to_vector(),
            timestep_mat=tfs.to_matrix(),
            unit_status={k: np.asarray(v, dtype=float) for k, v in status.items()},
            tag=tag,
        ))

    def clear(self) -> None:
        self._bank.clear()

    @property
    def bank_size(self) -> int:
        return len(self._bank)

    # ---- Prediction ----

    def predict(
        self,
        system_features: SystemFeatures,
        timestep_features: TimestepFeatures,
        generator_meta: list[_GenMeta],
    ) -> UCPrediction:
        committable = [g for g in generator_meta if g.committable]
        T = timestep_features.load_norm.shape[0]
        # Empty bank → all NaN; let warm-start fall back to cold.
        if not self._bank or not committable:
            sched = {g.name: np.full(T, np.nan) for g in committable}
            conf = {g.name: np.zeros(T) for g in committable}
            return UCPrediction(schedule=sched, confidence=conf, method=self.name)

        target_sys = system_features.to_vector()
        shortlist = _nearest_records(target_sys, self._bank, self.k_sys)
        target_mat = timestep_features.to_matrix()

        sched: dict[str, np.ndarray] = {}
        conf: dict[str, np.ndarray] = {}
        for g in committable:
            sched[g.name] = np.full(T, np.nan)
            conf[g.name] = np.zeros(T)

        for t in range(T):
            query = target_mat[t]
            votes = _nearest_timesteps(query, shortlist, self.k_step)
            if not votes:
                continue
            for g in committable:
                vals = [rec.unit_status[g.name][ti]
                        for rec, ti in votes
                        if g.name in rec.unit_status
                        and ti < rec.unit_status[g.name].shape[0]]
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=float)
                if g.clustered:
                    # Nearest-integer of the mean vote.
                    predicted = float(np.round(np.mean(arr)))
                    agreement = float(np.mean(np.abs(arr - predicted) < 0.5))
                else:
                    # Majority vote; ties resolve to 0 (safer — more flex).
                    predicted = 1.0 if np.mean(arr) > 0.5 else 0.0
                    agreement = float(np.mean(arr == predicted))
                sched[g.name][t] = predicted
                conf[g.name][t] = agreement

        return UCPrediction(schedule=sched, confidence=conf, method=self.name)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    # Pad the shorter vector so k-NN works across heterogeneous banks
    # (systems with different tech signatures yield different lengths).
    if a.shape != b.shape:
        n = max(a.shape[0], b.shape[0])
        aa = np.zeros(n); aa[:a.shape[0]] = a
        bb = np.zeros(n); bb[:b.shape[0]] = b
        a, b = aa, bb
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
    return 1.0 - float(np.dot(a, b) / (na * nb))


def _nearest_records(
    target: np.ndarray,
    bank: list[HistoricalRecord],
    k: int,
) -> list[HistoricalRecord]:
    scored = [(_cosine_distance(target, rec.system_vec), rec) for rec in bank]
    scored.sort(key=lambda x: x[0])
    return [rec for _, rec in scored[:max(k, 1)]]


def _nearest_timesteps(
    query: np.ndarray,
    shortlist: list[HistoricalRecord],
    k: int,
) -> list[tuple[HistoricalRecord, int]]:
    candidates: list[tuple[float, HistoricalRecord, int]] = []
    for rec in shortlist:
        mat = rec.timestep_mat
        if mat.size == 0:
            continue
        # Pad/trim query to the record's column count.
        q = query
        if q.shape[0] != mat.shape[1]:
            n = min(q.shape[0], mat.shape[1])
            q = q[:n]
            mat_local = mat[:, :n]
        else:
            mat_local = mat
        diffs = mat_local - q
        dists = np.linalg.norm(diffs, axis=1)
        for ti, d in enumerate(dists):
            candidates.append((float(d), rec, ti))
    candidates.sort(key=lambda x: x[0])
    return [(rec, ti) for _, rec, ti in candidates[:max(k, 1)]]


# ---------------------------------------------------------------------------
# GNN predictor (torch-optional)
# ---------------------------------------------------------------------------

try:  # pragma: no cover — probe at import time.
    import torch  # type: ignore
    torch_available = True
except Exception:  # pragma: no cover — torch not installed.
    torch = None  # type: ignore
    torch_available = False


class GNNPredictor(UCWarmstartPredictor):
    """Torch-optional wrapper for a trained GNN UC predictor.

    The class imports cleanly without torch so the rest of
    ``nexus_energy.ml`` stays torch-free. ``predict`` raises
    :class:`RuntimeError` if torch is missing or no ``model`` callable
    was passed, which lets callers fall back to one of the pure-numpy
    predictors instead of crashing the solve pipeline.
    """

    name = "gnn"

    def __init__(
        self,
        model: Callable[[SystemFeatures, TimestepFeatures], UCPrediction]
        | None = None,
    ) -> None:
        self.model = model

    def predict(
        self,
        system_features: SystemFeatures,
        timestep_features: TimestepFeatures,
        generator_meta: list[_GenMeta],
    ) -> UCPrediction:
        if self.model is None:
            raise RuntimeError(
                "GNNPredictor requires a trained model callable. "
                "Pass one via GNNPredictor(model=...). Without torch "
                "installed, prefer MeritOrderPredictor or "
                "HistoricalNeighborPredictor instead."
            )
        if not torch_available:
            raise RuntimeError(
                "GNNPredictor requires PyTorch at runtime. "
                "Install nexus-energy[ml] or provide a torch-free model."
            )
        return self.model(system_features, timestep_features)


# ---------------------------------------------------------------------------
# Driver functions
# ---------------------------------------------------------------------------

def predict_unit_commitment(
    sys: "EnergySystem",
    predictor: UCWarmstartPredictor | None = None,
) -> UCPrediction:
    """Run a predictor against the current system state.

    Defaults to :class:`MeritOrderPredictor` when no predictor is given.
    """
    if predictor is None:
        predictor = MeritOrderPredictor()
    sfs = extract_system_features(sys)
    tfs = extract_timestep_features(sys)
    meta = _generator_meta(sys)
    return predictor.predict(sfs, tfs, meta)


def warm_start_from_prediction(
    prediction: UCPrediction,
    confidence_threshold: float = 0.7,
    cold_start_fallback: bool = True,
    max_fix_fraction: float | None = None,
) -> dict[str, np.ndarray]:
    """Convert a :class:`UCPrediction` into an ``uc_fix_schedule`` dict.

    High-confidence entries (``confidence >= threshold``) are passed
    through; the rest become ``np.nan`` so the MIP solves them fresh.
    When ``cold_start_fallback=True`` and every confidence vector is
    zero (empty-bank case), an empty dict is returned — i.e. no
    fixings at all, which is equivalent to a cold start.

    ``max_fix_fraction`` (Phase 11.x) caps the total pinned cells to
    ``max_fix_fraction × total_predicted_cells`` across all generators,
    ranked by confidence. Use this to protect against over-fixing when
    the predictor reports uniformly high confidence (e.g. k-NN with a
    shallow bank). ``None`` disables the cap; a value in (0, 1] caps
    the global fraction.
    """
    if max_fix_fraction is not None:
        if not 0.0 < max_fix_fraction <= 1.0:
            raise ValueError("max_fix_fraction must be in (0, 1]")

    # First pass: gather all (confidence, name, t, value) candidates that
    # pass the threshold.
    candidates: list[tuple[float, str, int, float]] = []
    total_cells = 0
    for name, sched in prediction.schedule.items():
        conf = prediction.confidence.get(name, np.zeros_like(sched))
        total_cells += int(sched.shape[0])
        mask = (~np.isnan(sched)) & (conf >= confidence_threshold)
        if not np.any(mask):
            continue
        for t in np.where(mask)[0]:
            candidates.append((float(conf[t]), name, int(t), float(sched[t])))

    if not candidates and cold_start_fallback:
        return {}

    # Apply the global cap by sorting on confidence (desc) and truncating.
    if max_fix_fraction is not None and total_cells > 0:
        cap = int(max_fix_fraction * total_cells)
        if len(candidates) > cap:
            candidates.sort(key=lambda x: -x[0])
            candidates = candidates[:cap]
            if not candidates and cold_start_fallback:
                return {}

    out: dict[str, np.ndarray] = {}
    for _, name, t, val in candidates:
        if name not in out:
            sched = prediction.schedule[name]
            out[name] = np.full(sched.shape, np.nan)
        out[name][t] = val
    return out


# ---------------------------------------------------------------------------
# Warm-start retry driver (Phase 11.x)
# ---------------------------------------------------------------------------

@dataclass
class WarmStartOutcome:
    """Metadata attached to a :func:`solve_with_warm_retry` solve.

    Attributes:
        result: the :class:`OptimisationResult` returned by
            :meth:`EnergySystem.optimise`. ``None`` only in the (rare)
            case that the final retry also failed and
            ``raise_on_failure=False`` — in that case ``status`` is
            ``"infeasible"``.
        status: ``"warm"`` if the first warm attempt succeeded,
            ``"warm_retry"`` if a loosened retry succeeded,
            ``"cold"`` if the predictor was skipped or we fell back to
            cold after exhausting retries, ``"infeasible"`` if even the
            cold solve failed.
        retries: count of warm attempts made (0 if we went straight to
            cold).
        n_pinned: number of cells pinned in the final accepted solve (0
            for cold).
        attempts: list of (fix_fraction, result_status) tuples
            describing each warm attempt in order. Useful for reporting
            / regression diagnostics.
    """
    result: object | None
    status: str
    retries: int = 0
    n_pinned: int = 0
    attempts: list[tuple[float, str]] = field(default_factory=list)


def solve_with_warm_retry(
    sys: "EnergySystem",
    predictor: UCWarmstartPredictor | None = None,
    *,
    confidence_threshold: float = 0.7,
    max_fix_fraction: float = 0.75,
    max_retries: int = 2,
    shrink_factor: float = 0.5,
    cold_fallback: bool = True,
    raise_on_failure: bool = True,
    **optimise_kwargs,
) -> WarmStartOutcome:
    """Solve ``sys`` with a warm-start schedule, retrying on infeasibility.

    Loop:
      1. Predict a UC schedule, convert to an ``uc_fix_schedule`` with
         the current ``max_fix_fraction`` cap.
      2. Call ``sys.optimise(uc_fix_schedule=..., **optimise_kwargs)``.
      3. If the result is optimal: return it.
      4. Else if retries remain: shrink the cap by ``shrink_factor`` and
         retry from step 1.
      5. Else if ``cold_fallback``: solve cold and return it.
      6. Else (``raise_on_failure=True``): raise ``RuntimeError``;
         (``False``): return the last (infeasible) result.

    The shrink path is why this is a *retry* driver and not a one-shot
    helper: the first over-fixed attempt is often infeasible because the
    predictor pinned u-cells that conflict with min-up / min-down
    constraints. Halving the cap on each retry restores feasibility
    while keeping the easy pins intact.
    """
    if not 0.0 < max_fix_fraction <= 1.0:
        raise ValueError("max_fix_fraction must be in (0, 1]")
    if not 0.0 < shrink_factor < 1.0:
        raise ValueError("shrink_factor must be in (0, 1)")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    if predictor is None:
        predictor = MeritOrderPredictor()

    prediction = predict_unit_commitment(sys, predictor)
    attempts: list[tuple[float, str]] = []
    frac = float(max_fix_fraction)

    for attempt_i in range(max_retries + 1):
        fix = warm_start_from_prediction(
            prediction,
            confidence_threshold=confidence_threshold,
            cold_start_fallback=True,
            max_fix_fraction=frac,
        )
        if not fix:
            # No confident fixings — go straight to cold.
            break
        n_pinned = int(sum(np.count_nonzero(~np.isnan(v)) for v in fix.values()))
        result = sys.optimise(uc_fix_schedule=fix, **optimise_kwargs)
        status_str = getattr(result, "status", "unknown")
        attempts.append((frac, status_str))
        cost = getattr(result, "total_cost", None)
        if status_str == "optimal" and cost is not None and np.isfinite(cost):
            return WarmStartOutcome(
                result=result,
                status="warm" if attempt_i == 0 else "warm_retry",
                retries=attempt_i,
                n_pinned=n_pinned,
                attempts=attempts,
            )
        # Shrink for the next round.
        frac *= shrink_factor
        if frac <= 0.0:
            break

    if cold_fallback:
        result = sys.optimise(**optimise_kwargs)
        status_str = getattr(result, "status", "unknown")
        cost = getattr(result, "total_cost", None)
        if status_str == "optimal" and cost is not None and np.isfinite(cost):
            return WarmStartOutcome(
                result=result, status="cold",
                retries=len(attempts), n_pinned=0, attempts=attempts,
            )
        # Even cold failed.
        if raise_on_failure:
            raise RuntimeError(
                f"solve_with_warm_retry: cold fallback returned "
                f"status={status_str!r}; warm attempts={attempts}"
            )
        return WarmStartOutcome(
            result=result, status="infeasible",
            retries=len(attempts), n_pinned=0, attempts=attempts,
        )

    if raise_on_failure:
        raise RuntimeError(
            f"solve_with_warm_retry: all {len(attempts)} warm attempts "
            f"failed and cold_fallback=False; attempts={attempts}"
        )
    return WarmStartOutcome(
        result=None, status="infeasible",
        retries=len(attempts), n_pinned=0, attempts=attempts,
    )


# ---------------------------------------------------------------------------
# Adaptive confidence threshold (Phase 11.6)
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveThresholdController:
    """Stateful per-window confidence-threshold controller.

    The plain :func:`warm_start_from_prediction` uses one fixed
    ``confidence_threshold`` for the whole horizon. In a rolling-window
    MPC loop that is brittle: a single high-drift or infeasible window
    will over-pin (too low a threshold ⇒ many low-confidence cells
    locked ⇒ infeasibility against min-up / min-down), while a long run
    of clean windows leaves accuracy on the table (the threshold could
    be lowered to pin more cells and shrink the MIP).

    This controller adapts the threshold *per rolling window* based on
    the feasibility / drift signal of the **previous** window, in the
    spirit of an additive-increase / multiplicative-decrease (AIMD)
    congestion controller:

    - After an **infeasible** window (or one whose observed schedule
      drifted far from the prediction) the threshold is *raised* —
      multiplicatively — so the next window pins fewer, more-confident
      cells.
    - After a **clean** window (feasible, low drift) the threshold is
      *lowered* — additively — so the next window pins a few more cells.

    The threshold is always clamped to ``[min_threshold, max_threshold]``.
    The controller is intentionally pure-numpy and tiny; it holds only a
    scalar threshold plus a short history for diagnostics.

    Attributes:
        threshold: current confidence threshold (mutated in place).
        min_threshold / max_threshold: clamp bounds.
        increase_factor: multiplicative bump applied on a bad window
            (``threshold *= increase_factor``); must be > 1.
        decrease_step: additive relaxation applied on a clean window
            (``threshold -= decrease_step``); must be > 0.
        drift_tol: per-cell mismatch fraction above which a *feasible*
            window is still treated as "bad" (high drift). Drift is the
            fraction of pinned cells whose realised status differed from
            the predicted status.
        history: list of ``(threshold_used, signal)`` where ``signal`` is
            one of ``"clean"`` / ``"drift"`` / ``"infeasible"``.
    """

    threshold: float = 0.7
    min_threshold: float = 0.5
    max_threshold: float = 0.99
    increase_factor: float = 1.15
    decrease_step: float = 0.05
    drift_tol: float = 0.25
    history: list[tuple[float, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 < self.min_threshold <= self.max_threshold <= 1.0:
            raise ValueError(
                "require 0 < min_threshold <= max_threshold <= 1")
        if self.increase_factor <= 1.0:
            raise ValueError("increase_factor must be > 1")
        if self.decrease_step <= 0.0:
            raise ValueError("decrease_step must be > 0")
        self.threshold = float(
            np.clip(self.threshold, self.min_threshold, self.max_threshold))

    def current(self) -> float:
        """Return the threshold to use for the *next* window."""
        return float(self.threshold)

    @staticmethod
    def drift_fraction(
        fix: dict[str, np.ndarray],
        realised_status: dict[str, np.ndarray] | None,
    ) -> float:
        """Fraction of pinned cells whose realised status disagreed.

        ``fix`` is the ``uc_fix_schedule`` that was pinned (NaN = free);
        ``realised_status`` is the solved ``unit_status`` dict. Returns
        0.0 when nothing was pinned or no realised status is available.
        """
        if not fix or not realised_status:
            return 0.0
        n_pinned = 0
        n_diff = 0
        for name, pinned in fix.items():
            real = realised_status.get(name)
            if real is None:
                continue
            real = np.asarray(real, dtype=float)
            m = ~np.isnan(pinned)
            k = int(np.count_nonzero(m))
            if k == 0:
                continue
            T = min(pinned.shape[0], real.shape[0])
            mm = m[:T]
            n_pinned += int(np.count_nonzero(mm))
            n_diff += int(np.count_nonzero(
                np.abs(pinned[:T][mm] - real[:T][mm]) > 0.5))
        if n_pinned == 0:
            return 0.0
        return n_diff / n_pinned

    def update(
        self,
        *,
        feasible: bool,
        fix: dict[str, np.ndarray] | None = None,
        realised_status: dict[str, np.ndarray] | None = None,
    ) -> float:
        """Fold one window's outcome into the threshold and return the new value.

        Raises (multiplicatively) on infeasible or high-drift windows;
        relaxes (additively) on clean ones. ``fix`` + ``realised_status``
        are used only to compute the drift signal; pass them when the
        solve succeeded.
        """
        used = float(self.threshold)
        if not feasible:
            signal = "infeasible"
            self.threshold = min(
                self.max_threshold, self.threshold * self.increase_factor)
        else:
            drift = self.drift_fraction(fix or {}, realised_status)
            if drift > self.drift_tol:
                signal = "drift"
                self.threshold = min(
                    self.max_threshold, self.threshold * self.increase_factor)
            else:
                signal = "clean"
                self.threshold = max(
                    self.min_threshold, self.threshold - self.decrease_step)
        self.history.append((used, signal))
        return float(self.threshold)


def solve_with_adaptive_warmstart(
    systems: "list[EnergySystem] | Callable[[int], EnergySystem]",
    n_windows: int | None = None,
    predictor: UCWarmstartPredictor | None = None,
    *,
    controller: AdaptiveThresholdController | None = None,
    max_fix_fraction: float = 0.75,
    max_retries: int = 1,
    shrink_factor: float = 0.5,
    cold_fallback: bool = True,
    raise_on_failure: bool = False,
    **optimise_kwargs,
) -> list[WarmStartOutcome]:
    """Rolling-window warm-start solve with an adaptive confidence threshold.

    Wraps :func:`solve_with_warm_retry` once per window but feeds it the
    threshold produced by an :class:`AdaptiveThresholdController`, then
    folds each window's outcome back into the controller so the *next*
    window adapts. This is the per-window adaptive path on top of the
    fixed-threshold retry driver.

    ``systems`` may be a concrete list of per-window
    :class:`EnergySystem` objects, or a factory ``fn(window_index) ->
    EnergySystem`` (with ``n_windows`` then required). The predictor and
    all retry knobs are shared across windows; only the threshold floats.

    Returns one :class:`WarmStartOutcome` per window. The controller's
    ``history`` records the (threshold_used, signal) trail.
    """
    if controller is None:
        controller = AdaptiveThresholdController()
    if predictor is None:
        predictor = MeritOrderPredictor()

    if callable(systems):
        if n_windows is None:
            raise ValueError("n_windows is required when systems is a factory")
        get_system = systems  # type: ignore[assignment]
        count = int(n_windows)
    else:
        seq = list(systems)
        count = len(seq) if n_windows is None else min(int(n_windows), len(seq))
        get_system = lambda i: seq[i]  # noqa: E731

    outcomes: list[WarmStartOutcome] = []
    for w in range(count):
        sys = get_system(w)
        thr = controller.current()
        # Re-derive the pinned schedule at this threshold for drift scoring.
        prediction = predict_unit_commitment(sys, predictor)
        fix = warm_start_from_prediction(
            prediction,
            confidence_threshold=thr,
            cold_start_fallback=True,
            max_fix_fraction=max_fix_fraction,
        )
        outcome = solve_with_warm_retry(
            sys,
            predictor=predictor,
            confidence_threshold=thr,
            max_fix_fraction=max_fix_fraction,
            max_retries=max_retries,
            shrink_factor=shrink_factor,
            cold_fallback=cold_fallback,
            raise_on_failure=raise_on_failure,
            **optimise_kwargs,
        )
        feasible = outcome.status in ("warm", "warm_retry", "cold")
        realised = getattr(outcome.result, "unit_status", None) if outcome.result else None
        controller.update(feasible=feasible, fix=fix, realised_status=realised)
        outcomes.append(outcome)
    return outcomes
