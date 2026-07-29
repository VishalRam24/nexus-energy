"""
Phase 11 — learned representative-period selection.

The existing Phase 7 pipeline (``temporal.aggregate_to_representative_days``)
uses pure k-medoids on the raw per-feature signal. That works well in
the average case, but gets repeatedly wrong on the same calendar days
for the same system. If we have a bank of "historical solves whose
representative-period error we know", we can re-weight the feature
space so k-medoids picks reps that are closer to the hard-to-model
days and avoids re-running the bad ones.

:class:`LearnedClusterSelector` implements this as a supervised
re-weighting on top of the existing k-medoids. The model is
intentionally lightweight: per-feature scaling factors learnt from the
historical ``duration_curve_l1`` errors. It is *not* a GNN or deep
model — training requires only (timeseries, rep-period error) pairs
and a closed-form ridge fit.

:func:`learned_representative_periods` is a drop-in replacement for
``temporal.aggregate_to_representative_days`` that applies the learnt
feature weights before clustering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from nexus_energy.temporal import (
    RepresentativePeriods,
    aggregate_to_representative_days,
    k_medoids,
    ml_feature_embedding,
    representative_period_error,
)

if TYPE_CHECKING:
    from nexus_energy.temporal import AggregationError


@dataclass
class _HistoryEntry:
    feature_names: tuple[str, ...]
    per_feature_error: dict[str, float]  # duration-curve L1 per feature


# ---------------------------------------------------------------------------
# Shared feature-embedding kernel (Phase 11.6)
# ---------------------------------------------------------------------------

# Default statistical feature set, kept in sync with
# ``temporal.ml_feature_embedding`` so the learned selector and the
# feature-embedding aggregator agree on the embedding space.
_DEFAULT_EMB_FEATURES: tuple[str, ...] = (
    "mean", "std", "min", "max", "ramp_max", "peak_hour", "duration_p95",
)


def _embedding_kmedoids_periods(
    timeseries: dict[str, np.ndarray],
    n_days: int,
    hours_per_day: int,
    seed: int,
    *,
    features: tuple[str, ...] = _DEFAULT_EMB_FEATURES,
    column_weights: np.ndarray | None = None,
) -> RepresentativePeriods:
    """One shared kernel: cluster days on an ML feature embedding.

    This is the single code path that both
    :func:`learned_representative_periods` (when running in
    feature-embedding mode) and :func:`feature_embedding_periods` call,
    so the learned selector and the feature-embedding aggregator do not
    duplicate the embed → (weight) → k-medoids → pack logic.

    Steps:
      1. Build the per-day statistical embedding via
         :func:`temporal.ml_feature_embedding` (the *same* embedding the
         Phase 7 ``aggregate_with_feature_embedding`` uses).
      2. Optionally scale embedding columns by ``column_weights`` — this
         is how the learned per-feature weights bias the distance metric
         in embedding space rather than in raw-profile space.
      3. Run :func:`temporal.k_medoids`, then pack the medoids /
         labels / weights into a :class:`RepresentativePeriods` whose
         ``profiles`` carry the *physical* (unscaled) daily values.

    Returns a :class:`RepresentativePeriods` identical in structure to
    what ``temporal.aggregate_with_feature_embedding`` returns, by
    construction (same embedding, same k-medoids), so the two stay
    behaviourally unified.
    """
    names = sorted(timeseries.keys())
    arrays = [np.asarray(timeseries[name], dtype=float) for name in names]
    n_hours = min(len(a) for a in arrays)
    n_days_total = n_hours // hours_per_day
    n_hours = n_days_total * hours_per_day

    stacked = np.column_stack([a[:n_hours] for a in arrays])
    daily_profiles = stacked.reshape(n_days_total, hours_per_day, -1)

    emb = ml_feature_embedding(timeseries, hours_per_day, features)
    if column_weights is not None:
        cw = np.asarray(column_weights, dtype=float)
        if cw.shape[0] != emb.shape[1]:
            raise ValueError(
                f"column_weights length {cw.shape[0]} != embedding width "
                f"{emb.shape[1]}")
        emb = emb * cw[np.newaxis, :]

    k = min(n_days, n_days_total)
    medoids, labels, _ = k_medoids(emb, k, seed=seed)

    weights = np.zeros(k)
    for c in range(k):
        weights[c] = np.sum(labels == c)

    return RepresentativePeriods(
        n_periods=k,
        period_length=hours_per_day,
        medoid_indices=medoids,
        weights=weights,
        mapping=labels,
        profiles=daily_profiles[medoids],
    )


def feature_embedding_periods(
    timeseries: dict[str, np.ndarray],
    n_days: int = 12,
    hours_per_day: int = 24,
    seed: int = 42,
    features: tuple[str, ...] = _DEFAULT_EMB_FEATURES,
) -> RepresentativePeriods:
    """Feature-embedding representative-period selection (unified kernel).

    Thin public entry point over :func:`_embedding_kmedoids_periods` with
    no learned weighting. Behaviourally equivalent to
    ``temporal.aggregate_with_feature_embedding`` — it now shares the
    *same* internal kernel as the learned-selector feature-embedding path
    so the two no longer duplicate logic.
    """
    return _embedding_kmedoids_periods(
        timeseries, n_days=n_days, hours_per_day=hours_per_day,
        seed=seed, features=features,
    )


@dataclass
class LearnedClusterSelector:
    """Weighted-k-medoids representative-period selector.

    Maintains a bank of ``(feature_names, duration_curve_l1 per
    feature)`` pairs from past runs. ``fit`` reduces the bank to a
    per-feature weight vector via a ridge-regularised mean of the
    historical errors. At predict time, :func:`learned_representative_periods`
    scales each feature by its weight before the k-medoids call, so
    features that historically caused the worst aggregation errors
    dominate the clustering metric.

    Why this works: k-medoids minimises total within-cluster distance
    in Euclidean space. If the duration-curve-critical feature (e.g.
    net load) is dwarfed by a noisy feature (e.g. temperature), the
    clustering spends its budget on the wrong axis. Up-weighting the
    critical feature forces the reps to straddle its extremes.
    """

    history: list[_HistoryEntry] = field(default_factory=list)
    ridge: float = 0.1
    # When True, the learned weights bias the *shared feature-embedding*
    # kernel (one code path with :func:`feature_embedding_periods`)
    # instead of scaling the raw profiles before vanilla k-medoids. The
    # embedding path is the unified one; the raw path is kept as the
    # historical default so existing callers don't shift behaviour.
    use_embedding: bool = False
    embedding_features: tuple[str, ...] = _DEFAULT_EMB_FEATURES
    # Populated by ``fit``.
    weights: dict[str, float] = field(default_factory=dict)

    # ---- Training ----

    def observe(
        self,
        timeseries: dict[str, np.ndarray],
        rep: RepresentativePeriods,
    ) -> None:
        """Record one historical (timeseries, rep-period) pair.

        Computes the per-feature duration-curve L1 error of the rep
        and stashes it. The ``fit`` step then turns these errors into
        weights.
        """
        err = representative_period_error(timeseries, rep)
        entry = _HistoryEntry(
            feature_names=tuple(sorted(timeseries.keys())),
            per_feature_error=dict(err.duration_curve_l1),
        )
        self.history.append(entry)

    def fit(self) -> dict[str, float]:
        """Derive per-feature weights from the history bank.

        Weight for feature ``f`` is proportional to its average
        historical duration-curve L1 error plus a ridge regulariser to
        prevent any single feature from being down-weighted to zero::

            w_f = (mean_err_f + ridge) / (sum_over_f (mean_err_f + ridge))

        Returns the weight dict so callers can inspect it.
        """
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for entry in self.history:
            for name, err in entry.per_feature_error.items():
                totals[name] = totals.get(name, 0.0) + float(err)
                counts[name] = counts.get(name, 0) + 1
        weights: dict[str, float] = {}
        for name, tot in totals.items():
            avg = tot / max(counts[name], 1)
            weights[name] = avg + self.ridge
        total = sum(weights.values())
        if total > 0:
            for name in list(weights):
                weights[name] = weights[name] / total * len(weights)
        self.weights = weights
        return weights

    # ---- Prediction ----

    def weight_for(self, name: str) -> float:
        """Return the learnt weight for a feature (1.0 if unseen)."""
        return float(self.weights.get(name, 1.0))

    def embedding_column_weights(
        self,
        timeseries: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Per-embedding-column weights for the shared kernel.

        ``ml_feature_embedding`` lays out columns as ``feature × series``
        in the order ``features`` (outer) by ``sorted(series)`` (inner).
        Each series column inherits its learnt feature weight, repeated
        once per statistical feature, so a high-error series is
        up-weighted across all of its statistics in embedding space.
        """
        names = sorted(timeseries.keys())
        per_series = np.array([self.weight_for(n) for n in names], dtype=float)
        n_feat = len(self.embedding_features)
        return np.tile(per_series, n_feat)


def learned_representative_periods(
    timeseries: dict[str, np.ndarray],
    selector: LearnedClusterSelector | None = None,
    n_days: int = 12,
    hours_per_day: int = 24,
    seed: int = 42,
    extreme_periods: list[tuple[str, str]] | None = None,
) -> RepresentativePeriods:
    """Learned-weight representative-period aggregation.

    Wraps :func:`temporal.aggregate_to_representative_days`. If a
    fitted :class:`LearnedClusterSelector` is supplied, each time-series
    is scaled by the selector's per-feature weight before the
    clustering call — which, being on normalised features, directly
    biases the distance metric. After clustering we restore the
    unscaled values in the returned ``profiles`` array so that any
    subsequent constraint-building sees the physical magnitudes.

    Without a selector (or with an unfitted one), this function is
    equivalent to the vanilla k-medoids baseline and the two call
    paths produce identical results.
    """
    if selector is None or not selector.weights:
        return aggregate_to_representative_days(
            timeseries=timeseries,
            n_days=n_days,
            hours_per_day=hours_per_day,
            seed=seed,
            extreme_periods=extreme_periods,
        )

    # Unified feature-embedding path: bias the SAME kernel that
    # :func:`feature_embedding_periods` uses by the learnt column
    # weights, rather than duplicating embed/cluster logic here.
    if selector.use_embedding:
        cw = selector.embedding_column_weights(timeseries)
        return _embedding_kmedoids_periods(
            timeseries, n_days=n_days, hours_per_day=hours_per_day,
            seed=seed, features=selector.embedding_features,
            column_weights=cw,
        )

    scaled: dict[str, np.ndarray] = {}
    for name, series in timeseries.items():
        w = selector.weight_for(name)
        scaled[name] = np.asarray(series, dtype=float) * float(w)

    rep = aggregate_to_representative_days(
        timeseries=scaled,
        n_days=n_days,
        hours_per_day=hours_per_day,
        seed=seed,
        extreme_periods=extreme_periods,
    )

    # Un-scale the profiles so downstream code sees physical magnitudes.
    names = sorted(timeseries.keys())
    scales = np.array([selector.weight_for(n) for n in names], dtype=float)
    scales = np.where(scales == 0, 1.0, scales)
    unscaled_profiles = rep.profiles / scales[np.newaxis, np.newaxis, :]
    return RepresentativePeriods(
        n_periods=rep.n_periods,
        period_length=rep.period_length,
        medoid_indices=rep.medoid_indices,
        weights=rep.weights,
        mapping=rep.mapping,
        profiles=unscaled_profiles,
    )
