"""
Phase 5: Advanced Temporal Methods.

Provides time-series aggregation, representative periods, and
rolling horizon optimisation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class RepresentativePeriods:
    """Result of time-series aggregation into representative periods.

    Two mapping representations coexist:

    * ``mapping`` (always populated) — integer ``(n_original_periods,)``
      array, one rep per original period. K-medoids / extreme-period
      aggregation produces this directly; Tulipa-style fractional
      aggregation flattens its matrix by argmax so this field always
      carries a usable one-rep-per-day assignment.
    * ``mapping_matrix`` (optional, Phase 16.4) — ``(n_original_periods,
      n_periods)`` matrix where row d is a probability distribution over
      rep periods (rows sum to 1, entries in [0, 1]). A one-hot matrix is
      *degenerate*: it reproduces ``mapping`` exactly, so it's safe to
      apply even when LDS inter-period storage is active. A genuinely
      fractional matrix is rejected by LDS code paths (guard below
      points to Phase 16.5).

    ``period_resolution_hours`` records the wall-clock duration of a
    single timestep inside a rep period — needed so consumers that scale
    energy (MWh vs MW·step) don't have to re-derive ``dt``.
    """
    n_periods: int
    period_length: int  # timesteps per period (e.g., 24 for daily)
    medoid_indices: np.ndarray  # which original periods were selected
    weights: np.ndarray  # how many original periods each represents
    mapping: np.ndarray  # for each original period, which representative it maps to
    profiles: np.ndarray  # shape (n_periods, period_length, n_features)
    mapping_matrix: np.ndarray | None = None
    period_resolution_hours: float = 1.0
    # Column order of the last axis of ``profiles`` (the aggregation feature
    # names, in the order they were stacked). When populated,
    # ``apply_representative_days`` indexes profiles by this order rather than
    # re-deriving it from ``sorted(timeseries_map.keys())`` — which is wrong
    # whenever the aggregation used selection-only features (e.g. a residual)
    # not present in the apply map. ``None`` falls back to the legacy behaviour.
    feature_names: list | None = None

    def __post_init__(self):
        if self.mapping_matrix is None:
            return
        M = np.asarray(self.mapping_matrix, dtype=np.float64)
        if M.ndim != 2 or M.shape[1] != self.n_periods:
            raise ValueError(
                f"mapping_matrix must have shape (n_original_periods, "
                f"{self.n_periods}); got {M.shape}")
        if M.shape[0] != len(self.mapping):
            raise ValueError(
                f"mapping_matrix rows ({M.shape[0]}) must match len(mapping) "
                f"({len(self.mapping)})")
        if (M < -1e-12).any() or (M > 1.0 + 1e-12).any():
            raise ValueError(
                "mapping_matrix entries must lie in [0, 1]")
        row_sums = M.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            bad = int(np.argmax(np.abs(row_sums - 1.0)))
            raise ValueError(
                f"mapping_matrix rows must sum to 1; row {bad} sums to "
                f"{row_sums[bad]:.6g}")
        self.mapping_matrix = M

    @classmethod
    def from_fractional_matrix(
        cls,
        mapping_matrix: np.ndarray,
        profiles: np.ndarray,
        period_length: int,
        period_resolution_hours: float = 1.0,
    ) -> "RepresentativePeriods":
        """Build a ``RepresentativePeriods`` from a Tulipa-style fractional
        period→rep matrix.

        Weights are derived as column sums of ``mapping_matrix`` (so
        ``sum(weights) == n_original_periods``). ``mapping`` is populated by
        argmax along the rep axis so single-rep consumers (F5/LDS integer
        recursion) still see a valid assignment — but users who request LDS
        with a non-degenerate matrix will hit the guard in
        ``apply_representative_days``.
        """
        M = np.asarray(mapping_matrix, dtype=np.float64)
        profiles = np.asarray(profiles, dtype=np.float64)
        if profiles.ndim != 3:
            raise ValueError(
                f"profiles must be (n_periods, period_length, n_features); "
                f"got shape {profiles.shape}")
        n_periods = profiles.shape[0]
        if M.shape[1] != n_periods:
            raise ValueError(
                f"mapping_matrix has {M.shape[1]} columns but profiles has "
                f"{n_periods} rep periods")
        weights = M.sum(axis=0)
        mapping = np.argmax(M, axis=1).astype(np.int64)
        medoid_indices = np.arange(n_periods, dtype=np.int64)
        return cls(
            n_periods=n_periods,
            period_length=period_length,
            medoid_indices=medoid_indices,
            weights=weights,
            mapping=mapping,
            profiles=profiles,
            mapping_matrix=M,
            period_resolution_hours=period_resolution_hours,
        )


def _is_one_hot(matrix: np.ndarray, tol: float = 1e-9) -> bool:
    """True iff every row of ``matrix`` has exactly one entry near 1 and
    the rest near 0 — i.e., the fractional mapping degenerates to an
    integer one-rep-per-day assignment and can be fed to LDS safely.
    """
    M = np.asarray(matrix)
    if M.size == 0:
        return True
    near_one = np.isclose(M, 1.0, atol=tol)
    near_zero = np.isclose(M, 0.0, atol=tol)
    if not np.all(near_one | near_zero):
        return False
    return np.all(near_one.sum(axis=1) == 1)


def k_medoids(data: np.ndarray, k: int, max_iter: int = 100,
              seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    K-medoids clustering (PAM algorithm) with k-means++ seeding.

    Args:
        data: (n_samples, n_features) array
        k: number of clusters
        max_iter: maximum iterations
        seed: random seed

    Returns:
        (medoid_indices, labels, distances)
    """
    rng = np.random.RandomState(seed)
    n = len(data)

    # Pairwise distance matrix (squared Euclidean for speed).
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    dist_matrix = np.sum(diff ** 2, axis=2)

    # k-medoids++ seeding: pick first medoid uniformly, then each
    # subsequent medoid with probability proportional to its squared
    # distance from the nearest already-chosen medoid. Avoids the
    # collapse mode where random init lands every medoid in one cluster.
    medoids = np.empty(k, dtype=np.int64)
    medoids[0] = rng.randint(n)
    nearest = dist_matrix[:, medoids[0]].copy()
    for i in range(1, k):
        total = nearest.sum()
        if total <= 0:
            medoids[i] = rng.randint(n)
        else:
            r = rng.uniform(0, total)
            cum = np.cumsum(nearest)
            medoids[i] = int(np.searchsorted(cum, r))
        # Update nearest-distance tracker.
        nearest = np.minimum(nearest, dist_matrix[:, medoids[i]])

    for _ in range(max_iter):
        # Assign each point to nearest medoid
        dists_to_medoids = dist_matrix[:, medoids]  # (n, k)
        labels = np.argmin(dists_to_medoids, axis=1)

        # Update medoids: for each cluster, pick the point that
        # minimizes total distance to all other cluster members
        new_medoids = medoids.copy()
        for c in range(k):
            cluster_mask = labels == c
            if not np.any(cluster_mask):
                continue
            cluster_indices = np.where(cluster_mask)[0]
            # Sum of distances from each cluster member to all others in cluster
            intra_dists = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
            total_dists = intra_dists.sum(axis=1)
            best = cluster_indices[np.argmin(total_dists)]
            new_medoids[c] = best

        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    # Final assignment
    dists_to_medoids = dist_matrix[:, medoids]
    labels = np.argmin(dists_to_medoids, axis=1)
    distances = dists_to_medoids[np.arange(n), labels]

    return medoids, labels, distances


def aggregate_to_representative_days(
    timeseries: dict[str, np.ndarray],
    n_days: int = 12,
    hours_per_day: int = 24,
    seed: int = 42,
    extreme_periods: list[tuple[str, str]] | None = None,
) -> RepresentativePeriods:
    """
    Aggregate time-series data into representative days using k-medoids.

    Args:
        timeseries: dict mapping names to 1D arrays of hourly values
            Example: {"demand": demand_8760, "solar_cf": solar_8760}
        n_days: total number of representative days (k-medoids + extremes)
        hours_per_day: timesteps per day (default 24)
        seed: random seed for reproducibility
        extreme_periods: list of ``(direction, feature_name)`` tuples to
            force-include as standalone representatives.
            ``direction`` ∈ {``"max"``, ``"min"``}; ``feature_name`` must be
            a key in ``timeseries``. Each extreme day gets weight 1 (it
            represents only itself), and the originally-mapped k-medoids
            cluster's weight is decremented by 1 so total weight is
            preserved. Critical for capacity adequacy: k-medoids picks
            "typical" days and naturally drops the rare extremes that
            actually drive build decisions.

    Returns:
        RepresentativePeriods with selected days, weights, and profiles.
        ``n_periods`` ≤ ``n_days`` always; if extremes are added, k-medoids
        runs with ``k = n_days - len(extremes)``.
    """
    names = sorted(timeseries.keys())
    arrays = [timeseries[name] for name in names]
    n_hours = min(len(a) for a in arrays)

    n_complete_days = n_hours // hours_per_day
    n_hours = n_complete_days * hours_per_day

    stacked = np.column_stack([a[:n_hours] for a in arrays])
    n_features = stacked.shape[1]
    daily_profiles = stacked.reshape(n_complete_days, hours_per_day, n_features)

    # Resolve extreme-day indices first so k-medoids gets the remaining budget.
    extreme_indices: list[int] = []
    extreme_specs = list(extreme_periods or [])
    feature_idx = {name: i for i, name in enumerate(names)}
    for direction, feat in extreme_specs:
        if feat not in feature_idx:
            raise ValueError(
                f"extreme period feature {feat!r} not in timeseries "
                f"(known: {names})")
        # Per-day aggregate: peak hour for "max", trough hour for "min".
        f_idx = feature_idx[feat]
        per_day = daily_profiles[:, :, f_idx]
        if direction == "max":
            day = int(np.argmax(per_day.max(axis=1)))
        elif direction == "min":
            day = int(np.argmin(per_day.min(axis=1)))
        else:
            raise ValueError(
                f"extreme direction must be 'max' or 'min', got {direction!r}")
        if day not in extreme_indices:
            extreme_indices.append(day)

    n_extreme = len(extreme_indices)
    k_remaining = max(1, min(n_days - n_extreme, n_complete_days - n_extreme))

    flat_profiles = daily_profiles.reshape(n_complete_days, -1)
    means = flat_profiles.mean(axis=0, keepdims=True)
    stds = flat_profiles.std(axis=0, keepdims=True)
    stds[stds < 1e-10] = 1.0
    normalised = (flat_profiles - means) / stds

    # Run k-medoids on the days NOT already chosen as extremes — otherwise
    # k-medoids may rediscover an extreme as a medoid and we'd double-count.
    candidate_mask = np.ones(n_complete_days, dtype=bool)
    candidate_mask[extreme_indices] = False
    candidate_idx = np.where(candidate_mask)[0]
    if len(candidate_idx) >= k_remaining:
        sub_medoids, sub_labels, _ = k_medoids(
            normalised[candidate_idx], k_remaining, seed=seed)
        km_medoids = candidate_idx[sub_medoids]
    else:
        km_medoids = candidate_idx
        k_remaining = len(candidate_idx)
        sub_labels = np.zeros(len(candidate_idx), dtype=int)

    # Final medoid list: k-medoids picks first, then extremes.
    medoids = np.concatenate([km_medoids,
                              np.array(extreme_indices, dtype=np.int64)])
    k_final = len(medoids)

    # Assign every original day to its closest medoid (Euclidean in
    # normalised space). Extremes always map to themselves and stand for
    # exactly one day each.
    labels = np.empty(n_complete_days, dtype=int)
    diff = normalised[:, np.newaxis, :] - normalised[medoids][np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    # Force extreme days to map to their own medoid slot so weight=1 holds.
    for j, ed in enumerate(extreme_indices):
        slot = k_remaining + j
        dist[ed, :] = np.inf
        dist[ed, slot] = 0.0
    labels = np.argmin(dist, axis=1)

    weights = np.zeros(k_final)
    for c in range(k_final):
        weights[c] = np.sum(labels == c)

    rep_profiles = daily_profiles[medoids]

    return RepresentativePeriods(
        n_periods=k_final,
        period_length=hours_per_day,
        medoid_indices=medoids,
        weights=weights,
        mapping=labels,
        profiles=rep_profiles,
        feature_names=list(names),
    )


@dataclass
class AggregationError:
    """Quantitative gap estimate between a TDR aggregation and the
    original full-resolution series. Intended as an a-priori cost
    fidelity proxy — a small NRMSE / duration-curve gap is a good
    indicator that the LP solved on the reduced periods will land
    near the full-resolution objective.

    Fields:
        nrmse: dict[str, float] — normalised RMSE per feature
            (RMSE / range(original)). Lower = better.
        duration_curve_l1: dict[str, float] — mean absolute difference
            between the sorted (descending) reconstructed series and
            the sorted original. Captures whether the aggregation
            preserves the load/renewable duration curve.
        overall_nrmse: float — feature-averaged NRMSE (single scalar).
        n_periods: int
    """
    nrmse: dict
    duration_curve_l1: dict
    overall_nrmse: float
    n_periods: int


def representative_period_error(
    timeseries: dict[str, np.ndarray],
    rep: RepresentativePeriods,
) -> AggregationError:
    """Estimate the aggregation error of a representative-period
    reduction. Reconstructs the full-length series by replaying each
    rep period for as many original periods mapped to it, then compares
    the reconstruction to the input series in two ways:

      1. NRMSE per feature (RMSE / value range).
      2. L1 distance between sorted (descending) reconstructed and
         original series — the "duration curve" fidelity, which is the
         metric capacity-expansion really cares about.
    """
    names = sorted(timeseries.keys())
    arrays = [timeseries[name] for name in names]
    n_hours = min(len(a) for a in arrays)
    n_orig_days = n_hours // rep.period_length
    if n_orig_days != len(rep.mapping):
        raise ValueError(
            f"timeseries length implies {n_orig_days} periods but rep "
            f"mapping has {len(rep.mapping)} entries")

    nrmse = {}
    dc_l1 = {}
    feat_idx = {name: i for i, name in enumerate(names)}
    profiles = rep.profiles  # (k, period_length, n_features)
    for name in names:
        f = feat_idx[name]
        original = arrays[f][:n_orig_days * rep.period_length]
        # Reconstruct: for each original day d, paste the profile of its
        # representative period.
        reconstructed = np.empty_like(original)
        for d in range(n_orig_days):
            p = int(rep.mapping[d])
            reconstructed[d * rep.period_length:(d + 1) * rep.period_length] \
                = profiles[p, :, f]
        rmse = float(np.sqrt(np.mean((original - reconstructed) ** 2)))
        rng = float(original.max() - original.min())
        nrmse[name] = rmse / rng if rng > 0 else 0.0
        dc_l1[name] = float(np.mean(np.abs(
            np.sort(original)[::-1] - np.sort(reconstructed)[::-1])))

    overall = float(np.mean(list(nrmse.values()))) if nrmse else 0.0
    return AggregationError(
        nrmse=nrmse,
        duration_curve_l1=dc_l1,
        overall_nrmse=overall,
        n_periods=rep.n_periods,
    )


def apply_representative_days(
    system,
    rep: RepresentativePeriods,
    timeseries_map: dict[str, str],
    dt: float = 1.0,
):
    """
    Configure an EnergySystem to use representative days.

    Sets the system's timesteps to ``n_periods × period_length``, replaces
    load amounts / generator carrier_factors with the concatenated
    representative-day profiles, and installs per-timestep snapshot weights
    so cost / emission / policy aggregations scale up to the original
    horizon.

    Each timestep inside representative period ``p`` inherits weight
    ``rep.weights[p]`` (the count of original days that period stands for).

    Args:
        system: EnergySystem to modify
        rep: RepresentativePeriods from aggregate_to_representative_days
        timeseries_map: maps feature index names to component names
            Example: {"demand": "load_name", "solar_cf": "solar_gen_name"}
        dt: hours per timestep (default 1.0)

    Note: SOC / bus balance / ramp constraints continue to bind per
    snapshot — Phase 7 LDS work links inter-period storage separately.
    """
    T = rep.n_periods * rep.period_length
    system.set_timesteps(T, dt=dt)
    system._rep_periods = rep

    # Per-timestep weight: every hour of period p counts rep.weights[p] times.
    # Column sums of ``rep.mapping_matrix`` agree with ``rep.weights`` by
    # construction in ``from_fractional_matrix``, so the same broadcast
    # holds whether the aggregation is integer or fractional.
    snapshot_weights = np.repeat(rep.weights, rep.period_length)
    system.set_snapshot_weights(snapshot_weights)

    # Chronological mapping powers Phase 7 LDS inter-period storage
    # (Kotzur 2018). Storages without ``long_duration=True`` ignore it.
    # Phase 16.5 — a fractional mapping_matrix now drives LDS via the
    # generalised (weighted) Kotzur recursion in ``core``: the per-day storage
    # swing and realised intra-SOC become ``Σ_p M[d,p]·soc_intra_p``. The
    # matrix is read off ``system._rep_periods`` (set below), so no extra
    # plumbing is needed here. (Pre-16.5 this raised for fractional + LDS.)
    system.set_chronological_mapping(rep.mapping, rep.period_length)

    # Column order of rep.profiles' last axis. Prefer the order recorded at
    # aggregation time (rep.feature_names) — re-deriving it from
    # sorted(timeseries_map.keys()) silently installs the WRONG profile when
    # the aggregation used selection-only features (e.g. a residual) absent
    # from the apply map. Fall back to the legacy derivation only when
    # feature_names is unavailable (old RepresentativePeriods objects).
    if rep.feature_names is not None:
        feature_idx = {name: i for i, name in enumerate(rep.feature_names)}
    else:
        feature_idx = {name: i for i, name in enumerate(sorted(timeseries_map.keys()))}

    flat_profiles = rep.profiles.reshape(-1, rep.profiles.shape[-1])

    for ts_name, component_name in timeseries_map.items():
        if ts_name not in feature_idx:
            continue
        fidx = feature_idx[ts_name]
        values = flat_profiles[:, fidx]
        for load in system._loads:
            if load.name == component_name:
                load.amount = values
                break
        for gen in system._generators:
            if gen.name == component_name:
                gen.carrier_factor = values
                break


def ml_feature_embedding(
    timeseries: dict[str, np.ndarray],
    hours_per_day: int = 24,
    features: tuple[str, ...] = ("mean", "std", "min", "max",
                                 "ramp_max", "peak_hour", "duration_p95"),
) -> np.ndarray:
    """
    Convert per-day raw timeseries into a low-dimensional statistical
    feature embedding for clustering.

    Motivation (Kim & Sioshansi 2023, Sávio et al. 2024): k-medoids on
    the raw 24-hour vectors treats hour-of-day positions as independent
    dimensions — two days with the same shape but shifted peak hour look
    very different. A feature embedding over aggregate statistics (mean,
    std, ramp, peak position, duration-curve percentile) collapses that
    nuisance while preserving the operationally-relevant structure,
    which improves duration-curve reconstruction on peaky loads.

    Returns a ``(n_days, n_features × n_series)`` array ready to pass
    into :func:`k_medoids` via the existing aggregation pipeline.

    Available features:
      * ``mean``       — per-day average
      * ``std``        — per-day standard deviation
      * ``min``        — per-day min
      * ``max``        — per-day max (capacity-adequacy marker)
      * ``ramp_max``   — max |diff| between adjacent hours (ramp stress)
      * ``peak_hour``  — hour index (0..H-1) of the daily peak
      * ``duration_p95`` — 95th percentile of the day's load / CF values

    Usage::

        emb = ml_feature_embedding(timeseries)  # (n_days, F)
        medoids, labels, _ = k_medoids(emb, k=12)
    """
    names = sorted(timeseries.keys())
    arrays = [timeseries[name] for name in names]
    n_hours = min(len(a) for a in arrays)
    n_days = n_hours // hours_per_day
    n_hours = n_days * hours_per_day

    # (n_days, hours_per_day, n_features)
    stacked = np.column_stack([a[:n_hours] for a in arrays])
    daily = stacked.reshape(n_days, hours_per_day, -1)

    parts: list[np.ndarray] = []
    for f in features:
        if f == "mean":
            parts.append(daily.mean(axis=1))
        elif f == "std":
            parts.append(daily.std(axis=1))
        elif f == "min":
            parts.append(daily.min(axis=1))
        elif f == "max":
            parts.append(daily.max(axis=1))
        elif f == "ramp_max":
            diffs = np.abs(np.diff(daily, axis=1))
            parts.append(diffs.max(axis=1))
        elif f == "peak_hour":
            parts.append(np.argmax(daily, axis=1).astype(float))
        elif f == "duration_p95":
            parts.append(np.percentile(daily, 95, axis=1))
        else:
            raise ValueError(f"unknown feature {f!r}")

    emb = np.concatenate(parts, axis=1)
    # Normalize each column to zero-mean / unit-std so features on different
    # physical scales don't dominate the distance metric.
    mu = emb.mean(axis=0, keepdims=True)
    sd = emb.std(axis=0, keepdims=True)
    sd[sd < 1e-10] = 1.0
    return (emb - mu) / sd


def aggregate_with_feature_embedding(
    timeseries: dict[str, np.ndarray],
    n_days: int = 12,
    hours_per_day: int = 24,
    seed: int = 42,
    features: tuple[str, ...] = ("mean", "std", "min", "max",
                                 "ramp_max", "peak_hour", "duration_p95"),
) -> RepresentativePeriods:
    """
    Alternative to :func:`aggregate_to_representative_days` that clusters
    on an ML-style feature embedding instead of raw normalized profiles.

    Useful when the hour-by-hour shape doesn't generalize (peaky loads
    with varying peak hour) but aggregate statistics do.
    """
    names = sorted(timeseries.keys())
    arrays = [timeseries[name] for name in names]
    n_hours = min(len(a) for a in arrays)
    n_days_total = n_hours // hours_per_day
    n_hours = n_days_total * hours_per_day

    stacked = np.column_stack([a[:n_hours] for a in arrays])
    daily_profiles = stacked.reshape(n_days_total, hours_per_day, -1)

    emb = ml_feature_embedding(timeseries, hours_per_day, features)
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


def rolling_horizon_solve(
    system_factory,
    total_timesteps: int,
    window_size: int,
    overlap: int = 0,
    warm_start: bool = False,
    **solve_kwargs,
) -> dict:
    """
    Solve an energy system using rolling horizon.

    Args:
        system_factory: callable(start_t, end_t) -> EnergySystem
            Creates a system for a time window. The factory should set up
            the correct time-series slices.
        total_timesteps: total number of timesteps (e.g., 8760)
        window_size: timesteps per window (e.g., 168 for one week)
        overlap: number of overlapping timesteps between windows
        warm_start: when True, every window after the first warm-starts from
            the previous window. For MIPs the previous solution is forwarded to
            ``HiGHS.setSolution``. **Phase 10.6 / 11.5 — LP simplex-basis
            hot-start is now live**: the previous window's optimal basis
            (``Result.basis``, exposed by the nexus-opt basis-carryover API,
            N_En_Phase 18.a) is passed via the ``basis=`` solve kwarg when the
            next window has the same structure (equal length). HiGHS then skips
            presolve and resumes dual simplex from the warm basis, eliminating
            the cold start. A truncated final window (different size) silently
            falls back to a cold solve. The optimum is unchanged — only the
            solve path is warmed.

    Returns:
        dict with concatenated dispatch results across all windows.
    """
    results = {
        "generator_dispatch": {},
        "storage_soc": {},
        "total_cost": 0.0,
        "window_results": [],
    }

    step = window_size - overlap
    starts = list(range(0, total_timesteps, step))

    prev_result = None
    prev_span = None
    for i, start in enumerate(starts):
        end = min(start + window_size, total_timesteps)
        if end <= start:
            break

        sys = system_factory(start, end)
        kwargs = dict(solve_kwargs)
        if warm_start and prev_result is not None:
            prev_raw = getattr(prev_result, "_raw", None)
            # MIP variable warm-start (HiGHS.setSolution) — always forwarded.
            kwargs["warm_start"] = prev_raw
            # Phase 10.6 / 11.5 — LP simplex-basis hot-start: reuse the prior
            # window's optimal basis when the model structure matches (same
            # window length ⇒ same var/constraint counts). HiGHS skips presolve
            # and resumes from the warm basis. A size mismatch (truncated final
            # window) falls back to a cold solve.
            prev_basis = getattr(prev_raw, "basis", None) if prev_raw is not None else None
            if prev_basis is not None and prev_span == (end - start):
                kwargs["basis"] = prev_basis
        result = sys.optimise(**kwargs)
        prev_result = result
        prev_span = end - start
        results["window_results"].append(result)
        results["total_cost"] += result.total_cost

        # Determine which timesteps to keep (exclude overlap from previous)
        keep_start = 0 if i == 0 else overlap
        keep_end = end - start

        for name, vals in result.generator_dispatch.items():
            if name not in results["generator_dispatch"]:
                results["generator_dispatch"][name] = []
            results["generator_dispatch"][name].append(vals[keep_start:keep_end])

        for name, vals in result.storage_soc.items():
            if name not in results["storage_soc"]:
                results["storage_soc"][name] = []
            results["storage_soc"][name].append(vals[keep_start:keep_end])

    # Concatenate
    for name in results["generator_dispatch"]:
        results["generator_dispatch"][name] = np.concatenate(
            results["generator_dispatch"][name])
    for name in results["storage_soc"]:
        results["storage_soc"][name] = np.concatenate(
            results["storage_soc"][name])

    return results


# ---------------------------------------------------------------------------
# Phase 7.3 / 7.4 — variable-resolution (adaptive) time clustering
# ---------------------------------------------------------------------------


@dataclass
class ResolutionPlan:
    """Result of an adaptive / multi-resolution time clustering.

    * ``boundaries`` — start index (in the ORIGINAL clock) of each segment;
      ``len == n_segments``.
    * ``durations`` — wall-clock hours spanned by each segment
      (``n_steps_in_segment × dt``).
    * ``representatives`` — ``(n_segments, n_features)`` per-segment mean of
      the driving feature matrix (used to rewrite load / carrier-factor series).
    * ``max_abs_error`` / ``mean_abs_error`` — aggregation error of the
      representative values vs the original series (per Pineda & Morales 2018
      chronological time-period clustering — the deviation a downstream model
      inherits from collapsing the clock).
    """
    boundaries: np.ndarray
    durations: np.ndarray
    representatives: np.ndarray
    n_original: int
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0

    @property
    def n_segments(self) -> int:
        return int(len(self.boundaries))

    @property
    def compression(self) -> float:
        return self.n_original / max(1, self.n_segments)


def _normalise_columns(features: np.ndarray) -> np.ndarray:
    f = np.asarray(features, dtype=float)
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    rng = f.max(axis=0) - f.min(axis=0)
    rng[rng == 0] = 1.0
    return (f - f.min(axis=0)) / rng


def adaptive_resolution_plan(
    features: np.ndarray,
    *,
    threshold: float = 0.05,
    dt: float = 1.0,
    max_block: int | None = None,
) -> ResolutionPlan:
    """Greedy chronological clustering into variable-length segments (Phase 7.3).

    Adjacent timesteps are merged into one segment while every (normalised)
    feature stays within ``threshold`` of the growing segment mean; flat
    stretches collapse into long blocks, volatile / extreme periods stay at
    full resolution. This is the adaptive-timestep idea (high resolution where
    the system is dynamic, low resolution where it is flat) used by SpineOpt /
    Tulipa variable resolution and Pineda & Morales (2018).

    Args:
        features: ``(T,)`` or ``(T, n_features)`` driver series (load,
            renewable CF, price, …). Columns are min-max normalised so a
            single ``threshold`` applies across heterogeneous units.
        threshold: max normalised deviation any step may have from its
            segment mean before a new segment is opened (0 = full resolution).
        dt: hours per ORIGINAL timestep.
        max_block: optional cap on the number of original steps merged into
            one segment (keeps a minimum resolution even on long flat runs).

    Returns:
        ResolutionPlan with segment boundaries, per-segment durations (hours)
        and representative (mean) feature values, plus aggregation error.
    """
    norm = _normalise_columns(features)
    raw = np.asarray(features, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    T = norm.shape[0]
    if T == 0:
        raise ValueError("adaptive_resolution_plan: empty feature series")

    boundaries = [0]
    seg_start = 0
    for t in range(1, T):
        block = norm[seg_start:t + 1]
        within = np.all(np.abs(block - block.mean(axis=0)) <= threshold + 1e-12)
        too_long = (max_block is not None and (t - seg_start + 1) > int(max_block))
        if not within or too_long:
            boundaries.append(t)
            seg_start = t
    boundaries.append(T)  # sentinel end

    segs = list(zip(boundaries[:-1], boundaries[1:]))
    starts = np.array([s for s, _ in segs], dtype=np.int64)
    durations = np.array([(e - s) * dt for s, e in segs], dtype=float)
    reps = np.array([raw[s:e].mean(axis=0) for s, e in segs], dtype=float)

    # Aggregation error: how far each original step's value is from the
    # representative value it was collapsed into.
    err = np.zeros(T)
    for (s, e), rep in zip(segs, reps):
        err[s:e] = np.abs(raw[s:e] - rep).max(axis=1)
    return ResolutionPlan(
        boundaries=starts, durations=durations, representatives=reps,
        n_original=T, max_abs_error=float(err.max()),
        mean_abs_error=float(err.mean()),
    )


def multi_resolution_hierarchy(
    features: np.ndarray,
    *,
    block_sizes: list[int],
    dt: float = 1.0,
) -> list[ResolutionPlan]:
    """Coarse-to-fine nested resolution hierarchy (Phase 7.4).

    Builds one ``ResolutionPlan`` per requested uniform ``block_size`` (in
    original timesteps). Boundaries are *nested*: every coarse block is an
    exact union of finer blocks when the sizes divide, so shadow prices /
    storage levels stay consistent between levels (SpineOpt / Tulipa
    multi-resolution; the levels can drive a coarse master + fine subproblem
    decomposition). ``block_sizes`` should be sorted ascending (finest first);
    e.g. ``[1, 4, 24]`` → hourly, 4-hourly, daily.

    Returns a list of ResolutionPlans, one per block size.
    """
    raw = np.asarray(features, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    T = raw.shape[0]
    plans = []
    for bs in block_sizes:
        bs = int(bs)
        if bs < 1:
            raise ValueError("block_sizes must be >= 1")
        starts = list(range(0, T, bs))
        segs = [(s, min(s + bs, T)) for s in starts]
        durations = np.array([(e - s) * dt for s, e in segs], dtype=float)
        reps = np.array([raw[s:e].mean(axis=0) for s, e in segs], dtype=float)
        err = np.zeros(T)
        for (s, e), rep in zip(segs, reps):
            err[s:e] = np.abs(raw[s:e] - rep).max(axis=1)
        plans.append(ResolutionPlan(
            boundaries=np.array(starts, dtype=np.int64), durations=durations,
            representatives=reps, n_original=T,
            max_abs_error=float(err.max()), mean_abs_error=float(err.mean())))
    return plans


def apply_adaptive_resolution(
    system,
    *,
    threshold: float = 0.05,
    dt: float = 1.0,
    max_block: int | None = None,
    plan: ResolutionPlan | None = None,
) -> ResolutionPlan:
    """Re-clock an EnergySystem onto a variable-resolution grid (Phase 7.3/7.4).

    Collects every time-varying driver already on the system (each
    ``Load.amount`` array and each ``Generator.carrier_factor`` array), builds
    (or reuses) a :class:`ResolutionPlan`, then rewrites those series to their
    per-segment means, sets the number of timesteps to ``n_segments`` and
    installs the per-segment durations via ``set_snapshot_durations`` — so the
    solver moves the correct ``power × duration`` of energy in each block while
    using far fewer timesteps. Returns the plan (with its error bound) so the
    caller can report the exact maximal deviation from the full-resolution
    optimum.

    The clock is strictly opt-in: nothing changes until this is called, and
    the full-resolution model remains the baseline. Pass an explicit ``plan``
    (e.g. one level of :func:`multi_resolution_hierarchy`) to apply a chosen
    resolution instead of deriving one from ``threshold``.
    """
    # Gather driver series in a stable order.
    drivers = []  # (kind, obj, array)
    for load in system._loads:
        a = load.amount
        if isinstance(a, np.ndarray) and a.ndim == 1 and a.size > 1:
            drivers.append(("load", load, a.astype(float)))
    for gen in system._generators:
        cf = gen.carrier_factor
        if isinstance(cf, np.ndarray) and cf.ndim == 1 and cf.size > 1:
            drivers.append(("gen", gen, cf.astype(float)))
    if not drivers:
        raise ValueError(
            "apply_adaptive_resolution: no time-varying load/carrier_factor "
            "series found to cluster.")
    T = max(arr.size for _, _, arr in drivers)

    if plan is None:
        feat = np.column_stack([arr for _, _, arr in drivers if arr.size == T])
        plan = adaptive_resolution_plan(
            feat, threshold=threshold, dt=dt, max_block=max_block)

    segs = list(zip(plan.boundaries.tolist(),
                    plan.boundaries.tolist()[1:] + [T]))
    # Rewrite each driver to its per-segment mean.
    for kind, obj, arr in drivers:
        if arr.size != T:
            continue
        new = np.array([arr[s:e].mean() for s, e in segs], dtype=float)
        if kind == "load":
            obj.amount = new
        else:
            obj.carrier_factor = new

    system.set_timesteps(plan.n_segments, dt=dt)
    system.set_snapshot_durations(plan.durations)
    return plan


# ---------------------------------------------------------------------------
# Phase 19 (Paper 1) — certified a-posteriori error bounds for
# reduced-order (representative-period) energy optimisation.
# ---------------------------------------------------------------------------


@dataclass
class CertifiedBound:
    """A *computable, provably valid* optimality certificate for a model
    solved on a reduced temporal clock (representative periods).

    When a capacity-expansion / dispatch model is solved on a reduced clock
    (k representative periods instead of the full horizon) the resulting
    objective is, in general, **neither** an upper **nor** a lower bound on
    the true full-resolution optimum ``C*`` — period clustering can both
    drop cost-driving extremes (under-estimate) and over-weight an
    expensive medoid (over-estimate). This dataclass brackets ``C*`` with
    two bounds that *are* each individually valid by construction:

    * ``upper_bound`` (a-posteriori / restriction bound) — the reduced
      solution's investment decisions (extendable capacities) fixed into the
      **full**-resolution model, whose operations are then re-optimised over
      the entire horizon. The fixed capacities are a *feasible point* of the
      full investment problem, so its optimal re-dispatched cost is ``≥ C*``
      for a minimisation — a valid **upper** bound. (If the fixed capacities
      are operationally infeasible on some full-resolution period the full
      solve is infeasible and ``upper_bound`` is ``+inf``.)

    * ``lower_bound`` (relaxation bound) — a full-horizon model in which every
      represented period is replaced by an *optimistic surrogate* that
      element-wise dominates it in the cost-reducing direction (per-cluster
      **minimum** demand and **maximum** renewable availability), each
      surrogate weighted by its cluster size. Because every original period's
      feasible region is a superset of its surrogate's and its per-step cost
      is no smaller, the weighted surrogate optimum is ``≤ C*`` — a valid
      **lower** bound. This is tighter than the naive "drop the unselected
      periods' balance constraints" subset relaxation while remaining
      rigorously valid.

    The **certified gap** ``(upper_bound − lower_bound) / lower_bound`` is a
    guaranteed envelope: ``C*`` is provably inside ``[lower_bound,
    upper_bound]``, so ``gap_pct`` is an upper bound on the *true* reduction
    error ``(reduced_cost − C*)/C*`` in magnitude. A small certified gap is a
    machine-checkable proof that the chosen reduction is good without ever
    solving the (expensive) full model.

    Fields
    ------
    lower_bound : float   valid lower bound on the full-resolution optimum C*
    upper_bound : float   valid upper bound on C* (feasible-point cost; may be +inf)
    gap_abs     : float   upper_bound − lower_bound
    gap_pct     : float   100 · gap_abs / lower_bound  (certified % envelope)
    full_cost   : float   the upper_bound, i.e. the reduced plan evaluated on
                          the full horizon (the best *implementable* cost)
    reduced_cost: float   objective the reduced model itself reported (context
                          only — NOT a bound; recorded so callers can see how
                          far the reduced clock's own number sits inside the
                          certified envelope)

    References
    ----------
    A-posteriori restriction/feasible-point bounding and relaxation lower
    bounds for aggregated optimisation: Bahl et al. (2018) "Rigorous
    synthesis of energy systems by decomposition via time-series
    aggregation"; Hoffmann et al. (2020) review of TSA bounding; the
    optimistic-surrogate relaxation is the time-series analogue of scenario
    lower-bounding in stochastic programming.
    """
    lower_bound: float
    upper_bound: float
    gap_abs: float
    gap_pct: float
    full_cost: float
    reduced_cost: float

    @property
    def brackets_optimum(self) -> bool:
        """True iff the certificate is a finite, well-ordered envelope
        (``lower_bound ≤ upper_bound`` and both finite)."""
        return (np.isfinite(self.lower_bound) and np.isfinite(self.upper_bound)
                and self.lower_bound <= self.upper_bound + 1e-6)


def _gather_full_series(system) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Pull the time-varying driver series off a fully-built full-resolution
    EnergySystem. Returns ``(load_series, gen_cf_series, T)`` where each dict
    maps component name → 1-D array of length T."""
    loads: dict[str, np.ndarray] = {}
    gens: dict[str, np.ndarray] = {}
    T = 0
    for ld in system._loads:
        a = ld.amount
        if isinstance(a, np.ndarray) and a.ndim == 1 and a.size > 1:
            loads[ld.name] = a.astype(float)
            T = max(T, a.size)
    for g in system._generators:
        cf = g.carrier_factor
        if isinstance(cf, np.ndarray) and cf.ndim == 1 and cf.size > 1:
            gens[g.name] = cf.astype(float)
            T = max(T, cf.size)
    return loads, gens, T


def _cluster_mapping_from(
    reduced_system_or_plan,
    full_loads: dict[str, np.ndarray],
    full_gens: dict[str, np.ndarray],
    T: int,
    period_length: int,
) -> tuple[np.ndarray, int]:
    """Resolve a per-original-period → cluster assignment and the period
    length from whatever reduction descriptor the caller passed.

    Accepts a :class:`RepresentativePeriods` directly, or an EnergySystem on
    which ``apply_representative_days`` has stashed ``_rep_periods``.
    """
    rep = reduced_system_or_plan
    if isinstance(rep, RepresentativePeriods):
        return np.asarray(rep.mapping, dtype=np.int64), int(rep.period_length)
    rp = getattr(rep, "_rep_periods", None)
    if isinstance(rp, RepresentativePeriods):
        return np.asarray(rp.mapping, dtype=np.int64), int(rp.period_length)
    raise ValueError(
        "certify_reduction: reduced_system_or_plan must be a "
        "RepresentativePeriods or an EnergySystem configured by "
        "apply_representative_days (so it carries `_rep_periods`).")


def certify_reduction(
    full_system_factory,
    reduced_system_or_plan,
    reduced_capacities: dict[str, float],
    *,
    reduced_cost: float = float("nan"),
    period_length: int | None = None,
    **solve_kwargs,
) -> CertifiedBound:
    """Certify how far a representative-period reduction's solution can be
    from the true full-resolution optimum (Phase 19, Paper 1).

    Given (a) a way to (re)build the FULL-resolution model and (b) the
    capacities a reduced-clock solve produced, compute a *provably valid*
    bracket ``[lower_bound, upper_bound]`` around the unknown full optimum
    ``C*`` and return it as a :class:`CertifiedBound`. Neither the full model
    nor its exact optimum is ever required — the certificate is a-posteriori.

    Construction (see :class:`CertifiedBound` for the validity argument):

    * **upper_bound** — fix ``reduced_capacities`` into a fresh full-resolution
      model via ``optimise(benders_fix_caps=...)`` and re-optimise operations
      over the full horizon. A feasible point of the full investment problem ⇒
      a valid upper bound on ``C*``. Infeasible fixed caps ⇒ ``+inf``.
    * **lower_bound** — build a full-horizon model whose per-cluster periods are
      replaced by the cluster's *optimistic surrogate* (element-wise min demand,
      max renewable availability) weighted by cluster size, and solve with
      capacities free. Element-wise domination in the cost-reducing direction ⇒
      a valid lower bound on ``C*``.

    Args:
        full_system_factory: zero-argument callable returning a freshly built
            full-resolution :class:`EnergySystem` with its time-varying
            ``Load.amount`` / ``Generator.carrier_factor`` series and
            extendable capacities set, but NOT yet solved. Called twice (once
            for the upper bound, once — as a structural template — for the
            lower bound).
        reduced_system_or_plan: the :class:`RepresentativePeriods` used for the
            reduction, or the reduced EnergySystem that ``apply_representative_days``
            configured (it stashes ``_rep_periods``). Supplies the cluster
            assignment used to build the optimistic surrogate.
        reduced_capacities: ``capacity_additions`` dict from the reduced solve —
            the extendable capacities to fix into the full model for the upper
            bound. Names follow the ``benders_fix_caps`` convention.
        reduced_cost: (optional) the objective the reduced solve reported, stored
            on the result for context. Not used in any bound.
        period_length: timesteps per representative period; inferred from the
            reduction descriptor if omitted.
        **solve_kwargs: forwarded to every internal ``optimise()`` call.

    Returns:
        CertifiedBound with ``lower_bound ≤ C* ≤ upper_bound`` (when both
        finite), the absolute / percentage certified gap, and the reduced
        plan's full-horizon cost.
    """
    # --- Upper bound: feasible-point projection of the reduced plan ---
    ub_sys = full_system_factory()
    full_loads, full_gens, T = _gather_full_series(ub_sys)
    if T == 0:
        raise ValueError(
            "certify_reduction: full system has no time-varying driver "
            "series (load.amount / gen.carrier_factor) to certify over.")

    mapping, pl = _cluster_mapping_from(
        reduced_system_or_plan, full_loads, full_gens, T,
        period_length if period_length is not None else 1)
    if period_length is not None:
        pl = int(period_length)
    if pl <= 0:
        raise ValueError("certify_reduction: period_length must be positive.")

    n_orig = T // pl
    if n_orig != len(mapping):
        raise ValueError(
            f"certify_reduction: full horizon implies {n_orig} periods of "
            f"length {pl} but the reduction mapping has {len(mapping)} entries.")

    ub_res = ub_sys.optimise(benders_fix_caps=dict(reduced_capacities),
                             **solve_kwargs)
    if ub_res.status == "optimal":
        upper_bound = float(ub_res.total_cost)
    else:
        # Reduced plan is infeasible on the full horizon → no finite feasible
        # point from it; the honest valid upper bound is +inf.
        upper_bound = float("inf")

    # --- Lower bound: weighted optimistic-surrogate relaxation ---
    lb_sys = full_system_factory()

    clusters = sorted(set(int(c) for c in mapping.tolist()))
    cluster_weight = {c: int(np.sum(mapping == c)) for c in clusters}

    def _optimistic(series: np.ndarray, reducer) -> np.ndarray:
        """Per-cluster element-wise reduction of the (n_orig, pl) day matrix."""
        daily = series[:n_orig * pl].reshape(n_orig, pl)
        out = np.empty((len(clusters), pl), dtype=float)
        for j, c in enumerate(clusters):
            days = np.where(mapping == c)[0]
            out[j] = reducer(daily[days], axis=0)
        return out.reshape(-1)

    # Loads: cheapest = LOWEST demand in each cluster.
    for ld in lb_sys._loads:
        if ld.name in full_loads:
            ld.amount = _optimistic(full_loads[ld.name], np.min)
    # Generator availability: cheapest = MOST renewable headroom in each cluster.
    for g in lb_sys._generators:
        if g.name in full_gens:
            g.carrier_factor = _optimistic(full_gens[g.name], np.max)

    n_clusters = len(clusters)
    lb_sys.set_timesteps(n_clusters * pl, dt=lb_sys._dt)
    # Clear any rep-period plumbing the factory may have carried so we drive a
    # plain weighted solve (no LDS recursion on the surrogate).
    lb_sys._rep_periods = None
    lb_sys._chrono_mapping = None
    lb_sys._period_length = None
    lb_sys.set_snapshot_durations(None)
    weights = np.repeat(
        np.array([cluster_weight[c] for c in clusters], dtype=float), pl)
    lb_sys.set_snapshot_weights(weights)

    lb_res = lb_sys.optimise(**solve_kwargs)
    if lb_res.status != "optimal":
        raise RuntimeError(
            f"certify_reduction: optimistic-surrogate lower-bound solve did "
            f"not reach optimality (status={lb_res.status!r}).")
    lower_bound = float(lb_res.total_cost)

    gap_abs = upper_bound - lower_bound
    gap_pct = (100.0 * gap_abs / lower_bound) if np.isfinite(upper_bound) and lower_bound > 0 else float("inf")

    return CertifiedBound(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        gap_abs=gap_abs,
        gap_pct=gap_pct,
        full_cost=upper_bound,
        reduced_cost=float(reduced_cost),
    )


def certified_reduction_demo(
    *,
    n_days: int = 8,
    hours_per_day: int = 6,
    n_rep: int = 3,
    seed: int = 3,
) -> dict:
    """Tiny self-contained demo: build a full capacity-expansion instance,
    reduce it to ``n_rep`` representative days, certify the reduction, then
    (because the instance is tiny) solve the full model exactly to recover the
    true optimum and confirm the certificate brackets it.

    Returns a dict with ``certified`` (the :class:`CertifiedBound`),
    ``true_optimum``, ``actual_gap_pct`` (the genuine reduced-vs-true error)
    and ``certified_gap_pct`` — which should satisfy
    ``certified_gap_pct ≥ actual_gap_pct`` and both small when the reduction is
    good. Used by the Phase 19 test and as a runnable example.
    """
    from .core import EnergySystem  # local import: avoid module import cycle

    hours = np.arange(hours_per_day)
    rng = np.random.default_rng(seed)
    loads, solars = [], []
    for _ in range(n_days):
        base = rng.uniform(40, 90)
        amp = rng.uniform(0, 60)
        pk = rng.integers(0, hours_per_day)
        loads.append(base + amp * np.exp(-((hours - pk) ** 2) / 2.0))
        solars.append(np.clip(
            rng.uniform(0, 1) * np.sin((hours - 1) / hours_per_day * np.pi),
            0, 1))
    full_load = np.concatenate(loads)
    full_solar = np.concatenate(solars)

    def make_full() -> "EnergySystem":
        s = EnergySystem("cert_demo")
        b = s.add_bus("elec")
        g = s.add_generator("solar", b, capacity=1.0, marginal_cost=0.0,
                            extendable=True, max_capacity=500.0,
                            capital_cost=50.0)
        g.carrier_factor = full_solar.copy()
        s.add_generator("gas", b, capacity=1.0, marginal_cost=40.0,
                        extendable=True, max_capacity=500.0, capital_cost=10.0)
        s.add_load("demand", b, amount=full_load.copy())
        s.set_timesteps(n_days * hours_per_day, dt=1.0)
        return s

    # Reduced solve.
    rep = aggregate_to_representative_days(
        {"load": full_load, "solar": full_solar},
        n_days=n_rep, hours_per_day=hours_per_day)
    red = make_full()
    apply_representative_days(red, rep, {"load": "demand", "solar": "solar"})
    red_res = red.optimise()

    # Certificate.
    cert = certify_reduction(
        make_full, rep, dict(red_res.capacity_additions),
        reduced_cost=red_res.total_cost, period_length=hours_per_day)

    # Ground truth (tiny instance only).
    true_res = make_full().optimise()
    true_opt = float(true_res.total_cost)
    actual_gap_pct = abs(cert.full_cost - true_opt) / true_opt * 100.0

    return {
        "certified": cert,
        "true_optimum": true_opt,
        "reduced_cost": float(red_res.total_cost),
        "full_eval_cost": cert.full_cost,
        "actual_gap_pct": actual_gap_pct,
        "certified_gap_pct": cert.gap_pct,
    }
