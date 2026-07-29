"""
Phase 11 — feature extractors for ML-guided solving.

Every ML predictor consumes a pair of feature vectors:

- :class:`SystemFeatures` — static system-level descriptor: generator
  / bus / link counts per technology, total committable capacity,
  average marginal costs, storage power-to-energy ratio. Invariant
  across timesteps; used as a "which regime is this system in?" key
  for nearest-neighbour lookups over the historical solve bank.
- :class:`TimestepFeatures` — per-timestep descriptor: normalised
  load, renewable availability per tech, demand slope (ramp into
  ``t``), hour-of-day / day-of-week if a calendar is supplied.
  Fed to the warm-start predictor once per timestep.

We intentionally keep these as plain ``numpy`` vectors with typed
field tuples so the downstream predictor stays torch-free. A GNN
encoder may lift them to a message-passing graph internally, but the
external surface is flat arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nexus_energy.core import EnergySystem


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SystemFeatures:
    """System-level feature vector (timestep-invariant)."""
    n_buses: int
    n_committable: int
    n_vre: int
    n_storage: int
    n_links: int
    total_committable_mw: float
    total_vre_mw: float
    total_storage_power_mw: float
    total_storage_energy_mwh: float
    avg_marginal_cost: float
    peak_load_mw: float
    # Technology signature: per-tech MW totals in a fixed canonical order.
    tech_keys: tuple[str, ...] = field(default_factory=tuple)
    tech_mw: tuple[float, ...] = field(default_factory=tuple)

    def to_vector(self) -> np.ndarray:
        """Flatten to a fixed-length numpy vector for k-NN / cosine sim."""
        base = np.array([
            float(self.n_buses),
            float(self.n_committable),
            float(self.n_vre),
            float(self.n_storage),
            float(self.n_links),
            float(self.total_committable_mw),
            float(self.total_vre_mw),
            float(self.total_storage_power_mw),
            float(self.total_storage_energy_mwh),
            float(self.avg_marginal_cost),
            float(self.peak_load_mw),
        ], dtype=float)
        return np.concatenate([base, np.asarray(self.tech_mw, dtype=float)])


@dataclass
class TimestepFeatures:
    """Per-timestep feature block, shape (T, F)."""
    load_norm: np.ndarray          # (T,) total load / peak_load, in [0,1+ε].
    load_ramp: np.ndarray          # (T,) (load[t] - load[t-1]) / peak_load.
    vre_availability: np.ndarray   # (T, n_vre) carrier-factor per VRE gen.
    hour_of_day: np.ndarray        # (T,) 0..23 when T % 24 == 0.
    net_load_norm: np.ndarray      # (T,) (load - sum(vre_cap * cf)) / peak_load.
    generator_names: tuple[str, ...] = field(default_factory=tuple)
    vre_names: tuple[str, ...] = field(default_factory=tuple)

    def to_matrix(self) -> np.ndarray:
        """Flatten to an (T, F) 2-D array. Columns:

        ``[load_norm, load_ramp, net_load_norm, hour_sin, hour_cos,
           vre_availability[0..n_vre-1]]``.
        """
        T = self.load_norm.shape[0]
        hour = self.hour_of_day
        hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
        cols = [self.load_norm, self.load_ramp, self.net_load_norm,
                hour_sin, hour_cos]
        mat = np.stack(cols, axis=1)  # (T, 5)
        if self.vre_availability.size > 0:
            mat = np.concatenate([mat, self.vre_availability], axis=1)
        return mat


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_system_features(sys: "EnergySystem") -> SystemFeatures:
    """Build :class:`SystemFeatures` from an :class:`EnergySystem`."""
    n_buses = len(sys._buses)
    n_links = len(sys._links)

    committable = [g for g in sys._generators if g.committable]
    vre = [g for g in sys._generators if g.carrier_factor is not None]
    n_committable = len(committable)
    n_vre = len(vre)
    n_storage = len(sys._storages)

    total_committable_mw = sum(
        float(g.capacity) * (g.n_units if g.clustered else 1)
        for g in committable)
    total_vre_mw = sum(float(g.capacity) for g in vre)
    total_storage_power_mw = sum(float(s.power_capacity) for s in sys._storages)
    total_storage_energy_mwh = sum(
        float(s.energy_capacity) for s in sys._storages)

    if sys._generators:
        avg_mc = float(np.mean([float(g.marginal_cost)
                                for g in sys._generators]))
    else:
        avg_mc = 0.0

    peak_load = 0.0
    for load in sys._loads:
        amt = load.amount
        if isinstance(amt, np.ndarray):
            peak_load += float(np.max(amt))
        else:
            peak_load += float(amt)

    # Tech signature: deterministic-ordered tech -> total MW.
    tech_totals: dict[str, float] = {}
    for gen in sys._generators:
        tag = gen.tech if gen.tech is not None else "_untagged"
        tech_totals[tag] = tech_totals.get(tag, 0.0) + float(gen.capacity)
    keys = tuple(sorted(tech_totals))
    mw = tuple(tech_totals[k] for k in keys)

    return SystemFeatures(
        n_buses=n_buses,
        n_committable=n_committable,
        n_vre=n_vre,
        n_storage=n_storage,
        n_links=n_links,
        total_committable_mw=total_committable_mw,
        total_vre_mw=total_vre_mw,
        total_storage_power_mw=total_storage_power_mw,
        total_storage_energy_mwh=total_storage_energy_mwh,
        avg_marginal_cost=avg_mc,
        peak_load_mw=peak_load,
        tech_keys=keys,
        tech_mw=mw,
    )


def extract_timestep_features(sys: "EnergySystem") -> TimestepFeatures:
    """Build :class:`TimestepFeatures` from an :class:`EnergySystem`.

    The system should have time-aware loads / carrier factors set; a
    static-value system yields one-timestep features.
    """
    sys._infer_timesteps()
    T = max(sys._timesteps, 1)

    # Total load per timestep.
    load = np.zeros(T, dtype=float)
    for ld in sys._loads:
        amt = ld.amount
        if isinstance(amt, np.ndarray):
            load += np.asarray(amt, dtype=float)[:T]
        else:
            load += float(amt)
    peak = float(np.max(load)) if load.size and np.max(load) > 0 else 1.0
    load_norm = load / peak
    load_ramp = np.concatenate([[0.0], np.diff(load)]) / peak

    # VRE availability matrix.
    vre_names: list[str] = []
    vre_cols: list[np.ndarray] = []
    vre_cap_sum = np.zeros(T, dtype=float)
    for gen in sys._generators:
        if gen.carrier_factor is None:
            continue
        cf = np.asarray(gen.carrier_factor, dtype=float)
        if cf.shape[0] < T:
            cf = np.concatenate([cf, np.full(T - cf.shape[0], float(cf[-1]))])
        else:
            cf = cf[:T]
        vre_names.append(gen.name)
        vre_cols.append(cf)
        vre_cap_sum += float(gen.capacity) * cf
    if vre_cols:
        vre_avail = np.stack(vre_cols, axis=1)
    else:
        vre_avail = np.zeros((T, 0), dtype=float)

    net_load_norm = (load - vre_cap_sum) / peak
    hour_of_day = np.arange(T, dtype=float) % 24.0

    return TimestepFeatures(
        load_norm=load_norm,
        load_ramp=load_ramp,
        vre_availability=vre_avail,
        hour_of_day=hour_of_day,
        net_load_norm=net_load_norm,
        generator_names=tuple(g.name for g in sys._generators),
        vre_names=tuple(vre_names),
    )
