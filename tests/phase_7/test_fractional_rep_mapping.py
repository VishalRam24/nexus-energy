"""Phase 16.4 — fractional period→rep-period mapping.

A Tulipa-style fractional mapping matrix lets each original day smear
across several rep periods (rows sum to 1). This test suite covers:

1. A degenerate (one-hot) matrix produces the same system state as the
   legacy integer-mapping path — byte-identical snapshot weights and
   chronological mapping — so existing LP fixtures keep passing.
2. Validator rejects malformed matrices (wrong shape, rows that don't
   sum to 1, entries outside [0, 1]).
3. A genuinely fractional matrix threads through
   ``apply_representative_days`` without error when no LDS storage is
   active, and column sums match the reported weights.
4. LDS storage + non-degenerate matrix raises a ``ValueError`` that
   points at Phase 16.5 (integer-only Kotzur recursion).
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.temporal import (
    RepresentativePeriods,
    _is_one_hot,
    apply_representative_days,
    aggregate_to_representative_days,
)


def _tiny_series(n_days: int = 6, hours_per_day: int = 24,
                 seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    load = 50 + 10 * rng.standard_normal(n_days * hours_per_day)
    solar = np.clip(rng.uniform(0, 1, n_days * hours_per_day), 0, 1)
    return {"load": load, "solar_cf": solar}


def _make_system(amount: float = 0.0) -> ne.EnergySystem:
    sys = ne.EnergySystem("rp")
    bus = sys.add_bus("e")
    sys.add_load("load", bus=bus, amount=amount)
    sys.add_generator("solar", bus=bus, capacity=200,
                      marginal_cost=0.0, carrier_factor=0.0)
    sys.add_generator("gas", bus=bus, capacity=500, marginal_cost=50.0)
    return sys


class TestDegenerateEquivalence:
    """One-hot mapping_matrix must reproduce the legacy integer path."""

    def test_one_hot_matches_legacy(self):
        ts = _tiny_series(n_days=6)
        rep_int = aggregate_to_representative_days(
            {"load": ts["load"], "solar_cf": ts["solar_cf"]},
            n_days=2,
        )
        # Build an equivalent one-hot matrix from the integer mapping.
        n_orig = len(rep_int.mapping)
        M = np.zeros((n_orig, rep_int.n_periods))
        M[np.arange(n_orig), rep_int.mapping] = 1.0

        rep_frac = RepresentativePeriods.from_fractional_matrix(
            mapping_matrix=M,
            profiles=rep_int.profiles,
            period_length=rep_int.period_length,
        )

        assert _is_one_hot(rep_frac.mapping_matrix)
        assert np.array_equal(rep_frac.mapping, rep_int.mapping)
        assert np.allclose(rep_frac.weights, rep_int.weights)

        sys_int = _make_system()
        sys_frac = _make_system()
        apply_representative_days(
            sys_int, rep_int,
            timeseries_map={"load": "load", "solar_cf": "solar"})
        apply_representative_days(
            sys_frac, rep_frac,
            timeseries_map={"load": "load", "solar_cf": "solar"})

        assert np.array_equal(sys_int._snapshot_weights,
                              sys_frac._snapshot_weights)
        assert np.array_equal(sys_int._chrono_mapping,
                              sys_frac._chrono_mapping)
        assert sys_int._period_length == sys_frac._period_length


class TestValidation:
    """mapping_matrix validator catches obvious malformations."""

    def _profiles(self, n_periods=2, period_length=24, n_features=1):
        return np.zeros((n_periods, period_length, n_features))

    def test_row_sum_must_be_one(self):
        M = np.array([[0.6, 0.5], [0.4, 0.5]])  # row 0 sums to 1.1
        with pytest.raises(ValueError, match="rows must sum to 1"):
            RepresentativePeriods.from_fractional_matrix(
                mapping_matrix=M,
                profiles=self._profiles(n_periods=2, n_features=1),
                period_length=24,
            )

    def test_entries_must_be_in_unit_interval(self):
        M = np.array([[1.5, -0.5], [0.0, 1.0]])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            RepresentativePeriods.from_fractional_matrix(
                mapping_matrix=M,
                profiles=self._profiles(n_periods=2, n_features=1),
                period_length=24,
            )

    def test_column_count_must_match_profiles(self):
        M = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])  # 3 cols
        with pytest.raises(ValueError, match="rep periods"):
            RepresentativePeriods.from_fractional_matrix(
                mapping_matrix=M,
                profiles=self._profiles(n_periods=2, n_features=1),
                period_length=24,
            )


class TestFractionalWeights:
    """A genuinely fractional matrix threads through end-to-end."""

    def test_column_sums_equal_weights(self):
        # 4 original days, 2 rep periods, some smearing.
        M = np.array([
            [1.0, 0.0],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.0, 1.0],
        ])
        profiles = np.zeros((2, 24, 1))
        rep = RepresentativePeriods.from_fractional_matrix(
            mapping_matrix=M, profiles=profiles, period_length=24,
        )
        assert np.allclose(rep.weights, M.sum(axis=0))
        assert abs(rep.weights.sum() - M.shape[0]) < 1e-9

    def test_apply_without_lds_ok(self):
        M = np.array([
            [1.0, 0.0],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.0, 1.0],
        ])
        profiles = np.zeros((2, 24, 2))
        rep = RepresentativePeriods.from_fractional_matrix(
            mapping_matrix=M, profiles=profiles, period_length=24,
        )
        sys = _make_system()  # no storage -> LDS guard irrelevant
        apply_representative_days(
            sys, rep, timeseries_map={"load": "load", "solar_cf": "solar"})
        # n_periods * period_length timesteps.
        assert sys._timesteps == 2 * 24
        # snapshot weights broadcast the column sums over period_length.
        expected = np.repeat(M.sum(axis=0), 24)
        assert np.allclose(sys._snapshot_weights, expected)


class TestLDSGuard:
    """Phase 16.5 — LDS + fractional mapping_matrix is now SUPPORTED via the
    generalised (weighted) Kotzur recursion; it no longer raises."""

    def test_fractional_matrix_with_lds_now_supported(self):
        M = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ])
        profiles = np.zeros((2, 24, 2))
        rep = RepresentativePeriods.from_fractional_matrix(
            mapping_matrix=M, profiles=profiles, period_length=24,
        )
        sys = _make_system()
        bus = sys._buses[0]
        sys.add_storage(
            "h2_cavern", bus=bus, energy_capacity=1000,
            power_capacity=50, long_duration=True,
        )
        # Pre-16.5 this raised; now it configures cleanly (weighted Kotzur).
        apply_representative_days(
            sys, rep,
            timeseries_map={"load": "load", "solar_cf": "solar"})
        assert sys._rep_periods.mapping_matrix is not None
        r = sys.optimise()
        assert r.status == "optimal"

    def test_degenerate_matrix_with_lds_ok(self):
        # One-hot matrix must not trip the LDS guard.
        M = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        profiles = np.zeros((2, 24, 2))
        rep = RepresentativePeriods.from_fractional_matrix(
            mapping_matrix=M, profiles=profiles, period_length=24,
        )
        sys = _make_system()
        bus = sys._buses[0]
        sys.add_storage(
            "h2_cavern", bus=bus, energy_capacity=1000,
            power_capacity=50, long_duration=True,
        )
        # Should not raise.
        apply_representative_days(
            sys, rep, timeseries_map={"load": "load", "solar_cf": "solar"})
        assert sys._chrono_mapping is not None
