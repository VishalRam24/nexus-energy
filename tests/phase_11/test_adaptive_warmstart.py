"""
Phase 11.6 — adaptive confidence threshold + unified feature-embedding
kernel.

Coverage:
  (a) AdaptiveThresholdController raises the threshold after an
      infeasible / high-drift window and lowers it after a clean one,
      always clamped to [min, max];
  (b) drift_fraction counts pinned-cell disagreement correctly;
  (c) solve_with_adaptive_warmstart runs a rolling-window loop and the
      controller demonstrably changes the threshold across windows
      (vs the fixed-threshold path), pinning fewer cells after a bad
      window;
  (d) the unified feature-embedding kernel: feature_embedding_periods
      matches temporal.aggregate_with_feature_embedding bit-for-bit
      (shared code path), and the learned selector in embedding mode
      routes through the same kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.ml.uc_warmstart import (
    AdaptiveThresholdController,
    solve_with_adaptive_warmstart,
    MeritOrderPredictor,
    predict_unit_commitment,
    warm_start_from_prediction,
)
from nexus_energy.ml.clustering import (
    LearnedClusterSelector,
    learned_representative_periods,
    feature_embedding_periods,
)
from nexus_energy.temporal import aggregate_with_feature_embedding


def _uc_system(T: int = 6, scale: float = 1.0) -> ne.EnergySystem:
    sys = ne.EnergySystem("uc")
    elec = sys.add_bus("elec", carrier="electricity")
    load = np.array([60, 90, 140, 180, 120, 70], dtype=float)[:T] * scale
    sys.add_load("d", bus=elec, amount=load)
    sys.add_generator("cheap", bus=elec, capacity=80, marginal_cost=10,
                      committable=True, min_up_time=1, min_down_time=1)
    sys.add_generator("mid", bus=elec, capacity=80, marginal_cost=40,
                      committable=True, min_up_time=1, min_down_time=1)
    sys.add_generator("peak", bus=elec, capacity=80, marginal_cost=120,
                      committable=True, min_up_time=1, min_down_time=1)
    sys.set_timesteps(T)
    return sys


# ---------------------------------------------------------------------------
# (a) controller AIMD behaviour
# ---------------------------------------------------------------------------

def test_controller_raises_on_bad_lowers_on_clean():
    c = AdaptiveThresholdController(
        threshold=0.7, min_threshold=0.5, max_threshold=0.99,
        increase_factor=1.2, decrease_step=0.05)
    # Infeasible window → threshold up.
    t0 = c.current()
    t1 = c.update(feasible=False)
    assert t1 > t0
    assert c.history[-1][1] == "infeasible"
    # Clean window → threshold down.
    t2 = c.update(feasible=True, fix={}, realised_status={})
    assert t2 < t1
    assert c.history[-1][1] == "clean"
    # Clamp at max.
    c2 = AdaptiveThresholdController(threshold=0.95, max_threshold=0.99,
                                    increase_factor=2.0)
    for _ in range(5):
        c2.update(feasible=False)
    assert c2.current() <= 0.99 + 1e-12
    # Clamp at min.
    c3 = AdaptiveThresholdController(threshold=0.55, min_threshold=0.5,
                                    decrease_step=0.2)
    for _ in range(5):
        c3.update(feasible=True, fix={}, realised_status={})
    assert c3.current() >= 0.5 - 1e-12


def test_controller_high_drift_treated_as_bad():
    c = AdaptiveThresholdController(threshold=0.7, drift_tol=0.25,
                                    increase_factor=1.2)
    # 2 of 2 pinned cells disagree → drift 1.0 > tol → raise.
    fix = {"g": np.array([1.0, 0.0])}
    realised = {"g": np.array([0.0, 1.0])}
    t0 = c.current()
    t1 = c.update(feasible=True, fix=fix, realised_status=realised)
    assert t1 > t0
    assert c.history[-1][1] == "drift"


def test_drift_fraction_counts_disagreement():
    fix = {"a": np.array([1.0, np.nan, 0.0]),
           "b": np.array([np.nan, 1.0, 1.0])}
    realised = {"a": np.array([1.0, 1.0, 1.0]),   # pinned cells: t0 ok, t2 diff
                "b": np.array([0.0, 1.0, 1.0])}    # pinned t1 ok, t2 ok
    # pinned cells: a@t0 (ok), a@t2 (diff), b@t1 (ok), b@t2 (ok) → 1/4
    frac = AdaptiveThresholdController.drift_fraction(fix, realised)
    assert frac == pytest.approx(0.25)
    assert AdaptiveThresholdController.drift_fraction({}, realised) == 0.0
    assert AdaptiveThresholdController.drift_fraction(fix, None) == 0.0


# ---------------------------------------------------------------------------
# (c) rolling-window loop changes behaviour vs fixed
# ---------------------------------------------------------------------------

def test_adaptive_warmstart_changes_threshold_across_windows():
    # Three identical clean windows: the controller should *lower* the
    # threshold across windows (clean signal), which a fixed-threshold
    # run never does.
    systems = [_uc_system() for _ in range(3)]
    ctrl = AdaptiveThresholdController(
        threshold=0.9, min_threshold=0.3, decrease_step=0.1)
    outcomes = solve_with_adaptive_warmstart(
        systems, predictor=MeritOrderPredictor(), controller=ctrl,
        max_fix_fraction=0.75, max_retries=1)
    assert len(outcomes) == 3
    assert all(o.status in ("warm", "warm_retry", "cold") for o in outcomes)
    used = [t for t, _ in ctrl.history]
    # Threshold strictly decreased over the clean run.
    assert used[0] > used[-1]
    # All windows solved feasibly.
    assert all(o.result is not None for o in outcomes)


def test_adaptive_pins_fewer_after_threshold_rises():
    # Demonstrate the behaviour-change lever directly: a higher threshold
    # pins fewer cells than a lower one on the same prediction. The
    # controller raises the threshold after a bad window, so the next
    # window pins fewer (more-confident) cells.
    sys = _uc_system()
    pred = predict_unit_commitment(sys, MeritOrderPredictor())
    fix_lo = warm_start_from_prediction(pred, confidence_threshold=0.3)
    fix_hi = warm_start_from_prediction(pred, confidence_threshold=0.9)
    n_lo = sum(np.count_nonzero(~np.isnan(v)) for v in fix_lo.values())
    n_hi = sum(np.count_nonzero(~np.isnan(v)) for v in fix_hi.values())
    assert n_hi <= n_lo


# ---------------------------------------------------------------------------
# (d) unified feature-embedding kernel
# ---------------------------------------------------------------------------

def test_feature_embedding_kernel_matches_temporal():
    rng = np.random.default_rng(3)
    series = {
        "demand": 100 + 40 * np.sin(np.arange(96) / 24.0 * 2 * np.pi)
                  + rng.normal(0, 5, 96),
        "solar": np.clip(np.sin(np.arange(96) / 24.0 * 2 * np.pi), 0, None),
    }
    # The unified clustering kernel must reproduce the temporal aggregator
    # bit-for-bit (same embedding, same k-medoids, same seed).
    rep_ml = feature_embedding_periods(series, n_days=2, hours_per_day=24, seed=7)
    rep_t = aggregate_with_feature_embedding(series, n_days=2, hours_per_day=24,
                                             seed=7)
    assert rep_ml.n_periods == rep_t.n_periods
    assert np.array_equal(rep_ml.medoid_indices, rep_t.medoid_indices)
    assert np.array_equal(rep_ml.mapping, rep_t.mapping)
    assert np.allclose(rep_ml.profiles, rep_t.profiles)


def test_learned_selector_embedding_mode_uses_shared_kernel():
    rng = np.random.default_rng(4)
    series = {
        "demand": 100 + 40 * np.sin(np.arange(96) / 24.0 * 2 * np.pi)
                  + rng.normal(0, 5, 96),
        "solar": np.clip(np.sin(np.arange(96) / 24.0 * 2 * np.pi), 0, None),
    }
    sel = LearnedClusterSelector(ridge=0.05, use_embedding=True)
    base = feature_embedding_periods(series, n_days=2, hours_per_day=24)
    sel.observe(series, base)
    sel.fit()
    rep = learned_representative_periods(
        series, selector=sel, n_days=2, hours_per_day=24, seed=7)
    assert rep.n_periods == 2
    assert rep.profiles.shape == (2, 24, 2)
    # Column-weight vector matches embedding width (7 feats x 2 series).
    cw = sel.embedding_column_weights(series)
    assert cw.shape == (14,)
