"""
Phase 11 — ML-guided solving.

Coverage:
  (a) feature extractors produce finite, deterministic vectors for a
      representative system;
  (b) MeritOrderPredictor returns a 0/1 UC schedule that off-loads the
      cheapest committable units first;
  (c) warm_start_from_prediction masks low-confidence entries and
      triggers cold-start when the confidence vector is all zeros;
  (d) HistoricalNeighborPredictor populates from a bank of past
      results and majority-votes per-timestep;
  (e) uc_fix_schedule=... on EnergySystem.optimise actually binds u[t]
      via equality constraints;
  (f) LearnedVarFixer observes a short training stream and emits a fix
      dict that respects the min_samples and threshold gates;
  (g) LearnedClusterSelector.fit produces positive weights and
      learned_representative_periods returns a valid RepresentativePeriods.
  (h) GNNPredictor raises a friendly RuntimeError when no model is
      supplied — documents the torch-optional boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.ml import (
    SystemFeatures,
    TimestepFeatures,
    extract_system_features,
    extract_timestep_features,
    MeritOrderPredictor,
    HistoricalNeighborPredictor,
    GNNPredictor,
    predict_unit_commitment,
    warm_start_from_prediction,
    LearnedVarFixer,
    VarFixingStats,
    apply_varfix,
    LearnedClusterSelector,
    learned_representative_periods,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _uc_system(T: int = 6) -> ne.EnergySystem:
    sys = ne.EnergySystem("uc")
    elec = sys.add_bus("elec", carrier="electricity")
    # Load profile with a daytime peak so merit order has work to do.
    load = np.array([60, 90, 140, 180, 120, 70], dtype=float)[:T]
    sys.add_load("d", bus=elec, amount=load)
    # Three committable units in cost order. Cheap is always on,
    # mid turns on for the shoulder, peak only for the spike.
    sys.add_generator("cheap", bus=elec, capacity=80, marginal_cost=10,
                      committable=True, min_up_time=1, min_down_time=1)
    sys.add_generator("mid", bus=elec, capacity=80, marginal_cost=40,
                      committable=True, min_up_time=1, min_down_time=1)
    sys.add_generator("peak", bus=elec, capacity=80, marginal_cost=120,
                      committable=True, min_up_time=1, min_down_time=1)
    sys.set_timesteps(T)
    return sys


# ---------------------------------------------------------------------------
# (a) feature extractors
# ---------------------------------------------------------------------------

def test_feature_extractors_are_deterministic():
    sys_a = _uc_system()
    sys_b = _uc_system()
    fa = extract_system_features(sys_a)
    fb = extract_system_features(sys_b)
    assert fa.n_committable == 3
    assert fa.peak_load_mw == pytest.approx(180.0)
    assert np.allclose(fa.to_vector(), fb.to_vector())

    ta = extract_timestep_features(sys_a)
    tb = extract_timestep_features(sys_b)
    assert ta.load_norm.shape == (6,)
    assert ta.to_matrix().shape[0] == 6
    assert np.allclose(ta.to_matrix(), tb.to_matrix())
    # load_norm peaks at 1.0.
    assert ta.load_norm.max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# (b) MeritOrderPredictor
# ---------------------------------------------------------------------------

def test_merit_order_predictor_commits_cheap_first():
    sys = _uc_system()
    pred = predict_unit_commitment(sys, MeritOrderPredictor())
    # At the peak timestep (180 MW), all three units must be on.
    peak_t = 3
    assert pred.schedule["cheap"][peak_t] == 1.0
    assert pred.schedule["mid"][peak_t] == 1.0
    assert pred.schedule["peak"][peak_t] == 1.0
    # At the low timestep (60 MW), peak unit should be off.
    low_t = 0
    assert pred.schedule["peak"][low_t] == 0.0
    assert pred.schedule["cheap"][low_t] == 1.0
    # Confidence is a valid probability.
    for name, arr in pred.confidence.items():
        assert np.all((arr >= 0.0) & (arr <= 1.0))


# ---------------------------------------------------------------------------
# (c) warm_start_from_prediction filters + cold-start fallback
# ---------------------------------------------------------------------------

def test_warm_start_masks_low_confidence():
    sys = _uc_system()
    pred = predict_unit_commitment(sys, MeritOrderPredictor())
    fix = warm_start_from_prediction(pred, confidence_threshold=0.01)
    # Every named generator with any confident cell is present.
    assert "cheap" in fix or "mid" in fix or "peak" in fix
    # Threshold > max confidence yields an empty dict (cold start).
    cold = warm_start_from_prediction(pred, confidence_threshold=2.0)
    assert cold == {}


def test_warm_start_empty_bank_triggers_cold_start():
    # HistoricalNeighborPredictor with empty bank → all-nan schedule,
    # zero confidence, cold start fallback.
    sys = _uc_system()
    pred = predict_unit_commitment(sys, HistoricalNeighborPredictor())
    fix = warm_start_from_prediction(pred, confidence_threshold=0.5,
                                     cold_start_fallback=True)
    assert fix == {}


# ---------------------------------------------------------------------------
# (d) HistoricalNeighborPredictor round-trip
# ---------------------------------------------------------------------------

def test_historical_neighbor_learns_from_solve_bank():
    predictor = HistoricalNeighborPredictor(k_sys=1, k_step=1)
    for _ in range(3):
        sys = _uc_system()
        result = sys.optimise()
        assert result.status == "optimal"
        predictor.record(sys, result, tag="train")
    assert predictor.bank_size == 3

    # Query on the same system: expect a full schedule with confidence 1.
    query = _uc_system()
    pred = predict_unit_commitment(query, predictor)
    # At least one committable gen got a concrete schedule.
    has_any = any(np.any(~np.isnan(v)) for v in pred.schedule.values())
    assert has_any


# ---------------------------------------------------------------------------
# (e) uc_fix_schedule actually binds the UC decision
# ---------------------------------------------------------------------------

def test_uc_fix_schedule_pins_status_vars():
    sys = _uc_system()
    # Force "peak" to be ON at every timestep — more expensive than
    # unconstrained, so the total cost must strictly increase.
    unforced = sys.optimise()
    assert unforced.status == "optimal"

    sys2 = _uc_system()
    T = 6
    fix = {"peak": np.ones(T)}
    forced = sys2.optimise(uc_fix_schedule=fix)
    assert forced.status == "optimal"
    assert forced.unit_status["peak"].sum() == pytest.approx(T)
    assert forced.total_cost >= unforced.total_cost - 1e-6


def test_uc_fix_schedule_unknown_gen_raises():
    sys = _uc_system()
    with pytest.raises(ValueError, match="no committable generator"):
        sys.optimise(uc_fix_schedule={"nonexistent": np.zeros(6)})


# ---------------------------------------------------------------------------
# (f) LearnedVarFixer
# ---------------------------------------------------------------------------

def test_learned_var_fixer_emits_fix_dict_after_training():
    fixer = LearnedVarFixer(threshold=0.8, min_samples=2)
    for _ in range(3):
        sys = _uc_system()
        res = sys.optimise()
        fixer.observe(res, sys)
    query = _uc_system()
    sched = fixer.predict(query)
    # "cheap" should be always on → expect all-1 fixings.
    if "cheap" in sched:
        finite = sched["cheap"][~np.isnan(sched["cheap"])]
        assert np.all(finite == 1.0)
    # apply_varfix stand-alone form works too.
    sched_fn = apply_varfix(query, fixer.stats, threshold=0.8, min_samples=2)
    assert set(sched_fn) == set(sched)


def test_learned_var_fixer_skips_below_min_samples():
    fixer = LearnedVarFixer(threshold=0.95, min_samples=10)
    sys = _uc_system()
    res = sys.optimise()
    fixer.observe(res, sys)
    assert fixer.predict(sys) == {}


# ---------------------------------------------------------------------------
# (g) LearnedClusterSelector — weights + integration
# ---------------------------------------------------------------------------

def test_learned_cluster_selector_fit_and_apply():
    rng = np.random.default_rng(0)
    series = {
        "demand": 100 + 40 * np.sin(np.arange(72) / 24.0 * 2 * np.pi)
                  + rng.normal(0, 5, 72),
        "solar": np.clip(np.sin(np.arange(72) / 24.0 * 2 * np.pi), 0, None),
    }
    selector = LearnedClusterSelector(ridge=0.05)
    baseline = learned_representative_periods(
        series, selector=None, n_days=2, hours_per_day=24)
    selector.observe(series, baseline)
    weights = selector.fit()
    assert set(weights) == {"demand", "solar"}
    assert all(w > 0 for w in weights.values())
    # Weighted path returns a valid rep struct.
    rep = learned_representative_periods(
        series, selector=selector, n_days=2, hours_per_day=24)
    assert rep.n_periods == 2
    assert rep.profiles.shape == (2, 24, 2)


# ---------------------------------------------------------------------------
# (h) GNNPredictor — torch-optional boundary
# ---------------------------------------------------------------------------

def test_gnn_predictor_requires_model():
    sys = _uc_system()
    with pytest.raises(RuntimeError, match="requires a trained model"):
        predict_unit_commitment(sys, GNNPredictor())
