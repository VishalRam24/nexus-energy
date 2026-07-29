"""EC120 -- FBR -- F1b Load-Following -- Test Suite

Physics rules:
  - Xe reactivity in fast spectrum is negligible (~50 pcm vs ~3000 pcm thermal)
  - Sodium void coefficient is positive (less void at part load → positive reactivity)
  - Doppler coefficient is negative (temperature drops at part load → positive Doppler)
  - Net power coefficient should be manageable within design margins
  - Ramp rate 3%/min (slower than PWR due to large sodium pool)
  - Power output scales with PLR
  - Available reactivity > 0 at full power equilibrium (reactor can operate)
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 48.0,
                       "previous_power_fraction": 1.0})
    for k in ["power_output_mw", "xenon_concentration_rel", "sodium_void_reactivity_pcm",
              "doppler_reactivity_pcm", "available_reactivity_pcm",
              "ramp_rate_limit_pct_min", "can_restart"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC120"
    assert info["fidelity"] == "F1b"


def test_rated_power_output(model):
    """At full power: P_e = 2100 * 1.0 * 0.39 = 819 MW."""
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 48.0,
                       "previous_power_fraction": 1.0})
    expected = 2100.0 * 0.39
    assert abs(r["power_output_mw"] - expected) < 1.0, \
        f"Expected {expected:.0f} MW, got {r['power_output_mw']:.1f}"


def test_power_scales_with_fraction(model):
    r_full = model.predict({"power_fraction": 1.0,  "time_at_power_hours": 48.0,
                            "previous_power_fraction": 1.0})
    r_half = model.predict({"power_fraction": 0.5,  "time_at_power_hours": 48.0,
                            "previous_power_fraction": 0.5})
    assert abs(r_half["power_output_mw"] - r_full["power_output_mw"] / 2) < 1.0


def test_xenon_reactivity_negligible(model):
    """FBR Xe worth must be much smaller than thermal reactor (~50 pcm vs ~3000 pcm)."""
    m = model._model
    xe_worth = abs(m.xe_react_coeff)
    assert xe_worth <= 100.0, \
        f"FBR Xe reactivity worth {xe_worth:.0f} pcm should be <= 100 pcm (fast spectrum)"


def test_fast_spectrum_sigma_xe(model):
    """Fast-spectrum σ_Xe must be << thermal σ_Xe (2.65e-18 cm2)."""
    sigma_xe_fast = model._model.sigma_Xe
    sigma_xe_thermal = 2.65e-18  # thermal reference
    assert sigma_xe_fast < sigma_xe_thermal / 100.0, \
        f"FBR σ_Xe {sigma_xe_fast:.2e} should be << thermal {sigma_xe_thermal:.2e}"


def test_sodium_void_coeff_positive(model):
    """Sodium void coefficient must be positive (FBR characteristic)."""
    assert model._model.void_coeff > 0, \
        "Sodium void coefficient must be positive in large FBR core"


def test_doppler_coeff_negative(model):
    """Doppler coefficient must be negative (stabilizing)."""
    assert model._model.doppler_coeff < 0, \
        "Doppler coefficient must be negative (U-238 resonance absorption)"


def test_sodium_void_feedback_at_part_load(model):
    """At part load, sodium void feedback reactivity should be negative (less void → rho_void<0 at part load)."""
    m = model._model
    rho_void_part = m.sodium_void_reactivity_pcm(0.5)   # less void than rated → negative
    # void_coeff positive, delta_void negative at part load → rho_void negative
    assert rho_void_part < 0, \
        f"At part load, rho_void should be negative (less void), got {rho_void_part:.1f}"


def test_doppler_feedback_positive_at_part_load(model):
    """At part load, Na temperature drops → Doppler feedback positive (negative coeff × negative dT)."""
    m = model._model
    rho_D_part = m.doppler_reactivity_pcm(0.5)
    # T drops at part load → delta_T negative → negative coeff × negative dT → positive
    assert rho_D_part > 0, \
        f"At part load, Doppler feedback should be positive, got {rho_D_part:.1f} pcm"


def test_can_restart_at_equilibrium(model):
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 48.0,
                       "previous_power_fraction": 1.0})
    assert r["can_restart"] is True
    assert r["available_reactivity_pcm"] > 0


def test_ramp_rate_slower_than_pwr(model):
    """FBR ramp rate (3%/min) must be < PWR ramp rate (5%/min)."""
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 0.0,
                       "previous_power_fraction": 1.0})
    assert r["ramp_rate_limit_pct_min"] <= 3.0


def test_ramp_rate_constraint(model):
    """A 50% step in 1 minute limited to 3%."""
    m = model._model
    P_achievable, limited = m.ramp_rate_limit(0.5, 1.0, 1.0)
    assert limited is True
    assert abs(P_achievable - 0.53) < 0.001, \
        f"3%/min in 1 min from 0.5 should give 0.53, got {P_achievable:.4f}"


def test_xenon_nonnegative(model):
    m = model._model
    for t in np.linspace(0, 72, 50):
        Xe = m.xenon_transient(1.0, 0.3, t)
        assert Xe >= 0.0


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({"power_fraction": 0.7, "time_at_power_hours": 5.0,
                       "previous_power_fraction": 1.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
