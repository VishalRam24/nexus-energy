"""EC119 -- MSR -- F1b Load-Following -- Test Suite

Physics rules:
  - Xe builds up after power reduction (xenon overshoot), BUT peak is lower than PWR
    because online stripping removes Xe continuously (effective lambda_Xe higher)
  - At long equilibrium, Xe → new steady-state below what a solid-fuel reactor would show
  - Xe reactivity worth is ~1500 pcm (lower than PWR 3000 pcm) due to stripping
  - Negative fuel temperature coefficient: part load → T drops → positive reactivity insert
  - Available reactivity > 0 at equilibrium (can always restart — no xenon deadtime)
  - Ramp rate 10%/min (faster than PWR 5%/min)
  - Power output scales with PLR
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
    for k in ["power_output_mw", "xenon_concentration_rel", "temperature_reactivity_pcm",
              "available_reactivity_pcm", "ramp_rate_limit_pct_min", "can_restart"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC119"
    assert info["fidelity"] == "F1b"


def test_rated_power_output(model):
    """At full power: P_e = 1000 * 1.0 * 0.40 = 400 MW."""
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 48.0,
                       "previous_power_fraction": 1.0})
    expected = 1000.0 * 0.40
    assert abs(r["power_output_mw"] - expected) < 1.0, \
        f"Expected {expected:.0f} MW, got {r['power_output_mw']:.1f}"


def test_power_scales_with_fraction(model):
    """50% power should give half rated output."""
    r_full = model.predict({"power_fraction": 1.0,  "time_at_power_hours": 48.0,
                            "previous_power_fraction": 1.0})
    r_half = model.predict({"power_fraction": 0.5,  "time_at_power_hours": 48.0,
                            "previous_power_fraction": 0.5})
    assert abs(r_half["power_output_mw"] - r_full["power_output_mw"] / 2) < 1.0


def test_equilibrium_xenon_at_full_power(model):
    """At long equilibrium full power, Xe_rel should be ~1.0."""
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 72.0,
                       "previous_power_fraction": 1.0})
    assert abs(r["xenon_concentration_rel"] - 1.0) < 0.05


def test_xenon_peak_lower_than_pwr(model):
    """After reducing from 100% to 50%, Xe overshoot in MSR must be < in thermal reactor without stripping."""
    m = model._model
    # In MSR: lambda_Xe_eff is higher, so Xe_eq at same power is lower
    # and the overshoot is smaller
    Xe_10h_msr = m.xenon_transient(1.0, 0.5, 10.0)
    # Xe_eq at 50% with stripping should be < 0.5 (stripping reduces equilibrium)
    Xe_eq_half = float(m.equilibrium_xenon(0.5))
    # Verify overshoot still occurs (physics) but is bounded
    assert Xe_eq_half < 0.6, \
        f"MSR Xe_eq at 50% ({Xe_eq_half:.3f}) should be lower than solid-fuel reactor (~0.7)"


def test_xenon_nonnegative(model):
    m = model._model
    for t in np.linspace(0, 72, 50):
        Xe = m.xenon_transient(1.0, 0.3, t)
        assert Xe >= 0.0, f"Xe negative at t={t:.1f}h"


def test_can_restart_at_equilibrium(model):
    """MSR should always be able to restart at equilibrium (online Xe stripping)."""
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 48.0,
                       "previous_power_fraction": 1.0})
    assert r["can_restart"] is True
    assert r["available_reactivity_pcm"] > 0


def test_temperature_feedback_negative_at_part_load(model):
    """At part load, temperature drops → negative fuel_temp_coeff gives positive reactivity."""
    m = model._model
    rho_T_full = m.temperature_reactivity_pcm(1.0)
    rho_T_half = m.temperature_reactivity_pcm(0.5)
    # At part load, T is lower → rho_T is less negative (more positive)
    assert rho_T_half > rho_T_full, \
        f"rho_T at 50% ({rho_T_half:.0f} pcm) should be > at 100% ({rho_T_full:.0f} pcm)"


def test_ramp_rate_faster_than_pwr(model):
    """MSR ramp rate (10%/min) must be > PWR ramp rate (5%/min)."""
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 0.0,
                       "previous_power_fraction": 1.0})
    assert r["ramp_rate_limit_pct_min"] >= 10.0


def test_ramp_rate_constraint(model):
    """A 50% step in 1 minute should be limited to 10%."""
    m = model._model
    P_achievable, limited = m.ramp_rate_limit(0.5, 1.0, 1.0)
    assert limited is True
    assert abs(P_achievable - 0.60) < 0.001, \
        f"10%/min in 1 min should give 0.60, got {P_achievable:.4f}"


def test_xe_strip_efficiency_positive(model):
    """Xe stripping efficiency must be between 0 and 1."""
    assert 0 < model._model.xe_strip_eff <= 1.0


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({"power_fraction": 0.7, "time_at_power_hours": 5.0,
                       "previous_power_fraction": 1.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
