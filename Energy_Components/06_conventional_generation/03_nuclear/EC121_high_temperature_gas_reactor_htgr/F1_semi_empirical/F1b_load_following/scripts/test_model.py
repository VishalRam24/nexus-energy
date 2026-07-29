"""EC121 -- HTGR -- F1b Load-Following -- Test Suite

Physics rules:
  - Xe dynamics: thermal spectrum, same as PWR (graphite moderator)
  - Graphite thermal mass introduces temperature lag: T_fuel(t) approaches T_ss(PLR)
    with time constant tau ~ graphite_mass / rated_power
  - After step-down, fuel temperature is still high → large negative temp reactivity
    (reactor self-limits; provides passive safety)
  - Strongly negative temperature coefficient (-3.5 pcm/K) → large positive reactivity
    available at part load (temperature drops → reactivity rises)
  - Ramp rate 7%/min (intermediate between PWR 5%/min and MSR 10%/min)
  - Xenon overshoot occurs ~8-12h after power reduction (same as PWR)
  - Graphite inertia slows transient but does not eliminate xenon overshoot
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
    for k in ["power_output_mw", "xenon_concentration_rel", "fuel_temp_C",
              "temperature_reactivity_pcm", "available_reactivity_pcm",
              "ramp_rate_limit_pct_min", "can_restart"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC121"
    assert info["fidelity"] == "F1b"


def test_rated_power_output(model):
    """At full power: P_e = 250 * 1.0 * 0.42 = 105 MW."""
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 48.0,
                       "previous_power_fraction": 1.0})
    expected = 250.0 * 0.42
    assert abs(r["power_output_mw"] - expected) < 1.0, \
        f"Expected {expected:.0f} MW, got {r['power_output_mw']:.1f}"


def test_power_scales_with_fraction(model):
    r_full = model.predict({"power_fraction": 1.0,  "time_at_power_hours": 48.0,
                            "previous_power_fraction": 1.0})
    r_half = model.predict({"power_fraction": 0.5,  "time_at_power_hours": 48.0,
                            "previous_power_fraction": 0.5})
    assert abs(r_half["power_output_mw"] - r_full["power_output_mw"] / 2) < 1.0


def test_graphite_thermal_inertia(model):
    """Immediately after step-down (t=0), fuel temp still at previous steady state."""
    m = model._model
    T_after_0h = m.fuel_temp_transient_C(1.0, 0.5, 0.0)   # t=0: T at rated
    T_ss_rated = float(m.fuel_temp_ss_C(1.0))
    assert abs(T_after_0h - T_ss_rated) < 5.0, \
        f"At t=0 after step-down, T_fuel should still be ~{T_ss_rated:.0f}C, got {T_after_0h:.0f}C"


def test_fuel_temp_approaches_ss_after_long_time(model):
    """After 24h at new power level, fuel temperature should be near steady-state."""
    m = model._model
    T_24h = m.fuel_temp_transient_C(1.0, 0.5, 24.0)
    T_ss_half = float(m.fuel_temp_ss_C(0.5))
    assert abs(T_24h - T_ss_half) < 20.0, \
        f"After 24h, T_fuel ({T_24h:.0f}C) should be near T_ss(0.5) ({T_ss_half:.0f}C)"


def test_temp_coeff_negative(model):
    """Temperature coefficient must be negative (TRISO fuel safety feature)."""
    assert model._model.temp_coeff < 0, \
        "HTGR temperature coefficient must be negative (strongly negative per TRISO physics)"


def test_temperature_feedback_positive_at_part_load(model):
    """At part load, temperature drops → negative coeff * negative dT → positive reactivity."""
    m = model._model
    T_rated = float(m.fuel_temp_ss_C(1.0))
    T_part  = float(m.fuel_temp_ss_C(0.5))
    rho_rated = m.temperature_reactivity_pcm(T_rated)
    rho_part  = m.temperature_reactivity_pcm(T_part)
    assert rho_part > rho_rated, \
        f"At lower temp ({T_part:.0f}C), rho_T ({rho_part:.0f} pcm) should be > rated ({rho_rated:.0f} pcm)"


def test_can_restart_at_equilibrium(model):
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 48.0,
                       "previous_power_fraction": 1.0})
    assert r["can_restart"] is True
    assert r["available_reactivity_pcm"] > 0


def test_xenon_overshoot_after_power_reduction(model):
    """Xe peak above new equilibrium 8-12h after power reduction (thermal reactor xenon physics)."""
    m = model._model
    Xe_10h = m.xenon_transient(1.0, 0.5, 10.0)
    Xe_eq_50 = float(m.equilibrium_xenon(0.5))
    assert Xe_10h > Xe_eq_50, \
        f"Xe at 10h ({Xe_10h:.3f}) should be above eq at 50% ({Xe_eq_50:.3f})"


def test_xenon_returns_to_equilibrium(model):
    m = model._model
    Xe_72h = m.xenon_transient(1.0, 0.5, 72.0)
    Xe_eq_50 = float(m.equilibrium_xenon(0.5))
    assert abs(Xe_72h - Xe_eq_50) < 0.05, \
        f"Xe at 72h ({Xe_72h:.3f}) should be near eq ({Xe_eq_50:.3f})"


def test_ramp_rate_constraint(model):
    """7%/min: 50% step in 1 min → 57%."""
    m = model._model
    P_achievable, limited = m.ramp_rate_limit(0.5, 1.0, 1.0)
    assert limited is True
    assert abs(P_achievable - 0.57) < 0.001, \
        f"7%/min in 1 min from 0.5 should give 0.57, got {P_achievable:.4f}"


def test_ramp_rate_value(model):
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 0.0,
                       "previous_power_fraction": 1.0})
    assert r["ramp_rate_limit_pct_min"] == pytest.approx(7.0)


def test_xenon_nonnegative(model):
    m = model._model
    for t in np.linspace(0, 72, 50):
        Xe = m.xenon_transient(1.0, 0.4, t)
        assert Xe >= 0.0


def test_helium_outlet_temp_physical(model):
    """He outlet at rated must be in HTGR range (700-1000C)."""
    T_out = model._model.T_He_outlet
    assert 700 <= T_out <= 1000, f"He outlet {T_out}C outside HTGR range [700, 1000]C"


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({"power_fraction": 0.7, "time_at_power_hours": 5.0,
                       "previous_power_fraction": 1.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
