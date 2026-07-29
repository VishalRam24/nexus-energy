"""EC066 — Offshore Floating Wind — F1b Turbulence + Pitch — Test Suite

Tests must fail the model, not accommodate it.
Loosening requires # RATIONALE: comment.
"""

import sys
import time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_returns_required_keys(model):
    r = model.predict({"wind_speed_m_s": 10.0})
    for key in ["power_kw", "power_coefficient", "air_density", "pitch_factor"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC066"
    assert info["fidelity"] == "F1b"


# --- Cut-in / cut-out ---

def test_zero_power_below_cut_in(model):
    r = model.predict({"wind_speed_m_s": 2.0})
    assert float(r["power_kw"]) == 0.0


def test_zero_power_above_cut_out(model):
    r = model.predict({"wind_speed_m_s": 26.0})
    assert float(r["power_kw"]) == 0.0


def test_rated_power_reached(model):
    """At 15 m/s with zero TI and pitch=0, should reach rated output."""
    r = model.predict({"wind_speed_m_s": 15.0, "turbulence_intensity": 0.0,
                       "platform_pitch_deg": 0.0})
    assert float(r["power_kw"]) >= 14500.0, (
        f"Expected close to 15000 kW at rated, got {float(r['power_kw']):.1f}"
    )


def test_power_never_exceeds_rated(model):
    v = np.linspace(3, 25, 100)
    r = model.predict({"wind_speed_m_s": v})
    assert np.all(r["power_kw"] <= 15000.0 + 1e-6)


# --- Monotonicity in partial load ---

def test_power_increases_with_wind_in_partial_load(model):
    v = np.array([4.0, 6.0, 8.0, 10.0])
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.07,
                       "platform_pitch_deg": 0.0})
    assert np.all(np.diff(r["power_kw"]) > 0)


# --- Air density physics ---

def test_standard_density_approximate(model):
    """At 12 degC, 80% RH the density should be in realistic marine range."""
    r = model.predict({"wind_speed_m_s": 10.0, "air_temperature_degC": 12.0,
                       "relative_humidity": 0.80})
    rho = float(r["air_density"])
    assert 1.15 < rho < 1.30, f"rho = {rho:.4f}"


def test_humidity_reduces_density(model):
    r_dry = model.predict({"wind_speed_m_s": 10.0, "air_temperature_degC": 20.0,
                            "relative_humidity": 0.0})
    r_wet = model.predict({"wind_speed_m_s": 10.0, "air_temperature_degC": 20.0,
                            "relative_humidity": 1.0})
    assert float(r_dry["air_density"]) > float(r_wet["air_density"])


def test_cold_air_denser(model):
    r_cold = model.predict({"wind_speed_m_s": 10.0, "air_temperature_degC": 0.0,
                             "relative_humidity": 0.5})
    r_warm = model.predict({"wind_speed_m_s": 10.0, "air_temperature_degC": 30.0,
                             "relative_humidity": 0.5})
    assert float(r_cold["air_density"]) > float(r_warm["air_density"])


def test_denser_air_more_power(model):
    r_cold = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.07,
                             "platform_pitch_deg": 0.0,
                             "air_temperature_degC": 0.0, "relative_humidity": 0.3})
    r_warm = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.07,
                             "platform_pitch_deg": 0.0,
                             "air_temperature_degC": 35.0, "relative_humidity": 0.9})
    assert float(r_cold["power_kw"]) > float(r_warm["power_kw"])


# --- Platform pitch penalty ---

def test_pitch_factor_is_one_at_zero_pitch(model):
    r = model.predict({"wind_speed_m_s": 10.0, "platform_pitch_deg": 0.0})
    assert float(r["pitch_factor"]) == pytest.approx(1.0, abs=1e-9)


def test_pitch_factor_less_than_one_at_nonzero_pitch(model):
    r = model.predict({"wind_speed_m_s": 10.0, "platform_pitch_deg": 5.0})
    assert float(r["pitch_factor"]) < 1.0


def test_pitch_factor_decreases_with_pitch(model):
    pitches = np.array([0.0, 2.0, 5.0, 8.0, 10.0])
    factors = []
    for p in pitches:
        r = model.predict({"wind_speed_m_s": 10.0, "platform_pitch_deg": p})
        factors.append(float(r["pitch_factor"]))
    assert np.all(np.diff(factors) < 0), "Pitch factor must decrease as pitch increases"


def test_power_drops_with_platform_pitch(model):
    """Higher wave-induced pitch must reduce power output."""
    r_upright = model.predict({"wind_speed_m_s": 10.0, "platform_pitch_deg": 0.0})
    r_pitched = model.predict({"wind_speed_m_s": 10.0, "platform_pitch_deg": 8.0})
    assert float(r_upright["power_kw"]) > float(r_pitched["power_kw"]), (
        "Power must drop when platform is pitched"
    )


def test_pitch_10deg_cos2_loss(model):
    """cos^2(10 deg) = 0.9698, so pitch factor should match this."""
    r = model.predict({"wind_speed_m_s": 10.0, "platform_pitch_deg": 10.0})
    expected = np.cos(np.deg2rad(10.0)) ** 2
    assert float(r["pitch_factor"]) == pytest.approx(expected, rel=1e-5)


# --- Turbulence ---

def test_turbulence_changes_power_at_partial_load(model):
    """TI should noticeably change power at partial load."""
    r_lo = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.0,
                           "platform_pitch_deg": 0.0})
    r_hi = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.15,
                           "platform_pitch_deg": 0.0})
    assert float(r_lo["power_kw"]) != pytest.approx(float(r_hi["power_kw"]), abs=10.0)


def test_offshore_ti_lower_than_onshore_effect(model):
    """Compare power at TI=0.07 (offshore) vs TI=0.15 (onshore) at partial load."""
    r_offshore = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.07})
    r_onshore_ti = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.15})
    # Both should be valid floats; difference confirms TI has effect
    diff = abs(float(r_offshore["power_kw"]) - float(r_onshore_ti["power_kw"]))
    assert diff > 0.0, "TI=0.07 and TI=0.15 should produce different power at partial load"


# --- Betz limit ---

def test_cp_below_betz_limit(model):
    v = np.linspace(4, 25, 100)
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.07})
    assert np.all(r["power_coefficient"] < 0.60)


# --- Array inputs ---

def test_array_inputs(model):
    v = np.array([5.0, 10.0, 15.0, 20.0])
    r = model.predict({"wind_speed_m_s": v})
    assert r["power_kw"].shape == (4,)


# --- Benchmark ---

def test_benchmark(model):
    v = np.random.uniform(3, 25, 1000)
    ti = np.random.uniform(0.04, 0.12, 1000)
    pitch = np.random.uniform(0.0, 6.0, 1000)
    start = time.perf_counter()
    model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti,
                   "platform_pitch_deg": pitch})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
