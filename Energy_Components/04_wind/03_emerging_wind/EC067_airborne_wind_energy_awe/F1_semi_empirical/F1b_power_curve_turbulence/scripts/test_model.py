"""EC067 — Airborne Wind Energy (AWE) — F1b Turbulence + Altitude Density — Test Suite"""

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
    r = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.08})
    for key in ["power_kw", "capacity_factor", "air_density", "loyd_limit_kw", "turbulence_correction"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC067"
    assert info["fidelity"] == "F1b"


# --- Cut-in / cut-out ---

def test_zero_power_below_cut_in(model):
    r = model.predict({"wind_speed_m_s": 2.0, "turbulence_intensity": 0.10})
    assert float(r["power_kw"]) == 0.0


def test_zero_power_above_cut_out(model):
    r = model.predict({"wind_speed_m_s": 27.0, "turbulence_intensity": 0.10})
    assert float(r["power_kw"]) == 0.0


def test_rated_power_reached(model):
    """Above rated speed at standard atmosphere, should reach rated."""
    r = model.predict({"wind_speed_m_s": 12.0, "turbulence_intensity": 0.0})
    assert float(r["power_kw"]) >= 95.0, (
        f"Expected close to 100 kW at rated, got {float(r['power_kw']):.2f}"
    )


def test_power_never_exceeds_rated(model):
    v = np.linspace(0, 26, 200)
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.15})
    assert np.all(r["power_kw"] <= 100.0 + 1e-6)


# --- Monotonicity in partial load ---

def test_power_increases_with_wind_in_partial_load(model):
    v = np.array([4.5, 6.0, 7.5, 9.0])
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.08})
    assert np.all(np.diff(r["power_kw"]) > 0)


# --- Turbulence correction ---

def test_turbulence_changes_power(model):
    r_lo = model.predict({"wind_speed_m_s": 7.0, "turbulence_intensity": 0.0})
    r_hi = model.predict({"wind_speed_m_s": 7.0, "turbulence_intensity": 0.15})
    assert float(r_lo["power_kw"]) != pytest.approx(float(r_hi["power_kw"]), abs=0.1)


def test_zero_turbulence_matches_base_curve(model):
    v = np.array([5.0, 7.0, 9.0, 11.0])
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.0})
    p_base = model._model.power_curve(v)
    # At standard rho_ref density, should match base spline
    np.testing.assert_allclose(r["power_kw"], p_base, rtol=0.02)


# --- Altitude physics ---

def test_altitude_reduces_density(model):
    r_low = model.predict({"wind_speed_m_s": 8.0, "altitude_m": 100.0})
    r_high = model.predict({"wind_speed_m_s": 8.0, "altitude_m": 700.0})
    assert float(r_low["air_density"]) > float(r_high["air_density"])


def test_altitude_reduces_power(model):
    """Higher altitude -> lower air density -> lower power."""
    r_low = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.08,
                            "altitude_m": 100.0})
    r_high = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.08,
                             "altitude_m": 700.0})
    assert float(r_low["power_kw"]) > float(r_high["power_kw"])


def test_density_at_400m_isa(model):
    """ISA density at 400 m should be ~1.179 kg/m3 (about 3.8% below sea level)."""
    r = model.predict({"wind_speed_m_s": 8.0, "altitude_m": 400.0,
                       "air_temperature_degC": 15.0, "relative_humidity": 0.0})
    rho = float(r["air_density"])
    # ISA at 400 m: T = 288.15 - 0.0065*400 = 285.55 K; P ~ 96717 Pa; rho ~ 1.179
    assert 1.15 < rho < 1.21, f"Expected ~1.179, got {rho:.4f}"


def test_humidity_reduces_density(model):
    r_dry = model.predict({"wind_speed_m_s": 8.0, "altitude_m": 400.0,
                           "air_temperature_degC": 20.0, "relative_humidity": 0.0})
    r_wet = model.predict({"wind_speed_m_s": 8.0, "altitude_m": 400.0,
                           "air_temperature_degC": 20.0, "relative_humidity": 1.0})
    assert float(r_dry["air_density"]) > float(r_wet["air_density"])


# --- Loyd limit ---

def test_power_below_loyd_limit(model):
    """AWE power must never exceed Loyd theoretical limit."""
    v = np.linspace(4, 25, 100)
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.0,
                       "altitude_m": 400.0})
    # Exclude cut-in/cut-out zeros from check
    active = v >= 4.0
    assert np.all(r["power_kw"][active] <= r["loyd_limit_kw"][active] + 1e-3)


# --- Array inputs ---

def test_array_inputs(model):
    v = np.array([5.0, 7.0, 9.0, 11.0])
    ti = np.array([0.05, 0.08, 0.10, 0.12])
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
    assert r["power_kw"].shape == (4,)


# --- Benchmark ---

def test_benchmark(model):
    v = np.random.uniform(4, 25, 1000)
    ti = np.random.uniform(0.03, 0.15, 1000)
    start = time.perf_counter()
    model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti,
                   "altitude_m": 400.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
