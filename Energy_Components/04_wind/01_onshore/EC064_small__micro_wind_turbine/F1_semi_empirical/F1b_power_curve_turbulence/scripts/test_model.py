"""EC064 — Small/Micro Wind Turbine — F1b Turbulence + Air Density — Test Suite"""

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
    r = model.predict({"wind_speed_m_s": 7.0, "turbulence_intensity": 0.15})
    for key in ["power_kw", "power_coefficient", "air_density", "turbulence_correction"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC064"
    assert info["fidelity"] == "F1b"


# --- Cut-in / cut-out ---

def test_zero_power_below_cut_in(model):
    r = model.predict({"wind_speed_m_s": 1.0, "turbulence_intensity": 0.20})
    assert float(r["power_kw"]) == 0.0


def test_zero_power_above_cut_out(model):
    r = model.predict({"wind_speed_m_s": 22.0, "turbulence_intensity": 0.20})
    assert float(r["power_kw"]) == 0.0


def test_rated_power_reached(model):
    """Well above rated wind speed, turbine should be at rated output."""
    r = model.predict({"wind_speed_m_s": 14.0, "turbulence_intensity": 0.0})
    assert float(r["power_kw"]) >= 9.5, (
        f"Expected close to 10 kW at rated, got {float(r['power_kw']):.3f}"
    )


def test_power_never_exceeds_rated(model):
    v = np.linspace(0, 22, 200)
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.30})
    assert np.all(r["power_kw"] <= 10.0 + 1e-6)


# --- Monotonicity ---

def test_power_increases_with_wind_in_partial_load(model):
    v = np.array([3.0, 5.0, 7.0, 9.0])
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.10})
    assert np.all(np.diff(r["power_kw"]) > 0)


# --- Turbulence correction ---

def test_turbulence_changes_power(model):
    """TI correction must change power output at partial load."""
    r_lo = model.predict({"wind_speed_m_s": 7.0, "turbulence_intensity": 0.0})
    r_hi = model.predict({"wind_speed_m_s": 7.0, "turbulence_intensity": 0.30})
    assert float(r_lo["power_kw"]) != pytest.approx(float(r_hi["power_kw"]), abs=0.01)


def test_zero_turbulence_matches_base_curve(model):
    """With TI=0 at standard atmosphere, power should match base spline."""
    v = np.array([4.0, 7.0, 10.0, 12.0])
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.0})
    p_base = model._model.power_curve(v)
    np.testing.assert_allclose(r["power_kw"], p_base, rtol=0.01)


# --- Air density physics ---

def test_standard_density_at_sea_level(model):
    """Sea level 15degC dry air should give ~1.225 kg/m3."""
    r = model.predict({"wind_speed_m_s": 7.0, "pressure_pa": 101325.0,
                       "air_temperature_degC": 15.0, "relative_humidity": 0.0})
    rho = float(r["air_density"])
    assert abs(rho - 1.225) < 0.01, f"Expected ~1.225, got {rho:.4f}"


def test_lower_pressure_reduces_density(model):
    r_sl = model.predict({"wind_speed_m_s": 7.0, "pressure_pa": 101325.0,
                          "air_temperature_degC": 15.0})
    r_alt = model.predict({"wind_speed_m_s": 7.0, "pressure_pa": 84560.0,
                           "air_temperature_degC": 15.0})
    assert float(r_sl["air_density"]) > float(r_alt["air_density"])


def test_higher_temperature_reduces_density(model):
    r_cold = model.predict({"wind_speed_m_s": 7.0, "air_temperature_degC": 5.0})
    r_hot = model.predict({"wind_speed_m_s": 7.0, "air_temperature_degC": 35.0})
    assert float(r_cold["air_density"]) > float(r_hot["air_density"])


def test_humidity_reduces_density(model):
    r_dry = model.predict({"wind_speed_m_s": 7.0, "air_temperature_degC": 20.0,
                           "relative_humidity": 0.0})
    r_wet = model.predict({"wind_speed_m_s": 7.0, "air_temperature_degC": 20.0,
                           "relative_humidity": 1.0})
    assert float(r_dry["air_density"]) > float(r_wet["air_density"])


def test_denser_air_more_power(model):
    """Cold dry air (dense) should produce more power than hot humid air."""
    r_cold = model.predict({"wind_speed_m_s": 7.0, "turbulence_intensity": 0.10,
                             "pressure_pa": 101325.0, "air_temperature_degC": 0.0,
                             "relative_humidity": 0.0})
    r_warm = model.predict({"wind_speed_m_s": 7.0, "turbulence_intensity": 0.10,
                             "pressure_pa": 101325.0, "air_temperature_degC": 35.0,
                             "relative_humidity": 0.9})
    assert float(r_cold["power_kw"]) > float(r_warm["power_kw"])


def test_altitude_reduces_power(model):
    """1500 m altitude site should produce less power than sea-level."""
    r_sl = model.predict({"wind_speed_m_s": 7.0, "turbulence_intensity": 0.15,
                          "pressure_pa": 101325.0, "air_temperature_degC": 15.0})
    r_alt = model.predict({"wind_speed_m_s": 7.0, "turbulence_intensity": 0.15,
                           "pressure_pa": 84560.0, "air_temperature_degC": 15.0})
    assert float(r_sl["power_kw"]) > float(r_alt["power_kw"])


# --- Betz limit ---

def test_cp_below_betz_limit(model):
    v = np.linspace(3, 20, 100)
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.15})
    assert np.all(r["power_coefficient"] < 0.60)


# --- Array inputs ---

def test_array_inputs(model):
    v = np.array([4.0, 7.0, 10.0, 14.0])
    ti = np.array([0.10, 0.15, 0.20, 0.25])
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
    assert r["power_kw"].shape == (4,)
    assert r["air_density"].shape == () or r["air_density"].ndim <= 1


# --- Benchmark ---

def test_benchmark(model):
    v = np.random.uniform(2.5, 20, 1000)
    ti = np.random.uniform(0.05, 0.35, 1000)
    start = time.perf_counter()
    model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
