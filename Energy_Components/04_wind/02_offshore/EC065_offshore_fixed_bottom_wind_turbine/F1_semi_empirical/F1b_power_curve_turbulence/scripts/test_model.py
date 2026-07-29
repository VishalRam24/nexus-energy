"""EC065 — Offshore Wind — F1b Turbulence — Test Suite"""

import sys, time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_returns_dict(model):
    r = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.08})
    for key in ["power_kw", "power_coefficient", "air_density_corrected"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC065"
    assert info["fidelity"] == "F1b"


def test_zero_power_below_cut_in(model):
    r = model.predict({"wind_speed_m_s": 2.0, "turbulence_intensity": 0.08})
    assert float(r["power_kw"]) == 0.0


def test_rated_power(model):
    """At well above rated, should reach rated power."""
    r = model.predict({"wind_speed_m_s": 16.0, "turbulence_intensity": 0.0})
    assert float(r["power_kw"]) >= 7800.0


def test_zero_power_above_cut_out(model):
    r = model.predict({"wind_speed_m_s": 26.0, "turbulence_intensity": 0.08})
    assert float(r["power_kw"]) == 0.0


def test_power_increases_with_wind(model):
    v = np.array([4.0, 6.0, 8.0, 10.0])
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.08})
    assert np.all(np.diff(r["power_kw"]) > 0)


def test_power_capped_at_rated(model):
    v = np.linspace(3, 25, 100)
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.10})
    assert np.all(r["power_kw"] <= 8000.0)


def test_humid_air_density_reasonable(model):
    """At standard conditions (15C, 50% RH), density should be near 1.225."""
    r = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.08,
                        "air_temperature_degC": 15.0, "relative_humidity": 0.5})
    rho = float(r["air_density_corrected"])
    assert 1.15 < rho < 1.30, f"rho = {rho:.4f}"


def test_humidity_reduces_density(model):
    """Higher humidity -> lower air density (moist air is lighter)."""
    r_dry = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.08,
                            "air_temperature_degC": 25.0, "relative_humidity": 0.0})
    r_wet = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.08,
                            "air_temperature_degC": 25.0, "relative_humidity": 1.0})
    assert float(r_dry["air_density_corrected"]) > float(r_wet["air_density_corrected"])


def test_cold_air_denser(model):
    """Cold air should be denser than warm air."""
    r_cold = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.08,
                             "air_temperature_degC": 0.0, "relative_humidity": 0.5})
    r_warm = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.08,
                             "air_temperature_degC": 30.0, "relative_humidity": 0.5})
    assert float(r_cold["air_density_corrected"]) > float(r_warm["air_density_corrected"])


def test_cold_air_more_power(model):
    """Cold, dense air should produce more power."""
    r_cold = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.08,
                             "air_temperature_degC": 0.0, "relative_humidity": 0.3})
    r_warm = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.08,
                             "air_temperature_degC": 30.0, "relative_humidity": 0.8})
    assert float(r_cold["power_kw"]) > float(r_warm["power_kw"])


def test_turbulence_effect(model):
    """Turbulence should change power at partial load."""
    r_low = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.0})
    r_high = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.15})
    assert float(r_low["power_kw"]) != pytest.approx(float(r_high["power_kw"]), abs=1.0)


def test_betz_limit(model):
    v = np.linspace(4, 25, 100)
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.08})
    assert np.all(r["power_coefficient"] < 0.60)


def test_array_inputs(model):
    v = np.array([5.0, 10.0, 15.0])
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.08})
    assert r["power_kw"].shape == (3,)


def test_benchmark(model):
    v = np.random.uniform(3, 25, 1000)
    ti = np.random.uniform(0.04, 0.12, 1000)
    start = time.perf_counter()
    model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
