"""EC062 — HAWT Onshore — F1b Turbulence — Test Suite"""

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
    r = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.10})
    for key in ["power_kw", "power_coefficient", "capacity_factor_correction"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC062"
    assert info["fidelity"] == "F1b"


def test_zero_power_below_cut_in(model):
    r = model.predict({"wind_speed_m_s": 2.0, "turbulence_intensity": 0.15})
    assert float(r["power_kw"]) == 0.0


def test_rated_power_at_high_wind(model):
    """At well above rated speed, power should be at or near rated."""
    r = model.predict({"wind_speed_m_s": 15.0, "turbulence_intensity": 0.0})
    assert float(r["power_kw"]) >= 2900.0


def test_zero_power_above_cut_out(model):
    r = model.predict({"wind_speed_m_s": 26.0, "turbulence_intensity": 0.15})
    assert float(r["power_kw"]) == 0.0


def test_power_increases_with_wind(model):
    """In below-rated region, power should increase."""
    v = np.array([4.0, 6.0, 8.0, 10.0])
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.10})
    assert np.all(np.diff(r["power_kw"]) > 0)


def test_turbulence_changes_power(model):
    """Turbulence should modify power output."""
    r_low = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.0})
    r_high = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.20})
    # Not necessarily higher or lower — depends on d2P/dV2 sign
    assert float(r_low["power_kw"]) != pytest.approx(float(r_high["power_kw"]), abs=0.1)


def test_zero_turbulence_matches_base_curve(model):
    """With TI=0, should match base power curve."""
    v = np.array([5.0, 8.0, 12.0, 15.0])
    r_f1b = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.0})
    # Power should be similar to spline values
    p_base = model._model.power_curve(v)
    np.testing.assert_allclose(r_f1b["power_kw"], p_base, rtol=0.01)


def test_capacity_factor_correction_nonzero(model):
    """CF correction should be non-zero when TI > 0."""
    r = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.15})
    assert float(r["capacity_factor_correction"]) != 0.0


def test_betz_limit(model):
    """Cp should not exceed Betz limit."""
    v = np.linspace(4, 25, 100)
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.10})
    assert np.all(r["power_coefficient"] < 0.60)


def test_power_clipped_to_rated(model):
    """Power should never exceed rated."""
    v = np.linspace(3, 25, 100)
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.25})
    assert np.all(r["power_kw"] <= 3000.0)


def test_array_inputs(model):
    v = np.array([5.0, 10.0, 15.0])
    ti = np.array([0.10, 0.15, 0.20])
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
    assert r["power_kw"].shape == (3,)


def test_benchmark(model):
    v = np.random.uniform(3, 25, 1000)
    ti = np.random.uniform(0.05, 0.25, 1000)
    start = time.perf_counter()
    model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
