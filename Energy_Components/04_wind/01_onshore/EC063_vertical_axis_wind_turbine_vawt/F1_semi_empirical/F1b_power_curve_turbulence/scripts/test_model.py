"""EC063 — VAWT — F1b Turbulence + Air Density — Test Suite

Tests are designed to FAIL the model, not accommodate it.
Loosening of tolerances requires a # RATIONALE: comment.
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
    r = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.10})
    for key in ["power_kw", "power_coefficient", "capacity_factor",
                "air_density", "ti_modifier"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC063"
    assert info["fidelity"] == "F1b"


# --- Cut-in / cut-out ---

def test_zero_power_below_cut_in(model):
    r = model.predict({"wind_speed_m_s": 2.0, "turbulence_intensity": 0.10})
    assert float(r["power_kw"]) == 0.0


def test_zero_power_above_cut_out(model):
    r = model.predict({"wind_speed_m_s": 23.0, "turbulence_intensity": 0.10})
    assert float(r["power_kw"]) == 0.0


def test_rated_power_reached(model):
    """At rated speed with zero TI, power should reach rated output."""
    r = model.predict({"wind_speed_m_s": 11.0, "turbulence_intensity": 0.0})
    assert float(r["power_kw"]) >= 9.5, (
        f"Expected ~10 kW at rated speed with zero TI, got {float(r['power_kw']):.2f}"
    )


def test_power_never_exceeds_rated(model):
    v = np.linspace(3, 22, 100)
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.20})
    assert np.all(r["power_kw"] <= 10.0 + 1e-9)


# --- Monotonicity in operating range ---

def test_power_increases_with_wind_in_partial_load(model):
    v = np.array([3.5, 5.0, 7.0, 9.0])
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.05})
    assert np.all(np.diff(r["power_kw"]) > 0)


# --- Air density physics ---

def test_standard_density_at_sea_level_15c(model):
    """rho at 15 degC, sea level should be ~1.225 kg/m3."""
    r = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.0,
                       "air_temperature_degC": 15.0, "altitude_m": 0.0})
    rho = float(r["air_density"])
    assert 1.21 < rho < 1.24, f"rho = {rho:.4f}, expected ~1.225"


def test_cold_air_denser_than_warm(model):
    r_cold = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.0,
                             "air_temperature_degC": -10.0, "altitude_m": 0.0})
    r_warm = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.0,
                             "air_temperature_degC": 35.0, "altitude_m": 0.0})
    assert float(r_cold["air_density"]) > float(r_warm["air_density"])


def test_higher_altitude_lower_density(model):
    r_low = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.0,
                            "air_temperature_degC": 15.0, "altitude_m": 0.0})
    r_high = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.0,
                             "air_temperature_degC": 15.0, "altitude_m": 2000.0})
    assert float(r_low["air_density"]) > float(r_high["air_density"])


def test_power_drops_with_altitude(model):
    """Higher altitude means lower air density and lower power."""
    r_low = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.0,
                            "air_temperature_degC": 15.0, "altitude_m": 0.0})
    r_high = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.0,
                             "air_temperature_degC": 15.0, "altitude_m": 2000.0})
    assert float(r_low["power_kw"]) > float(r_high["power_kw"])


def test_cold_dense_air_more_power(model):
    """Cold air at sea level should give more power than warm air at altitude."""
    r_cold_sea = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.0,
                                 "air_temperature_degC": 0.0, "altitude_m": 0.0})
    r_warm_alt = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.0,
                                 "air_temperature_degC": 30.0, "altitude_m": 1500.0})
    assert float(r_cold_sea["power_kw"]) > float(r_warm_alt["power_kw"])


# --- TI physics (critical: VAWT power drops with TI) ---

def test_power_drops_with_turbulence_at_partial_load(model):
    """Higher TI must reduce power at partial load (below rated speed)."""
    v_partial = 7.0  # well below rated 11 m/s
    r_low_ti = model.predict({"wind_speed_m_s": v_partial, "turbulence_intensity": 0.05})
    r_high_ti = model.predict({"wind_speed_m_s": v_partial, "turbulence_intensity": 0.30})
    assert float(r_low_ti["power_kw"]) > float(r_high_ti["power_kw"]), (
        "VAWT power must decrease with TI at partial load"
    )


def test_ti_modifier_is_less_than_one_at_partial_load(model):
    """TI modifier < 1 when TI > 0 and below rated speed."""
    r = model.predict({"wind_speed_m_s": 7.0, "turbulence_intensity": 0.15})
    assert float(r["ti_modifier"]) < 1.0


def test_ti_modifier_is_one_at_rated_speed(model):
    """At rated speed, TI penalty should not apply (turbine is regulated)."""
    r = model.predict({"wind_speed_m_s": 11.0, "turbulence_intensity": 0.30})
    assert float(r["ti_modifier"]) == pytest.approx(1.0, abs=1e-9)


def test_ti_modifier_is_one_at_zero_ti(model):
    r = model.predict({"wind_speed_m_s": 7.0, "turbulence_intensity": 0.0})
    assert float(r["ti_modifier"]) == pytest.approx(1.0, abs=1e-9)


def test_power_monotonically_decreases_with_ti(model):
    """Power at 7 m/s should strictly decrease as TI increases from 0 to 0.40."""
    v_partial = 7.0
    ti_vals = np.linspace(0.0, 0.40, 10)
    powers = []
    for ti in ti_vals:
        r = model.predict({"wind_speed_m_s": v_partial, "turbulence_intensity": ti})
        powers.append(float(r["power_kw"]))
    assert np.all(np.diff(powers) < 0), "Power should strictly decrease with TI at partial load"


# --- Betz limit ---

def test_cp_below_betz_limit(model):
    v = np.linspace(4, 22, 100)
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.0})
    assert np.all(r["power_coefficient"] < 0.60)


def test_vawt_cp_below_hawt_max(model):
    """VAWT max Cp should stay below HAWT theoretical max (~0.45 typical)."""
    v = np.linspace(4, 12, 50)
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.0})
    assert np.max(r["power_coefficient"]) < 0.45


# --- Array inputs ---

def test_array_inputs(model):
    v = np.array([4.0, 7.0, 10.0, 14.0])
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.10})
    assert r["power_kw"].shape == (4,)


# --- Benchmark ---

def test_benchmark(model):
    v = np.random.uniform(2, 23, 1000)
    ti = np.random.uniform(0.0, 0.30, 1000)
    start = time.perf_counter()
    model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
