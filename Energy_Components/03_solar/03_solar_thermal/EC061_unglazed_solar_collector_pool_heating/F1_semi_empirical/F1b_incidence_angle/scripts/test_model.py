"""EC061 — Unglazed Solar Collector (Pool Heating) — F1b IAM — Test Suite"""

import sys, time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_returns_all_keys(model):
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 30.0,
                        "T_inlet_degC": 26.0, "T_ambient_degC": 22.0})
    for key in ["useful_heat_w", "efficiency", "iam_factor",
                 "T_outlet_degC", "U_L_effective", "Q_sky_loss_w"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC061"
    assert info["fidelity"] == "F1b"


# --- IAM physics ---

def test_iam_unity_at_normal(model):
    """IAM = 1 at normal incidence."""
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
                        "T_inlet_degC": 26.0, "T_ambient_degC": 22.0})
    assert float(r["iam_factor"]) == pytest.approx(1.0, abs=0.005)


def test_iam_decreases_with_angle(model):
    theta = np.array([0.0, 20.0, 40.0, 60.0])
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": theta,
                        "T_inlet_degC": 26.0, "T_ambient_degC": 22.0})
    assert np.all(np.diff(r["iam_factor"]) < 0)


def test_iam_between_0_and_1(model):
    theta = np.linspace(0, 79, 50)
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": theta,
                        "T_inlet_degC": 26.0, "T_ambient_degC": 22.0})
    assert np.all(r["iam_factor"] >= 0)
    assert np.all(r["iam_factor"] <= 1)


# --- Wind correction ---

def test_u_L_increases_with_wind(model):
    """U_L should increase with wind speed."""
    r_still = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 20.0,
                              "T_inlet_degC": 26.0, "T_ambient_degC": 22.0, "v_wind_m_s": 0.0})
    r_windy = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 20.0,
                              "T_inlet_degC": 26.0, "T_ambient_degC": 22.0, "v_wind_m_s": 5.0})
    assert float(r_windy["U_L_effective"]) > float(r_still["U_L_effective"])


def test_wind_reduces_efficiency(model):
    """Higher wind speed -> more heat loss -> lower efficiency."""
    r_still = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 20.0,
                              "T_inlet_degC": 26.0, "T_ambient_degC": 22.0, "v_wind_m_s": 0.0})
    r_windy = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 20.0,
                              "T_inlet_degC": 26.0, "T_ambient_degC": 22.0, "v_wind_m_s": 8.0})
    assert float(r_still["efficiency"]) > float(r_windy["efficiency"])


# --- Useful heat physics ---

def test_useful_heat_positive(model):
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
                        "T_inlet_degC": 25.0, "T_ambient_degC": 22.0})
    assert float(r["useful_heat_w"]) > 0


def test_no_heat_at_zero_irradiance(model):
    r = model.predict({"irradiance_w_m2": 0.0, "incidence_angle_deg": 0.0,
                        "T_inlet_degC": 25.0, "T_ambient_degC": 22.0})
    assert float(r["useful_heat_w"]) == 0.0


def test_heat_increases_with_irradiance(model):
    G = np.array([200.0, 400.0, 600.0, 800.0])
    r = model.predict({"irradiance_w_m2": G, "incidence_angle_deg": 20.0,
                        "T_inlet_degC": 25.0, "T_ambient_degC": 22.0})
    assert np.all(np.diff(r["useful_heat_w"]) > 0)


def test_heat_reduced_at_high_angle(model):
    r_normal = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
                               "T_inlet_degC": 25.0, "T_ambient_degC": 22.0})
    r_angled = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 50.0,
                               "T_inlet_degC": 25.0, "T_ambient_degC": 22.0})
    assert float(r_normal["useful_heat_w"]) > float(r_angled["useful_heat_w"])


# --- Sky radiation loss ---

def test_sky_loss_positive(model):
    """Unglazed collector radiates to cold sky — loss is always > 0."""
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 20.0,
                        "T_inlet_degC": 26.0, "T_ambient_degC": 22.0})
    assert float(r["Q_sky_loss_w"]) > 0


# --- Outlet temperature ---

def test_outlet_above_inlet_when_collecting(model):
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
                        "T_inlet_degC": 25.0, "T_ambient_degC": 22.0})
    assert float(r["T_outlet_degC"]) >= 25.0


# --- Efficiency ---

def test_efficiency_range_for_pool_heating(model):
    """Unglazed collectors at pool temps (25-30C, small ΔT): eta ~ 40-80%."""
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
                        "T_inlet_degC": 25.0, "T_ambient_degC": 22.0, "v_wind_m_s": 2.0})
    eta = float(r["efficiency"])
    assert 0.20 < eta < 0.90, f"Efficiency = {eta:.4f}"


# --- Array inputs ---

def test_array_inputs(model):
    G = np.array([400.0, 700.0, 900.0])
    r = model.predict({"irradiance_w_m2": G, "incidence_angle_deg": 20.0,
                        "T_inlet_degC": 26.0, "T_ambient_degC": 22.0})
    assert r["useful_heat_w"].shape == (3,)


# --- Benchmark ---

def test_benchmark(model):
    G = np.random.uniform(200, 1000, 1000)
    theta = np.random.uniform(0, 70, 1000)
    v = np.random.uniform(0, 8, 1000)
    start = time.perf_counter()
    model.predict({"irradiance_w_m2": G, "incidence_angle_deg": theta,
                    "T_inlet_degC": 26.0, "T_ambient_degC": 22.0, "v_wind_m_s": v})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
