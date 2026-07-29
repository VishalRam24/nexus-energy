"""EC060 — Solar Pond — F1b Incidence Angle Modifier — Test Suite"""

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
                        "T_lcz_degC": 80.0, "T_ambient_degC": 25.0})
    for key in ["useful_heat_w", "efficiency", "iam_factor", "T_extraction_degC"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC060"
    assert info["fidelity"] == "F1b"


# --- IAM physics (Snell/Fresnel) ---

def test_iam_unity_at_normal_incidence(model):
    """IAM = 1.0 by definition at theta=0 (ratio tau/tau_0 = 1)."""
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
                        "T_lcz_degC": 80.0, "T_ambient_degC": 25.0})
    assert float(r["iam_factor"]) == pytest.approx(1.0, abs=0.005)


def test_iam_decreases_with_angle(model):
    """IAM should decrease monotonically with incidence angle."""
    theta = np.array([0.0, 20.0, 40.0, 60.0, 75.0])
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": theta,
                        "T_lcz_degC": 80.0, "T_ambient_degC": 25.0})
    assert np.all(np.diff(r["iam_factor"]) < 0)


def test_iam_between_zero_and_one(model):
    theta = np.linspace(0, 79, 50)
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": theta,
                        "T_lcz_degC": 80.0, "T_ambient_degC": 25.0})
    assert np.all(r["iam_factor"] >= 0.0)
    assert np.all(r["iam_factor"] <= 1.0)


# --- Useful heat physics ---

def test_useful_heat_positive(model):
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 30.0,
                        "T_lcz_degC": 40.0, "T_ambient_degC": 20.0})
    assert float(r["useful_heat_w"]) > 0


def test_no_heat_at_zero_irradiance(model):
    r = model.predict({"irradiance_w_m2": 0.0, "incidence_angle_deg": 30.0,
                        "T_lcz_degC": 40.0, "T_ambient_degC": 20.0})
    assert float(r["useful_heat_w"]) == 0.0


def test_heat_increases_with_irradiance(model):
    G = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"irradiance_w_m2": G, "incidence_angle_deg": 20.0,
                        "T_lcz_degC": 40.0, "T_ambient_degC": 20.0})
    assert np.all(np.diff(r["useful_heat_w"]) > 0)


def test_heat_decreases_with_lcz_temperature(model):
    """Higher LCZ temp -> more losses -> less net useful heat."""
    T_lcz = np.array([40.0, 55.0, 70.0, 85.0])
    r = model.predict({"irradiance_w_m2": 700.0, "incidence_angle_deg": 20.0,
                        "T_lcz_degC": T_lcz, "T_ambient_degC": 20.0})
    # Heat should decrease with temperature
    assert r["useful_heat_w"][0] > r["useful_heat_w"][-1]


def test_heat_reduced_at_high_angle(model):
    """Higher incidence angle reduces useful heat via IAM."""
    r_normal = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
                               "T_lcz_degC": 60.0, "T_ambient_degC": 20.0})
    r_angled = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 60.0,
                               "T_lcz_degC": 60.0, "T_ambient_degC": 20.0})
    assert float(r_normal["useful_heat_w"]) > float(r_angled["useful_heat_w"])


# --- Efficiency ---

def test_efficiency_positive(model):
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 20.0,
                        "T_lcz_degC": 60.0, "T_ambient_degC": 25.0})
    assert float(r["efficiency"]) > 0


def test_efficiency_range(model):
    """Solar pond efficiency typically 10-50% depending on LCZ temp and losses."""
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 20.0,
                        "T_lcz_degC": 50.0, "T_ambient_degC": 20.0})
    eta = float(r["efficiency"])
    assert 0.05 < eta < 0.75, f"Efficiency = {eta:.4f}"


# --- Temperature output ---

def test_extraction_temp_above_lcz(model):
    """Extraction temp should be >= LCZ when collecting heat."""
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 20.0,
                        "T_lcz_degC": 80.0, "T_ambient_degC": 25.0})
    assert float(r["T_extraction_degC"]) >= 80.0


# --- Array inputs ---

def test_array_inputs(model):
    G = np.array([400.0, 700.0, 900.0])
    r = model.predict({"irradiance_w_m2": G, "incidence_angle_deg": 20.0,
                        "T_lcz_degC": 80.0, "T_ambient_degC": 25.0})
    assert r["useful_heat_w"].shape == (3,)


# --- Benchmark ---

def test_benchmark(model):
    G = np.random.uniform(100, 1000, 1000)
    theta = np.random.uniform(0, 70, 1000)
    start = time.perf_counter()
    model.predict({"irradiance_w_m2": G, "incidence_angle_deg": theta,
                    "T_lcz_degC": 80.0, "T_ambient_degC": 25.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
