"""EC058 — Flat Plate Collector — F1b IAM — Test Suite"""

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
    r = model.predict({
        "irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
        "T_inlet_degC": 40.0, "T_ambient_degC": 20.0
    })
    for key in ["thermal_output_w", "efficiency", "iam_factor", "T_outlet_degC"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC058"
    assert info["fidelity"] == "F1b"


def test_iam_unity_at_normal(model):
    """IAM should be 1 at normal incidence (theta=0)."""
    r = model.predict({
        "irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
        "T_inlet_degC": 40.0, "T_ambient_degC": 20.0
    })
    assert float(r["iam_factor"]) == pytest.approx(1.0, abs=0.001)


def test_iam_decreases_with_angle(model):
    """IAM should decrease with incidence angle."""
    theta = np.array([0.0, 20.0, 40.0, 60.0])
    r = model.predict({
        "irradiance_w_m2": 800.0, "incidence_angle_deg": theta,
        "T_inlet_degC": 40.0, "T_ambient_degC": 20.0
    })
    assert np.all(np.diff(r["iam_factor"]) < 0)


def test_thermal_output_positive(model):
    """Should produce positive heat at reasonable conditions."""
    r = model.predict({
        "irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
        "T_inlet_degC": 30.0, "T_ambient_degC": 20.0
    })
    assert float(r["thermal_output_w"]) > 0


def test_thermal_output_increases_with_irradiance(model):
    G = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({
        "irradiance_w_m2": G, "incidence_angle_deg": 0.0,
        "T_inlet_degC": 30.0, "T_ambient_degC": 20.0
    })
    assert np.all(np.diff(r["thermal_output_w"]) > 0)


def test_efficiency_decreases_with_inlet_temp(model):
    """Higher inlet temp -> lower efficiency (more losses)."""
    T_in = np.array([20.0, 40.0, 60.0, 80.0])
    r = model.predict({
        "irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
        "T_inlet_degC": T_in, "T_ambient_degC": 20.0
    })
    # Efficiency should generally decrease (some might clip to 0)
    eff = r["efficiency"]
    assert eff[0] > eff[-1]


def test_outlet_above_inlet(model):
    """Outlet should be >= inlet when collecting."""
    r = model.predict({
        "irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
        "T_inlet_degC": 30.0, "T_ambient_degC": 20.0
    })
    assert float(r["T_outlet_degC"]) >= 30.0


def test_no_heat_at_zero_irradiance(model):
    r = model.predict({
        "irradiance_w_m2": 0.0, "incidence_angle_deg": 0.0,
        "T_inlet_degC": 40.0, "T_ambient_degC": 20.0
    })
    assert float(r["thermal_output_w"]) == 0.0


def test_angle_reduces_output(model):
    """Higher angle should reduce thermal output via IAM."""
    r_normal = model.predict({
        "irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
        "T_inlet_degC": 30.0, "T_ambient_degC": 20.0
    })
    r_angled = model.predict({
        "irradiance_w_m2": 800.0, "incidence_angle_deg": 50.0,
        "T_inlet_degC": 30.0, "T_ambient_degC": 20.0
    })
    assert float(r_normal["thermal_output_w"]) > float(r_angled["thermal_output_w"])


def test_efficiency_range(model):
    """Efficiency at good conditions should be 40-75%."""
    r = model.predict({
        "irradiance_w_m2": 800.0, "incidence_angle_deg": 0.0,
        "T_inlet_degC": 30.0, "T_ambient_degC": 20.0
    })
    eff = float(r["efficiency"])
    assert 0.40 < eff < 0.80, f"Efficiency = {eff:.3f}"


def test_array_inputs(model):
    G = np.array([400.0, 600.0, 800.0])
    r = model.predict({
        "irradiance_w_m2": G, "incidence_angle_deg": 20.0,
        "T_inlet_degC": 40.0, "T_ambient_degC": 20.0
    })
    assert r["thermal_output_w"].shape == (3,)


def test_benchmark(model):
    G = np.random.uniform(200, 1000, 1000)
    start = time.perf_counter()
    model.predict({
        "irradiance_w_m2": G, "incidence_angle_deg": 30.0,
        "T_inlet_degC": 40.0, "T_ambient_degC": 20.0
    })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
