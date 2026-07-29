"""
EC034 -- Aluminum-Ion Battery -- F1b SOC-Thermal -- Test Suite
Run: python -m pytest test_model.py -v
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


def test_predict_returns_all_keys(model):
    r = model.predict({"soc": 0.5, "current": 1.0, "temperature": 298.15})
    for k in ["terminal_voltage", "power", "heat_generation",
              "effective_capacity", "internal_resistance", "ocv", "dsoc_dt"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC034"
    assert model.get_info()["fidelity"] == "F1b"


def test_resistance_increases_at_low_T(model):
    r_cold = float(model.predict({"soc": 0.5, "current": 0, "temperature": 263.15})["internal_resistance"])
    r_ref  = float(model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance"])
    r_hot  = float(model.predict({"soc": 0.5, "current": 0, "temperature": 333.15})["internal_resistance"])
    assert r_cold > r_ref > r_hot


def test_resistance_at_T_ref(model):
    r = float(model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance"])
    assert abs(r - 0.030) < 1e-6


def test_zero_current_voltage_equals_ocv(model):
    r = model.predict({"soc": 0.5, "current": 0.0, "temperature": 298.15})
    assert abs(float(r["terminal_voltage"]) - float(r["ocv"])) < 1e-10


def test_voltage_lower_at_cold(model):
    v_cold = float(model.predict({"soc": 0.5, "current": 5.0, "temperature": 263.15})["terminal_voltage"])
    v_hot  = float(model.predict({"soc": 0.5, "current": 5.0, "temperature": 333.15})["terminal_voltage"])
    assert v_cold < v_hot, "Cold T -> higher R -> lower voltage at high current"


def test_ocv_at_full_near_nominal(model):
    """Al-ion OCV at SOC=1 should be near the 2.45 V max."""
    ocv = float(model.predict({"soc": 1.0, "current": 0, "temperature": 298.15})["ocv"])
    assert 2.0 <= ocv <= 2.5, f"Al-ion OCV at SOC=1 should be ~2.4V, got {ocv:.3f}"


def test_heat_generation_positive_discharge(model):
    """At moderate discharge, Joule heating dominates -> Q > 0."""
    q = float(model.predict({"soc": 0.5, "current": 3.0, "temperature": 298.15})["heat_generation"])
    assert q > 0, "Heat generation must be positive at discharge"


def test_capacity_increases_with_T(model):
    """Al-ion capacity is strongly T-dependent (ionic liquid viscosity)."""
    c_cold = float(model.predict({"soc": 0.5, "current": 0, "temperature": 263.15})["effective_capacity"])
    c_hot  = float(model.predict({"soc": 0.5, "current": 0, "temperature": 333.15})["effective_capacity"])
    assert c_hot > c_cold, "Capacity must increase with T (lower ionic liquid viscosity)"


def test_high_rate_capability(model):
    """Al-ion supports 60-100C rates -> test at high current without crash."""
    r = model.predict({"soc": 0.5, "current": 8.0, "temperature": 298.15})
    assert np.isfinite(float(r["terminal_voltage"])), "Must handle high current"


def test_array_inputs(model):
    socs  = np.array([0.2, 0.5, 0.8])
    currs = np.array([1.0, 3.0, 7.0])
    temps = np.array([273.15, 298.15, 318.15])
    r = model.predict({"soc": socs, "current": currs, "temperature": temps})
    assert r["terminal_voltage"].shape == (3,)


def test_benchmark_1000_predictions(model):
    socs  = np.random.uniform(0.0, 1.0, 1000)
    currs = np.random.uniform(-10.0, 10.0, 1000)
    temps = np.random.uniform(263.15, 333.15, 1000)
    start = time.perf_counter()
    model.predict({"soc": socs, "current": currs, "temperature": temps})
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
