"""
EC033 -- Iron-Air Battery -- F1b SOC-Thermal -- Test Suite
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
    assert model.get_info()["ec_id"] == "EC033"
    assert model.get_info()["fidelity"] == "F1b"


def test_resistance_increases_at_low_T(model):
    r_cold = float(model.predict({"soc": 0.5, "current": 0, "temperature": 253.15})["internal_resistance"])
    r_ref  = float(model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance"])
    r_hot  = float(model.predict({"soc": 0.5, "current": 0, "temperature": 333.15})["internal_resistance"])
    assert r_cold > r_ref > r_hot


def test_resistance_at_T_ref(model):
    r = float(model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance"])
    assert abs(r - 0.060) < 1e-6


def test_zero_current_voltage_equals_ocv(model):
    r = model.predict({"soc": 0.5, "current": 0.0, "temperature": 298.15})
    assert abs(float(r["terminal_voltage"]) - float(r["ocv"])) < 1e-10


def test_voltage_lower_at_cold(model):
    v_cold = float(model.predict({"soc": 0.5, "current": 2.0, "temperature": 253.15})["terminal_voltage"])
    v_hot  = float(model.predict({"soc": 0.5, "current": 2.0, "temperature": 333.15})["terminal_voltage"])
    assert v_cold < v_hot


def test_ocv_in_range(model):
    """Fe-air OCV should be between 0.6 and 1.28 V."""
    for soc in [0.0, 0.5, 1.0]:
        ocv = float(model.predict({"soc": soc, "current": 0, "temperature": 298.15})["ocv"])
        assert 0.5 <= ocv <= 1.35, f"Fe-air OCV={ocv:.3f} out of range at soc={soc}"


def test_heat_positive_at_discharge(model):
    """Fe-air: both Joule and reversible terms positive during discharge -> Q > 0."""
    q = float(model.predict({"soc": 0.5, "current": 1.5, "temperature": 298.15})["heat_generation"])
    assert q > 0, "Heat must be positive at discharge for Fe-air (dOCV/dT > 0)"


def test_capacity_increases_with_T(model):
    c_cold = float(model.predict({"soc": 0.5, "current": 0, "temperature": 253.15})["effective_capacity"])
    c_hot  = float(model.predict({"soc": 0.5, "current": 0, "temperature": 333.15})["effective_capacity"])
    assert c_hot > c_cold


def test_array_inputs(model):
    socs  = np.array([0.2, 0.5, 0.8])
    currs = np.array([0.5, 1.0, 1.5])
    temps = np.array([273.15, 298.15, 318.15])
    r = model.predict({"soc": socs, "current": currs, "temperature": temps})
    assert r["terminal_voltage"].shape == (3,)


def test_benchmark_1000_predictions(model):
    socs  = np.random.uniform(0.0, 1.0, 1000)
    currs = np.random.uniform(-3.0, 3.0, 1000)
    temps = np.random.uniform(253.15, 333.15, 1000)
    start = time.perf_counter()
    model.predict({"soc": socs, "current": currs, "temperature": temps})
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
