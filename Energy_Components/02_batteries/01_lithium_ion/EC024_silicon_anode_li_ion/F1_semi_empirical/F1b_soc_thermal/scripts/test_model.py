"""
EC024 -- Silicon-Anode Li-ion Battery -- F1b SOC-Thermal -- Test Suite

Physics sanity checks for temperature-dependent Si/NMC battery model.
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
    result = model.predict({"soc": 0.5, "current": 1.0, "temperature": 298.15})
    for key in ["terminal_voltage", "power", "heat_generation",
                "effective_capacity", "internal_resistance"]:
        assert key in result


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC024"
    assert info["fidelity"] == "F1b"


def test_resistance_increases_at_low_temperature(model):
    r_cold = model.predict({"soc": 0.5, "current": 0, "temperature": 253.15})["internal_resistance"]
    r_ref = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance"]
    r_hot = model.predict({"soc": 0.5, "current": 0, "temperature": 333.15})["internal_resistance"]
    assert float(r_cold) > float(r_ref), "R at -20C must be > R at 25C"
    assert float(r_ref) > float(r_hot), "R at 25C must be > R at 60C"


def test_resistance_at_reference_equals_R_ref(model):
    r = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance"]
    assert abs(float(r) - 0.020) < 1e-6, f"R at T_ref should be 0.020, got {float(r)}"


def test_voltage_lower_at_cold_high_current(model):
    v_cold = model.predict({"soc": 0.5, "current": 10.0, "temperature": 253.15})["terminal_voltage"]
    v_hot = model.predict({"soc": 0.5, "current": 10.0, "temperature": 333.15})["terminal_voltage"]
    assert float(v_cold) < float(v_hot), "Voltage at -20C must be lower than at 60C"


def test_heat_generation_positive_discharge(model):
    q = model.predict({"soc": 0.5, "current": 3.5, "temperature": 298.15})["heat_generation"]
    assert float(q) > 0


def test_heat_generation_increases_with_current(model):
    q1 = model.predict({"soc": 0.5, "current": 1.0, "temperature": 298.15})["heat_generation"]
    q5 = model.predict({"soc": 0.5, "current": 5.0, "temperature": 298.15})["heat_generation"]
    q10 = model.predict({"soc": 0.5, "current": 10.0, "temperature": 298.15})["heat_generation"]
    assert float(q5) > float(q1)
    assert float(q10) > float(q5)


def test_capacity_increases_with_temperature(model):
    c_cold = model.predict({"soc": 0.5, "current": 0, "temperature": 253.15})["effective_capacity"]
    c_ref = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["effective_capacity"]
    c_hot = model.predict({"soc": 0.5, "current": 0, "temperature": 333.15})["effective_capacity"]
    assert float(c_cold) < float(c_ref) < float(c_hot)


def test_capacity_at_reference(model):
    c = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["effective_capacity"]
    assert abs(float(c) - 3.5) < 1e-6


def test_soc_zero(model):
    result = model.predict({"soc": 0.0, "current": 1.0, "temperature": 298.15})
    assert np.isfinite(float(result["terminal_voltage"]))


def test_soc_one(model):
    result = model.predict({"soc": 1.0, "current": 1.0, "temperature": 298.15})
    assert np.isfinite(float(result["terminal_voltage"]))


def test_temperature_extremes(model):
    for T in [253.15, 333.15]:
        result = model.predict({"soc": 0.5, "current": 1.0, "temperature": T})
        assert np.isfinite(float(result["terminal_voltage"]))
        assert float(result["internal_resistance"]) > 0


def test_zero_current_voltage_equals_ocv(model):
    result = model.predict({"soc": 0.5, "current": 0.0, "temperature": 298.15})
    assert abs(float(result["terminal_voltage"]) - float(result["ocv"])) < 1e-10


def test_array_inputs(model):
    socs = np.array([0.2, 0.5, 0.8])
    currents = np.array([1.0, 2.0, 3.0])
    temps = np.array([273.15, 298.15, 313.15])
    result = model.predict({"soc": socs, "current": currents, "temperature": temps})
    assert result["terminal_voltage"].shape == (3,)


def test_benchmark_1000_predictions(model):
    socs = np.random.uniform(0.0, 1.0, 1000)
    currents = np.random.uniform(-10.0, 10.0, 1000)
    temps = np.random.uniform(253.15, 333.15, 1000)
    start = time.perf_counter()
    model.predict({"soc": socs, "current": currents, "temperature": temps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
