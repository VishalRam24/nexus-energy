"""EC027 — Solid-State Li Battery — F1a SOC-Only — Test Suite"""

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


def test_predict_returns_dict(model):
    result = model.predict({"soc": 0.5, "current": 1.0})
    for key in ["voltage", "ocv", "power", "dsoc_dt"]:
        assert key in result


def test_get_info_returns_dict(model):
    info = model.get_info()
    assert info["ec_id"] == "EC027"
    assert info["fidelity"] == "F1a"


def test_ocv_increases_with_soc(model):
    socs = np.linspace(0.1, 0.9, 50)
    result = model.predict({"soc": socs, "current": 0.0})
    assert result["ocv"][-1] > result["ocv"][0]


def test_voltage_drops_with_discharge_current(model):
    currents = np.array([0.0, 1.0, 5.0, 10.0, 20.0])
    result = model.predict({"soc": 0.5, "current": currents})
    assert np.all(np.diff(result["voltage"]) <= 0)


def test_voltage_rises_with_charge_current(model):
    result = model.predict({"soc": 0.5, "current": -5.0})
    ocv_result = model.predict({"soc": 0.5, "current": 0.0})
    assert result["voltage"] >= ocv_result["voltage"]


def test_voltage_within_bounds(model):
    currents = np.linspace(-20.0, 20.0, 100)
    for soc in [0.0, 0.5, 1.0]:
        result = model.predict({"soc": soc, "current": currents})
        assert np.all(result["voltage"] >= 3.0)
        assert np.all(result["voltage"] <= 4.3)


def test_ocv_at_full_charge(model):
    result = model.predict({"soc": 1.0, "current": 0.0})
    assert 4.1 <= result["ocv"] <= 4.4


def test_ocv_at_empty(model):
    result = model.predict({"soc": 0.0, "current": 0.0})
    assert 2.9 <= result["ocv"] <= 3.2


def test_zero_current(model):
    result = model.predict({"soc": 0.5, "current": 0.0})
    assert abs(result["voltage"] - result["ocv"]) < 1e-10


def test_soc_derivative_sign(model):
    assert model.predict({"soc": 0.5, "current": 5.0})["dsoc_dt"] < 0
    assert model.predict({"soc": 0.5, "current": -5.0})["dsoc_dt"] > 0


def test_array_inputs(model):
    socs = np.array([0.2, 0.5, 0.8])
    currents = np.array([1.0, 2.0, 3.0])
    result = model.predict({"soc": socs, "current": currents})
    assert result["voltage"].shape == (3,)


def test_benchmark_1000_predictions(model):
    socs = np.random.uniform(0.0, 1.0, 1000)
    currents = np.random.uniform(-10.0, 10.0, 1000)
    start = time.perf_counter()
    model.predict({"soc": socs, "current": currents})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
