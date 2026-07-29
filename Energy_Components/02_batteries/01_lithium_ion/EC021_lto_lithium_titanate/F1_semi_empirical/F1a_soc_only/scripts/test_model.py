"""
EC021 — LTO Battery — F1a SOC-Only — Test Suite

Physics sanity checks, edge cases, and performance benchmark.
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


def test_predict_returns_dict(model):
    result = model.predict({"soc": 0.5, "current": 1.0})
    assert isinstance(result, dict)
    for key in ["voltage", "ocv", "power", "dsoc_dt"]:
        assert key in result


def test_get_info_returns_dict(model):
    info = model.get_info()
    assert info["ec_id"] == "EC021"
    assert info["fidelity"] == "F1a"


def test_ocv_increases_with_soc(model):
    """OCV must overall increase with SOC for LTO."""
    socs = np.linspace(0.1, 0.9, 50)
    result = model.predict({"soc": socs, "current": 0.0})
    ocv = result["ocv"]
    assert ocv[-1] > ocv[0]


def test_voltage_drops_with_discharge_current(model):
    currents = np.array([0.0, 1.0, 5.0, 10.0, 25.0])
    result = model.predict({"soc": 0.5, "current": currents})
    v = result["voltage"]
    assert np.all(np.diff(v) <= 0)


def test_voltage_rises_with_charge_current(model):
    result = model.predict({"soc": 0.5, "current": -5.0})
    ocv_result = model.predict({"soc": 0.5, "current": 0.0})
    assert result["voltage"] >= ocv_result["voltage"]


def test_voltage_within_bounds(model):
    """Terminal voltage must stay within [V_min, V_max] for LTO."""
    currents = np.linspace(-30.0, 30.0, 100)
    for soc in [0.0, 0.5, 1.0]:
        result = model.predict({"soc": soc, "current": currents})
        assert np.all(result["voltage"] >= 1.5)
        assert np.all(result["voltage"] <= 2.7)


def test_ocv_at_full_charge(model):
    """OCV at SOC=1.0 should be near V_max ~2.7V."""
    result = model.predict({"soc": 1.0, "current": 0.0})
    assert 2.5 <= result["ocv"] <= 2.8


def test_ocv_at_empty(model):
    """OCV at SOC=0.0 should be near V_min ~1.5V."""
    result = model.predict({"soc": 0.0, "current": 0.0})
    assert 1.4 <= result["ocv"] <= 1.7


def test_ocv_plateau_near_24v(model):
    """LTO OCV in mid-SOC should sit around the 2.4 V plateau."""
    result = model.predict({"soc": 0.5, "current": 0.0})
    assert 2.30 <= float(result["ocv"]) <= 2.50


def test_zero_current(model):
    result = model.predict({"soc": 0.5, "current": 0.0})
    assert abs(result["voltage"] - result["ocv"]) < 1e-10


def test_soc_derivative_sign(model):
    result_discharge = model.predict({"soc": 0.5, "current": 5.0})
    result_charge = model.predict({"soc": 0.5, "current": -5.0})
    assert result_discharge["dsoc_dt"] < 0
    assert result_charge["dsoc_dt"] > 0


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
