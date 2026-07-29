"""EC018 — LFP Battery — F1a — Test Suite"""

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
    result = model.predict({"soc": 0.5, "current": 1.0})
    assert isinstance(result, dict)
    for key in ["voltage", "ocv", "power", "dsoc_dt"]:
        assert key in result


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC018"
    assert info["fidelity"] == "F1a"


def test_ocv_increases_with_soc(model):
    socs = np.linspace(0.1, 0.9, 50)
    result = model.predict({"soc": socs, "current": 0.0})
    assert result["ocv"][-1] > result["ocv"][0]


def test_lfp_flat_plateau(model):
    """LFP should have a flat voltage plateau between ~20-80% SOC."""
    socs = np.linspace(0.2, 0.8, 50)
    result = model.predict({"soc": socs, "current": 0.0})
    v_range = result["ocv"].max() - result["ocv"].min()
    assert v_range < 0.5, f"LFP plateau should be flat, got {v_range:.3f}V range"


def test_voltage_drops_with_current(model):
    currents = np.array([0.0, 1.0, 5.0, 10.0])
    result = model.predict({"soc": 0.5, "current": currents})
    assert np.all(np.diff(result["voltage"]) <= 0)


def test_voltage_within_bounds(model):
    for soc in [0.0, 0.5, 1.0]:
        currents = np.linspace(-17.5, 17.5, 100)
        result = model.predict({"soc": soc, "current": currents})
        assert np.all(result["voltage"] >= 2.0)
        assert np.all(result["voltage"] <= 3.6)


def test_ocv_at_full(model):
    result = model.predict({"soc": 1.0, "current": 0.0})
    assert 3.3 <= result["ocv"] <= 3.7, f"OCV at full = {result['ocv']:.3f}V"


def test_ocv_at_empty(model):
    result = model.predict({"soc": 0.0, "current": 0.0})
    assert 2.4 <= result["ocv"] <= 3.0, f"OCV at empty = {result['ocv']:.3f}V"


def test_zero_current(model):
    result = model.predict({"soc": 0.5, "current": 0.0})
    assert abs(result["voltage"] - result["ocv"]) < 1e-10


def test_soc_derivative_sign(model):
    r1 = model.predict({"soc": 0.5, "current": 5.0})
    r2 = model.predict({"soc": 0.5, "current": -5.0})
    assert r1["dsoc_dt"] < 0
    assert r2["dsoc_dt"] > 0


def test_benchmark(model):
    socs = np.random.uniform(0.0, 1.0, 1000)
    currents = np.random.uniform(-10.0, 10.0, 1000)
    start = time.perf_counter()
    model.predict({"soc": socs, "current": currents})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
