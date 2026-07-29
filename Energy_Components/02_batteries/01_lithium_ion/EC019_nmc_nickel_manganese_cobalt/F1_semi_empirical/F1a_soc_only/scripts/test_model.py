"""
EC019 — NMC Battery — F1a SOC-Only — Test Suite

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


# --- Output structure ---

def test_predict_returns_dict(model):
    result = model.predict({"soc": 0.5, "current": 1.0})
    assert isinstance(result, dict)
    for key in ["voltage", "ocv", "power", "dsoc_dt"]:
        assert key in result


def test_get_info_returns_dict(model):
    info = model.get_info()
    assert info["ec_id"] == "EC019"
    assert info["fidelity"] == "F1a"


# --- Physics monotonicity ---

def test_ocv_increases_with_soc(model):
    """OCV must generally increase with SOC for NMC."""
    socs = np.linspace(0.1, 0.9, 50)
    result = model.predict({"soc": socs, "current": 0.0})
    ocv = result["ocv"]
    # Allow small non-monotonic regions but overall trend must be positive
    assert ocv[-1] > ocv[0], "OCV at SOC=0.9 must be higher than at SOC=0.1"


def test_voltage_drops_with_discharge_current(model):
    """Higher discharge current -> lower terminal voltage."""
    currents = np.array([0.0, 1.0, 5.0, 10.0, 25.0])
    result = model.predict({"soc": 0.5, "current": currents})
    v = result["voltage"]
    assert np.all(np.diff(v) <= 0), "Voltage must decrease with increasing discharge current"


def test_voltage_rises_with_charge_current(model):
    """During charging (negative current), voltage should be above OCV."""
    result = model.predict({"soc": 0.5, "current": -5.0})
    ocv_result = model.predict({"soc": 0.5, "current": 0.0})
    assert result["voltage"] >= ocv_result["voltage"], \
        "Charging voltage must be >= OCV"


# --- Known limits ---

def test_voltage_within_bounds(model):
    """Terminal voltage must stay within [V_min, V_max]."""
    socs = np.linspace(0.0, 1.0, 100)
    currents = np.linspace(-25.0, 25.0, 100)
    for soc in [0.0, 0.5, 1.0]:
        result = model.predict({"soc": soc, "current": currents})
        assert np.all(result["voltage"] >= 2.5), "Voltage must be >= V_min (2.5V)"
        assert np.all(result["voltage"] <= 4.2), "Voltage must be <= V_max (4.2V)"


def test_ocv_at_full_charge(model):
    """OCV at SOC=1.0 should be close to V_max (~4.2V)."""
    result = model.predict({"soc": 1.0, "current": 0.0})
    assert 3.9 <= result["ocv"] <= 4.3, f"OCV at full charge = {result['ocv']:.3f}V, expected ~4.2V"


def test_ocv_at_empty(model):
    """OCV at SOC=0.0 should be close to V_min (~2.5-3.0V)."""
    result = model.predict({"soc": 0.0, "current": 0.0})
    assert 2.5 <= result["ocv"] <= 3.2, f"OCV at empty = {result['ocv']:.3f}V, expected ~2.8V"


# --- Edge cases ---

def test_zero_current(model):
    """At zero current, terminal voltage = OCV."""
    result = model.predict({"soc": 0.5, "current": 0.0})
    assert abs(result["voltage"] - result["ocv"]) < 1e-10


def test_soc_derivative_sign(model):
    """Discharge (positive I) should decrease SOC; charge should increase SOC."""
    result_discharge = model.predict({"soc": 0.5, "current": 5.0})
    result_charge = model.predict({"soc": 0.5, "current": -5.0})
    assert result_discharge["dsoc_dt"] < 0, "Discharge must decrease SOC"
    assert result_charge["dsoc_dt"] > 0, "Charge must increase SOC"


def test_array_inputs(model):
    """Model must handle array inputs."""
    socs = np.array([0.2, 0.5, 0.8])
    currents = np.array([1.0, 2.0, 3.0])
    result = model.predict({"soc": socs, "current": currents})
    assert result["voltage"].shape == (3,)


# --- Benchmark ---

def test_benchmark_1000_predictions(model):
    """Report time for 1000 predictions (should be <1s for F1a)."""
    socs = np.random.uniform(0.0, 1.0, 1000)
    currents = np.random.uniform(-10.0, 10.0, 1000)
    start = time.perf_counter()
    model.predict({"soc": socs, "current": currents})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s for 1000 predictions"
