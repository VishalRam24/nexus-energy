"""
EC032 -- Zinc-Air Battery -- F1b SOC-Thermal -- Test Suite
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

def test_predict_returns_all_keys(model):
    r = model.predict({"soc": 0.5, "current": 1.0, "temperature": 298.15})
    for k in ["terminal_voltage", "power", "heat_generation",
              "effective_capacity", "internal_resistance", "ocv", "dsoc_dt"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC032"
    assert info["fidelity"] == "F1b"


# --- Arrhenius resistance ---

def test_resistance_increases_at_low_T(model):
    r_cold = float(model.predict({"soc": 0.5, "current": 0, "temperature": 253.15})["internal_resistance"])
    r_ref  = float(model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance"])
    r_hot  = float(model.predict({"soc": 0.5, "current": 0, "temperature": 333.15})["internal_resistance"])
    assert r_cold > r_ref > r_hot, "R must decrease with increasing T (Arrhenius)"


def test_resistance_at_T_ref_equals_R_ref(model):
    r = float(model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance"])
    assert abs(r - 0.050) < 1e-6, f"R at T_ref must equal R_ref=0.050, got {r}"


# --- Voltage ---

def test_zero_current_voltage_equals_ocv(model):
    r = model.predict({"soc": 0.5, "current": 0.0, "temperature": 298.15})
    assert abs(float(r["terminal_voltage"]) - float(r["ocv"])) < 1e-10


def test_voltage_lower_at_cold_high_current(model):
    v_cold = float(model.predict({"soc": 0.5, "current": 3.0, "temperature": 253.15})["terminal_voltage"])
    v_hot  = float(model.predict({"soc": 0.5, "current": 3.0, "temperature": 333.15})["terminal_voltage"])
    assert v_cold < v_hot, "Cold T -> higher R -> lower voltage at high discharge current"


def test_ocv_decreases_with_soc(model):
    """Zn-air OCV decreases from ~1.65 V (full) to ~0.9 V (empty)."""
    ocv_full  = float(model.predict({"soc": 1.0, "current": 0, "temperature": 298.15})["ocv"])
    ocv_empty = float(model.predict({"soc": 0.0, "current": 0, "temperature": 298.15})["ocv"])
    assert ocv_full > ocv_empty, "OCV must decrease from full to empty"


def test_ocv_at_full_is_high(model):
    """Zn-air OCV at SOC=1 should be close to 1.65 V (E^0 for Zn/O2)."""
    ocv = float(model.predict({"soc": 1.0, "current": 0, "temperature": 298.15})["ocv"])
    assert 1.5 <= ocv <= 1.7, f"Zn-air OCV at SOC=1 should be ~1.65 V, got {ocv:.3f}"


# --- Heat generation ---

def test_heat_generation_positive_at_moderate_discharge(model):
    """At moderate discharge, Joule heating must dominate -> Q > 0."""
    q = float(model.predict({"soc": 0.5, "current": 2.0, "temperature": 298.15})["heat_generation"])
    assert q > 0, "Heat generation must be positive at moderate discharge (I^2*R dominates)"


def test_heat_increases_with_current(model):
    q1 = float(model.predict({"soc": 0.5, "current": 1.0, "temperature": 298.15})["heat_generation"])
    q3 = float(model.predict({"soc": 0.5, "current": 3.0, "temperature": 298.15})["heat_generation"])
    assert q3 > q1, "Heat must increase with current (I^2*R dominates)"


# --- Capacity ---

def test_capacity_increases_with_temperature(model):
    c_cold = float(model.predict({"soc": 0.5, "current": 0, "temperature": 253.15})["effective_capacity"])
    c_hot  = float(model.predict({"soc": 0.5, "current": 0, "temperature": 333.15})["effective_capacity"])
    assert c_hot > c_cold, "Capacity must increase with T"


# --- Edge cases ---

def test_soc_boundary_conditions(model):
    for soc in [0.0, 1.0]:
        r = model.predict({"soc": soc, "current": 1.0, "temperature": 298.15})
        assert np.isfinite(float(r["terminal_voltage"]))


def test_array_inputs(model):
    socs  = np.array([0.2, 0.5, 0.8])
    currs = np.array([0.5, 1.0, 2.0])
    temps = np.array([273.15, 298.15, 318.15])
    r = model.predict({"soc": socs, "current": currs, "temperature": temps})
    assert r["terminal_voltage"].shape == (3,)


# --- Benchmark ---

def test_benchmark_1000_predictions(model):
    socs  = np.random.uniform(0.0, 1.0, 1000)
    currs = np.random.uniform(-5.0, 5.0, 1000)
    temps = np.random.uniform(253.15, 333.15, 1000)
    start = time.perf_counter()
    model.predict({"soc": socs, "current": currs, "temperature": temps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
