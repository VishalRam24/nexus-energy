"""
EC035 -- NaS Battery -- F1b SOC-Thermal -- Test Suite

Key physics: cell is NON-FUNCTIONAL outside 300-350 degC (573-623 K).
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
    result = model.predict({"soc": 0.5, "current": 50.0, "temperature": 593.15})
    for key in ["terminal_voltage", "power", "heat_generation",
                "effective_capacity", "internal_resistance", "functional"]:
        assert key in result


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC035"
    assert info["fidelity"] == "F1b"


# --- Functional window enforcement ---

def test_functional_true_within_operating_window(model):
    """Cell must report functional=True at 300-350 degC."""
    for T_degC in [300, 320, 350]:
        T_K = T_degC + 273.15
        result = model.predict({"soc": 0.5, "current": 0.0, "temperature": T_K})
        assert bool(result["functional"]), f"Cell should be functional at {T_degC} degC"


def test_non_functional_below_300C(model):
    """Below 300 degC electrodes solidify: cell must be non-functional."""
    result = model.predict({"soc": 0.5, "current": 50.0, "temperature": 550.0})  # 277 degC
    assert not bool(result["functional"]), "Cell must be non-functional below 300 degC"
    assert float(result["terminal_voltage"]) == 0.0, "Voltage must be 0 when non-functional"
    assert float(result["power"]) == 0.0, "Power must be 0 when non-functional"


def test_non_functional_above_350C(model):
    """Above 350 degC: cell is thermally unsafe, must be non-functional."""
    result = model.predict({"soc": 0.5, "current": 50.0, "temperature": 650.0})  # 377 degC
    assert not bool(result["functional"]), "Cell must be non-functional above 350 degC"
    assert float(result["terminal_voltage"]) == 0.0


# --- Arrhenius within operating window ---

def test_resistance_increases_at_lower_operating_temp(model):
    """Within 300-350C: R increases as T decreases (Arrhenius)."""
    r_lo = model.predict({"soc": 0.5, "current": 0, "temperature": 573.15})["internal_resistance"]
    r_ref = model.predict({"soc": 0.5, "current": 0, "temperature": 593.15})["internal_resistance"]
    r_hi = model.predict({"soc": 0.5, "current": 0, "temperature": 623.15})["internal_resistance"]
    assert float(r_lo) > float(r_ref), "R at 300C must be > R at 320C"
    assert float(r_ref) > float(r_hi), "R at 320C must be > R at 350C"


def test_resistance_at_operating_reference(model):
    r = model.predict({"soc": 0.5, "current": 0, "temperature": 593.15})["internal_resistance"]
    assert abs(float(r) - 0.005) < 1e-6


def test_voltage_lower_at_low_end_of_operating_window(model):
    """Higher R at 300C vs 350C -> lower terminal voltage at moderate current.
    # RATIONALE: At very high current (100 A), the ohmic drop (0.5 V) clips terminal voltage
    # to v_min=1.78 V at both temperatures, masking the R(T) difference.
    # At 20 A, drop is ~0.09-0.11 V, leaving both voltages above v_min and distinguishable.
    # Ref: NaS cell R_ref=0.005 Ohm, E_a=12000 J/mol, R(300C)/R(320C) ~ 1.09x (Wen et al. 2008).
    """
    v_lo = model.predict({"soc": 0.5, "current": 20.0, "temperature": 573.15})["terminal_voltage"]
    v_hi = model.predict({"soc": 0.5, "current": 20.0, "temperature": 623.15})["terminal_voltage"]
    assert float(v_lo) < float(v_hi), "V at 300C must be lower than at 350C at 20A current"


# --- Heat generation ---

def test_heat_generation_positive_discharge(model):
    q = model.predict({"soc": 0.5, "current": 50.0, "temperature": 593.15})["heat_generation"]
    assert float(q) > 0, "Heat generation must be positive during discharge"


def test_heat_generation_increases_with_current(model):
    q1 = model.predict({"soc": 0.5, "current": 10.0, "temperature": 593.15})["heat_generation"]
    q5 = model.predict({"soc": 0.5, "current": 50.0, "temperature": 593.15})["heat_generation"]
    q10 = model.predict({"soc": 0.5, "current": 100.0, "temperature": 593.15})["heat_generation"]
    assert float(q5) > float(q1)
    assert float(q10) > float(q5)


def test_heat_generation_zero_when_non_functional(model):
    q = model.predict({"soc": 0.5, "current": 50.0, "temperature": 400.0})["heat_generation"]
    assert float(q) == 0.0, "Heat generation must be 0 when cell is non-functional"


# --- Capacity ---

def test_capacity_increases_with_temperature_in_operating_window(model):
    c_lo = model.predict({"soc": 0.5, "current": 0, "temperature": 573.15})["effective_capacity"]
    c_ref = model.predict({"soc": 0.5, "current": 0, "temperature": 593.15})["effective_capacity"]
    c_hi = model.predict({"soc": 0.5, "current": 0, "temperature": 623.15})["effective_capacity"]
    assert float(c_lo) < float(c_ref) < float(c_hi)


def test_capacity_at_op_reference(model):
    c = model.predict({"soc": 0.5, "current": 0, "temperature": 593.15})["effective_capacity"]
    assert abs(float(c) - 100.0) < 1e-6


# --- Edge cases ---

def test_zero_current_voltage_equals_ocv_when_functional(model):
    result = model.predict({"soc": 0.5, "current": 0.0, "temperature": 593.15})
    assert abs(float(result["terminal_voltage"]) - float(result["ocv"])) < 1e-10


def test_soc_bounds(model):
    for soc in [0.0, 1.0]:
        result = model.predict({"soc": soc, "current": 50.0, "temperature": 593.15})
        assert np.isfinite(float(result["terminal_voltage"]))


def test_array_inputs(model):
    socs = np.array([0.2, 0.5, 0.8])
    currents = np.array([10.0, 50.0, 100.0])
    temps = np.array([573.15, 593.15, 623.15])
    result = model.predict({"soc": socs, "current": currents, "temperature": temps})
    assert result["terminal_voltage"].shape == (3,)


def test_benchmark_1000_predictions(model):
    socs = np.random.uniform(0.0, 1.0, 1000)
    currents = np.random.uniform(-100.0, 100.0, 1000)
    temps = np.random.uniform(573.15, 623.15, 1000)
    start = time.perf_counter()
    model.predict({"soc": socs, "current": currents, "temperature": temps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
