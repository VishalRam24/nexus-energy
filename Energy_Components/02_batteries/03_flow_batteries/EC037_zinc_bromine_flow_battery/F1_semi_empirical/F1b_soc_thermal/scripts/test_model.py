"""
EC037 -- Zinc-Bromine Flow Battery -- F1b SOC-Thermal -- Test Suite

Physics sanity checks for temperature-dependent ZBFB Nernst model.
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
    result = model.predict({"soc": 0.5, "current": 50.0, "temperature": 298.15})
    for key in ["stack_voltage", "cell_voltage", "power", "heat_generation",
                "internal_resistance_cell", "e_nernst", "efficiency"]:
        assert key in result


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC037"
    assert info["fidelity"] == "F1b"


# --- Arrhenius resistance ---

def test_resistance_increases_at_lower_temperature(model):
    r_cold = model.predict({"soc": 0.5, "current": 0, "temperature": 288.15})["internal_resistance_cell"]
    r_ref = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance_cell"]
    r_hot = model.predict({"soc": 0.5, "current": 0, "temperature": 313.15})["internal_resistance_cell"]
    assert float(r_cold) > float(r_ref), "R at 15C must be > R at 25C"
    assert float(r_ref) > float(r_hot), "R at 25C must be > R at 40C"


def test_resistance_at_reference(model):
    R_cell_ref = 1.5 / 1000.0  # Ohm.cm2 / cm2
    r = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance_cell"]
    assert abs(float(r) - R_cell_ref) < 1e-8


# --- Nernst: voltage increases with SOC ---

def test_nernst_increases_with_soc(model):
    e_lo = model.predict({"soc": 0.2, "current": 0, "temperature": 298.15})["e_nernst"]
    e_hi = model.predict({"soc": 0.8, "current": 0, "temperature": 298.15})["e_nernst"]
    assert float(e_hi) > float(e_lo), "Nernst potential must increase with SOC"


# --- Stack voltage higher at higher temperature (lower R -> less drop) ---

def test_stack_voltage_lower_at_cold_high_current(model):
    v_cold = model.predict({"soc": 0.5, "current": 100.0, "temperature": 288.15})["stack_voltage"]
    v_hot = model.predict({"soc": 0.5, "current": 100.0, "temperature": 313.15})["stack_voltage"]
    assert float(v_cold) < float(v_hot), "Stack voltage at 15C must be lower than at 40C (high current)"


# --- Heat generation ---

def test_heat_generation_positive_discharge(model):
    q = model.predict({"soc": 0.5, "current": 50.0, "temperature": 298.15})["heat_generation"]
    assert float(q) > 0, "Heat generation must be positive during discharge"


def test_heat_generation_increases_with_current(model):
    q1 = model.predict({"soc": 0.5, "current": 10.0, "temperature": 298.15})["heat_generation"]
    q5 = model.predict({"soc": 0.5, "current": 50.0, "temperature": 298.15})["heat_generation"]
    q10 = model.predict({"soc": 0.5, "current": 100.0, "temperature": 298.15})["heat_generation"]
    assert float(q5) > float(q1)
    assert float(q10) > float(q5)


# --- Efficiency is in (0, 1] ---

def test_efficiency_between_zero_and_one(model):
    eta = model.predict({"soc": 0.5, "current": 50.0, "temperature": 298.15})["efficiency"]
    assert 0.0 < float(eta) <= 1.0, f"Efficiency must be in (0, 1], got {float(eta)}"


def test_efficiency_increases_with_temperature(model):
    """Higher T -> lower R -> less ohmic loss -> higher efficiency."""
    eta_cold = model.predict({"soc": 0.5, "current": 50.0, "temperature": 288.15})["efficiency"]
    eta_hot = model.predict({"soc": 0.5, "current": 50.0, "temperature": 313.15})["efficiency"]
    assert float(eta_hot) > float(eta_cold), "Efficiency must be higher at higher temperature"


# --- Zero current: zero power, zero heat ---

def test_zero_current_zero_power(model):
    result = model.predict({"soc": 0.5, "current": 0.0, "temperature": 298.15})
    assert abs(float(result["power"])) < 1e-10


# --- SOC edge cases ---

def test_soc_extreme_clamped_no_error(model):
    """SOC near 0 and 1 must be clamped internally, not raise errors."""
    for soc in [0.05, 0.95]:
        result = model.predict({"soc": soc, "current": 50.0, "temperature": 298.15})
        assert np.isfinite(float(result["stack_voltage"]))


# --- Array inputs ---

def test_array_inputs(model):
    socs = np.array([0.2, 0.5, 0.8])
    currents = np.array([10.0, 50.0, 100.0])
    temps = np.array([288.15, 298.15, 313.15])
    result = model.predict({"soc": socs, "current": currents, "temperature": temps})
    assert result["stack_voltage"].shape == (3,)


# --- Benchmark ---

def test_benchmark_1000_predictions(model):
    socs = np.random.uniform(0.05, 0.95, 1000)
    currents = np.random.uniform(-100.0, 100.0, 1000)
    temps = np.random.uniform(288.15, 313.15, 1000)
    start = time.perf_counter()
    model.predict({"soc": socs, "current": currents, "temperature": temps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
