"""
EC039 -- Organic Flow Battery -- F1b SOC-Thermal -- Test Suite

Physics sanity checks for temperature-dependent OFB Nernst model (AQDS/ferricyanide).
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
    result = model.predict({"soc": 0.5, "current": 20.0, "temperature": 298.15})
    for key in ["stack_voltage", "cell_voltage", "power", "heat_generation",
                "pump_loss", "internal_resistance_cell", "e_nernst", "efficiency"]:
        assert key in result


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC039"
    assert info["fidelity"] == "F1b"


# --- Arrhenius resistance ---

def test_resistance_increases_at_lower_temperature(model):
    r_cold = model.predict({"soc": 0.5, "current": 0, "temperature": 283.15})["internal_resistance_cell"]
    r_ref = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance_cell"]
    r_hot = model.predict({"soc": 0.5, "current": 0, "temperature": 313.15})["internal_resistance_cell"]
    assert float(r_cold) > float(r_ref), "R at 10C must be > R at 25C"
    assert float(r_ref) > float(r_hot), "R at 25C must be > R at 40C"


def test_resistance_at_reference(model):
    # R_cell_ref = 6.0 Ohm.cm2 / 100 cm2 = 0.06 Ohm
    R_cell_ref = 6.0 / 100.0
    r = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance_cell"]
    assert abs(float(r) - R_cell_ref) < 1e-8


# --- Nernst potential ---

def test_nernst_increases_with_soc(model):
    e_lo = model.predict({"soc": 0.2, "current": 0, "temperature": 298.15})["e_nernst"]
    e_hi = model.predict({"soc": 0.8, "current": 0, "temperature": 298.15})["e_nernst"]
    assert float(e_hi) > float(e_lo), "Nernst potential must increase with SOC"


def test_nernst_at_ref_midpoint(model):
    """At SOC=0.5, Nernst = E0(T_ref) = 0.70 V (log term = 0)."""
    e = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["e_nernst"]
    assert abs(float(e) - 0.70) < 1e-6, f"Nernst at SOC=0.5, T_ref should be E0=0.70V, got {float(e):.6f}"


def test_nernst_decreases_with_temperature(model):
    """E0(T) = E0_ref + dOCV_dT*(T-T_ref), dOCV_dT < 0 -> higher T gives lower E0."""
    e_cold = model.predict({"soc": 0.5, "current": 0, "temperature": 283.15})["e_nernst"]
    e_hot = model.predict({"soc": 0.5, "current": 0, "temperature": 313.15})["e_nernst"]
    assert float(e_cold) > float(e_hot), "Higher temperature must reduce E0 (dOCV_dT < 0)"


# --- Stack voltage ---

def test_stack_voltage_lower_at_cold_high_current(model):
    v_cold = model.predict({"soc": 0.5, "current": 40.0, "temperature": 283.15})["stack_voltage"]
    v_hot = model.predict({"soc": 0.5, "current": 40.0, "temperature": 313.15})["stack_voltage"]
    # RATIONALE: At I=40A (near max), Arrhenius R dominates. R(10C)/R(40C) ~= exp(16000/8.314 *
    # (1/283.15 - 1/313.15)) ~ exp(16000/8.314 * 0.000338) ~ exp(0.651) ~ 1.92.
    # Net ohmic loss difference outweighs E0(T) change from dOCV_dT = -0.0003 V/K.
    assert float(v_cold) < float(v_hot), "Stack voltage at 10C must be lower than at 40C (high current)"


def test_stack_voltage_proportional_to_cell_count(model):
    """V_stack = N_cells * V_cell."""
    result = model.predict({"soc": 0.5, "current": 20.0, "temperature": 298.15})
    v_stack = float(result["stack_voltage"])
    v_cell = float(result["cell_voltage"])
    N = 20
    assert abs(v_stack - N * v_cell) < 1e-9


# --- Pump loss ---

def test_pump_loss_quadratic(model):
    """P_pump = k_pump * I^2 -> P(2I)/P(I) = 4."""
    p1 = float(model.predict({"soc": 0.5, "current": 10.0, "temperature": 298.15})["pump_loss"])
    p2 = float(model.predict({"soc": 0.5, "current": 20.0, "temperature": 298.15})["pump_loss"])
    assert abs(p2 / p1 - 4.0) < 1e-9, "Pump loss must scale as I^2"


# --- Heat generation ---

def test_heat_generation_positive_discharge(model):
    q = model.predict({"soc": 0.5, "current": 20.0, "temperature": 298.15})["heat_generation"]
    assert float(q) > 0, "Heat generation must be positive during discharge"


def test_heat_generation_increases_with_current(model):
    q1 = model.predict({"soc": 0.5, "current": 5.0, "temperature": 298.15})["heat_generation"]
    q2 = model.predict({"soc": 0.5, "current": 20.0, "temperature": 298.15})["heat_generation"]
    q3 = model.predict({"soc": 0.5, "current": 50.0, "temperature": 298.15})["heat_generation"]
    assert float(q2) > float(q1)
    assert float(q3) > float(q2)


# --- Efficiency ---

def test_efficiency_between_zero_and_one(model):
    eta = model.predict({"soc": 0.5, "current": 20.0, "temperature": 298.15})["efficiency"]
    assert 0.0 < float(eta) <= 1.0, f"Efficiency must be in (0, 1], got {float(eta)}"


def test_efficiency_increases_with_temperature(model):
    """Higher T -> lower R -> less ohmic loss -> higher voltage efficiency."""
    eta_cold = model.predict({"soc": 0.5, "current": 20.0, "temperature": 283.15})["efficiency"]
    eta_hot = model.predict({"soc": 0.5, "current": 20.0, "temperature": 313.15})["efficiency"]
    assert float(eta_hot) > float(eta_cold), "Efficiency must be higher at higher temperature"


# --- Zero current ---

def test_zero_current_zero_heat(model):
    q = model.predict({"soc": 0.5, "current": 0.0, "temperature": 298.15})["heat_generation"]
    assert abs(float(q)) < 1e-10


# --- SOC edge cases ---

def test_soc_extreme_clamped_no_error(model):
    for soc in [0.05, 0.95]:
        result = model.predict({"soc": soc, "current": 20.0, "temperature": 298.15})
        assert np.isfinite(float(result["stack_voltage"]))


def test_invalid_soc_raises(model):
    with pytest.raises(ValueError):
        model.predict({"soc": 1.2, "current": 0, "temperature": 298.15})


# --- Array inputs ---

def test_array_inputs(model):
    socs = np.array([0.2, 0.5, 0.8])
    currents = np.array([5.0, 20.0, 40.0])
    temps = np.array([283.15, 298.15, 313.15])
    result = model.predict({"soc": socs, "current": currents, "temperature": temps})
    assert result["stack_voltage"].shape == (3,)


# --- Benchmark ---

def test_benchmark_1000_predictions(model):
    socs = np.random.uniform(0.05, 0.95, 1000)
    currents = np.random.uniform(-50.0, 50.0, 1000)
    temps = np.random.uniform(283.15, 313.15, 1000)
    start = time.perf_counter()
    model.predict({"soc": socs, "current": currents, "temperature": temps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
