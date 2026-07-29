"""
EC041 -- EDLC Supercapacitor -- F1b RC-Thermal -- Test Suite

Physics sanity checks. Key: EDLC has NO reversible/entropic heat (no electrochemistry).
Q = I^2 * ESR(T) only. Very wide temperature range: -40 to 65 degC.
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
    result = model.predict({"v_cap": 1.35, "current": 100.0, "temperature": 298.15})
    for key in ["terminal_voltage", "power", "heat_generation", "esr", "capacitance",
                "soc", "stored_energy", "dvcap_dt"]:
        assert key in result


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC041"
    assert info["fidelity"] == "F1b"


# --- Arrhenius: ESR increases at lower temperatures ---

def test_esr_increases_at_low_temperature(model):
    """ESR must increase as temperature decreases (Arrhenius)."""
    esr_cold = model.predict({"v_cap": 1.35, "current": 0, "temperature": 233.15})["esr"]
    esr_ref = model.predict({"v_cap": 1.35, "current": 0, "temperature": 298.15})["esr"]
    esr_hot = model.predict({"v_cap": 1.35, "current": 0, "temperature": 338.15})["esr"]
    assert float(esr_cold) > float(esr_ref), "ESR at -40C must be > ESR at 25C"
    assert float(esr_ref) > float(esr_hot), "ESR at 25C must be > ESR at 65C"


def test_esr_at_reference(model):
    esr = model.predict({"v_cap": 1.35, "current": 0, "temperature": 298.15})["esr"]
    assert abs(float(esr) - 0.00029) < 1e-8


def test_esr_moderate_temperature_dependence(model):
    """EDLC E_a=8000 J/mol -> ESR(-40C)/ESR(25C) should be ~3-5x (weaker than batteries)."""
    esr_cold = float(model.predict({"v_cap": 1.35, "current": 0, "temperature": 233.15})["esr"])
    esr_ref = float(model.predict({"v_cap": 1.35, "current": 0, "temperature": 298.15})["esr"])
    ratio = esr_cold / esr_ref
    # RATIONALE: EDLC E_a~8 kJ/mol (Rafik et al. 2007). At -40C:
    # ratio = exp(8000/8.314*(1/233.15-1/298.15)) ~ exp(8000/8.314*0.000935) ~ exp(0.899) ~ 2.46
    # Threshold > 1.5 confirms positive E_a and meaningful T-dependence.
    # Threshold < 10 confirms it is weaker than battery-class cells (E_a~15-35 kJ/mol).
    assert ratio > 1.5, f"EDLC ESR(-40C)/ESR(25C)={ratio:.2f} must be >1.5"
    assert ratio < 10.0, f"EDLC ESR ratio {ratio:.2f} should be <10 (weaker T-dep than batteries)"


# --- Capacitance increases with temperature ---

def test_capacitance_increases_with_temperature(model):
    """C(T) must increase with temperature (alpha_C > 0)."""
    c_cold = model.predict({"v_cap": 1.35, "current": 0, "temperature": 233.15})["capacitance"]
    c_ref = model.predict({"v_cap": 1.35, "current": 0, "temperature": 298.15})["capacitance"]
    c_hot = model.predict({"v_cap": 1.35, "current": 0, "temperature": 338.15})["capacitance"]
    assert float(c_cold) < float(c_ref) < float(c_hot), "Capacitance must increase with T"


def test_capacitance_at_reference(model):
    c = model.predict({"v_cap": 1.35, "current": 0, "temperature": 298.15})["capacitance"]
    assert abs(float(c) - 3000.0) < 1e-6


# --- Voltage drop at high current ---

def test_terminal_voltage_drops_with_high_current(model):
    """Higher current -> larger I*ESR drop -> lower terminal voltage."""
    v_lo = model.predict({"v_cap": 1.35, "current": 50.0, "temperature": 298.15})["terminal_voltage"]
    v_hi = model.predict({"v_cap": 1.35, "current": 200.0, "temperature": 298.15})["terminal_voltage"]
    assert float(v_hi) < float(v_lo), "Higher current should give lower terminal voltage"


def test_terminal_voltage_lower_at_cold_high_current(model):
    """At high current, cold T -> higher ESR -> lower terminal voltage."""
    v_cold = model.predict({"v_cap": 1.35, "current": 200.0, "temperature": 233.15})["terminal_voltage"]
    v_hot = model.predict({"v_cap": 1.35, "current": 200.0, "temperature": 338.15})["terminal_voltage"]
    assert float(v_cold) < float(v_hot), "V at -40C must be lower than at 65C (high current)"


# --- Heat generation: purely Joule, no entropic term ---

def test_heat_generation_positive_discharge(model):
    q = model.predict({"v_cap": 1.35, "current": 100.0, "temperature": 298.15})["heat_generation"]
    assert float(q) > 0, "Heat generation must be positive during discharge"


def test_heat_generation_equals_i_squared_r(model):
    """EDLC: Q must equal exactly I^2 * ESR(T) (no entropic term)."""
    I = 100.0
    T = 298.15
    result = model.predict({"v_cap": 1.35, "current": I, "temperature": T})
    expected = I**2 * float(result["esr"])
    assert abs(float(result["heat_generation"]) - expected) < 1e-10, \
        "EDLC heat must be purely I^2*ESR (no entropic contribution)"


def test_heat_generation_increases_with_current(model):
    q1 = model.predict({"v_cap": 1.35, "current": 10.0, "temperature": 298.15})["heat_generation"]
    q5 = model.predict({"v_cap": 1.35, "current": 100.0, "temperature": 298.15})["heat_generation"]
    q10 = model.predict({"v_cap": 1.35, "current": 200.0, "temperature": 298.15})["heat_generation"]
    assert float(q5) > float(q1)
    assert float(q10) > float(q5)


# --- Stored energy ---

def test_stored_energy_increases_with_v_cap(model):
    e_lo = model.predict({"v_cap": 0.5, "current": 0, "temperature": 298.15})["stored_energy"]
    e_hi = model.predict({"v_cap": 2.5, "current": 0, "temperature": 298.15})["stored_energy"]
    assert float(e_hi) > float(e_lo), "Stored energy must increase with V_cap"


def test_stored_energy_formula(model):
    """E = 0.5 * C(T) * V_cap^2 at reference."""
    v_cap = 2.0
    result = model.predict({"v_cap": v_cap, "current": 0, "temperature": 298.15})
    expected = 0.5 * float(result["capacitance"]) * v_cap**2
    assert abs(float(result["stored_energy"]) - expected) < 1e-3


# --- SOC ---

def test_soc_at_rated_voltage(model):
    """V_cap = V_max -> SOC = 1."""
    result = model.predict({"v_cap": 2.7, "current": 0, "temperature": 298.15})
    assert abs(float(result["soc"]) - 1.0) < 1e-6


def test_soc_at_zero_voltage(model):
    result = model.predict({"v_cap": 0.0, "current": 0, "temperature": 298.15})
    assert abs(float(result["soc"])) < 1e-6


# --- dvcap_dt ---

def test_dvcap_dt_negative_during_discharge(model):
    """Discharging (I>0) must decrease V_cap."""
    result = model.predict({"v_cap": 1.35, "current": 100.0, "temperature": 298.15})
    assert float(result["dvcap_dt"]) < 0, "V_cap must decrease during discharge"


# --- Edge cases ---

def test_temperature_extremes(model):
    for T in [233.15, 338.15]:
        result = model.predict({"v_cap": 1.35, "current": 100.0, "temperature": T})
        assert np.isfinite(float(result["terminal_voltage"]))
        assert float(result["esr"]) > 0


def test_array_inputs(model):
    v_caps = np.array([0.5, 1.35, 2.5])
    currents = np.array([10.0, 100.0, 200.0])
    temps = np.array([253.15, 298.15, 318.15])
    result = model.predict({"v_cap": v_caps, "current": currents, "temperature": temps})
    assert result["terminal_voltage"].shape == (3,)


# --- Benchmark ---

def test_benchmark_1000_predictions(model):
    v_caps = np.random.uniform(0.1, 2.7, 1000)
    currents = np.random.uniform(-200.0, 200.0, 1000)
    temps = np.random.uniform(233.15, 338.15, 1000)
    start = time.perf_counter()
    model.predict({"v_cap": v_caps, "current": currents, "temperature": temps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
