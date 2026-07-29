"""
EC021 -- LTO Battery -- F1b SOC-Thermal -- Test Suite

Physics sanity checks for temperature-dependent LTO battery model.
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
    result = model.predict({"soc": 0.5, "current": 1.0, "temperature": 298.15})
    assert isinstance(result, dict)
    for key in ["terminal_voltage", "power", "heat_generation",
                "effective_capacity", "internal_resistance"]:
        assert key in result


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC021"
    assert info["fidelity"] == "F1b"


# --- Arrhenius: R increases at lower temperatures ---

def test_resistance_increases_at_low_temperature(model):
    """Arrhenius: R must increase as temperature decreases."""
    r_cold = model.predict({"soc": 0.5, "current": 0, "temperature": 243.15})["internal_resistance"]
    r_ref = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance"]
    r_hot = model.predict({"soc": 0.5, "current": 0, "temperature": 333.15})["internal_resistance"]
    assert float(r_cold) > float(r_ref), "R at -30C must be > R at 25C"
    assert float(r_ref) > float(r_hot), "R at 25C must be > R at 60C"


def test_resistance_at_reference_equals_R_ref(model):
    """At T_ref, R(T) should equal R_ref."""
    r = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance"]
    assert abs(float(r) - 0.020) < 1e-6, f"R at T_ref should be R_ref=0.020, got {float(r)}"


# --- LTO-specific: better low-T performance than graphite-anode cells ---
# LTO E_a=15000 J/mol -> R(-30C)/R(25C) ~ exp(15000/8.314*(1/243.15 - 1/298.15)) ~ 3.6x
# NMC E_a=25000 J/mol -> R(-30C)/R(25C) ~ 10x
# LTO ratio must be < NMC ratio, confirming lower activation energy.

def test_lto_low_temp_ratio_lower_than_nmc(model):
    """LTO R(-30C)/R(25C) should be lower than NMC equivalent (~10x) due to lower E_a=15kJ/mol."""
    r_cold = float(model.predict({"soc": 0.5, "current": 0, "temperature": 243.15})["internal_resistance"])
    r_ref = float(model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance"])
    ratio = r_cold / r_ref
    # RATIONALE: LTO E_a~15 kJ/mol (He et al. 2013) vs NMC E_a~25 kJ/mol (Ecker et al. 2015).
    # At -30C, LTO R ratio ~3-4x; NMC ~10x. Threshold set at 6x to give margin.
    assert ratio < 6.0, f"LTO R(-30C)/R(25C)={ratio:.2f} should be <6 due to low E_a"


# --- Voltage decreases with temperature at high current ---

def test_voltage_lower_at_cold_high_current(model):
    """At high discharge current, cold temperature -> higher R -> lower voltage."""
    v_cold = model.predict({"soc": 0.5, "current": 10.0, "temperature": 243.15})["terminal_voltage"]
    v_hot = model.predict({"soc": 0.5, "current": 10.0, "temperature": 333.15})["terminal_voltage"]
    assert float(v_cold) < float(v_hot), "Voltage at -30C must be lower than at 60C (high current)"


# --- Heat generation ---

def test_heat_generation_positive_discharge(model):
    """Heat generation must be positive for moderate discharge current."""
    q = model.predict({"soc": 0.5, "current": 5.0, "temperature": 298.15})["heat_generation"]
    assert float(q) > 0, "Heat generation must be positive during discharge"


def test_heat_generation_increases_with_current(model):
    """Heat generation should increase with current magnitude (I^2*R dominates)."""
    q1 = model.predict({"soc": 0.5, "current": 1.0, "temperature": 298.15})["heat_generation"]
    q5 = model.predict({"soc": 0.5, "current": 5.0, "temperature": 298.15})["heat_generation"]
    q10 = model.predict({"soc": 0.5, "current": 10.0, "temperature": 298.15})["heat_generation"]
    assert float(q5) > float(q1), "Q at 5A > Q at 1A"
    assert float(q10) > float(q5), "Q at 10A > Q at 5A"


# --- Capacity increases with temperature ---

def test_capacity_increases_with_temperature(model):
    """Effective capacity must increase with temperature (alpha_c > 0)."""
    c_cold = model.predict({"soc": 0.5, "current": 0, "temperature": 243.15})["effective_capacity"]
    c_ref = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["effective_capacity"]
    c_hot = model.predict({"soc": 0.5, "current": 0, "temperature": 333.15})["effective_capacity"]
    assert float(c_cold) < float(c_ref) < float(c_hot), "Capacity must increase with T"


def test_capacity_at_reference(model):
    """At T_ref, effective capacity should equal C_ref."""
    c = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["effective_capacity"]
    assert abs(float(c) - 2.9) < 1e-6


# --- Edge cases ---

def test_soc_zero(model):
    result = model.predict({"soc": 0.0, "current": 1.0, "temperature": 298.15})
    assert np.isfinite(float(result["terminal_voltage"]))


def test_soc_one(model):
    result = model.predict({"soc": 1.0, "current": 1.0, "temperature": 298.15})
    assert np.isfinite(float(result["terminal_voltage"]))


def test_temperature_extremes(model):
    """Model must handle temperature at range boundaries."""
    for T in [243.15, 333.15]:
        result = model.predict({"soc": 0.5, "current": 1.0, "temperature": T})
        assert np.isfinite(float(result["terminal_voltage"]))
        assert np.isfinite(float(result["internal_resistance"]))
        assert float(result["internal_resistance"]) > 0


def test_zero_current_voltage_equals_ocv(model):
    """At zero current, terminal voltage should equal OCV."""
    result = model.predict({"soc": 0.5, "current": 0.0, "temperature": 298.15})
    assert abs(float(result["terminal_voltage"]) - float(result["ocv"])) < 1e-10


def test_array_inputs(model):
    """Model must handle array inputs."""
    socs = np.array([0.2, 0.5, 0.8])
    currents = np.array([1.0, 2.0, 3.0])
    temps = np.array([263.15, 298.15, 313.15])
    result = model.predict({"soc": socs, "current": currents, "temperature": temps})
    assert result["terminal_voltage"].shape == (3,)


# --- Benchmark ---

def test_benchmark_1000_predictions(model):
    """1000 predictions should complete in < 1s."""
    socs = np.random.uniform(0.0, 1.0, 1000)
    currents = np.random.uniform(-10.0, 10.0, 1000)
    temps = np.random.uniform(243.15, 333.15, 1000)
    start = time.perf_counter()
    model.predict({"soc": socs, "current": currents, "temperature": temps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
