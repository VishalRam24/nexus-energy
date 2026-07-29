"""
EC042 -- Pseudocapacitor -- F1b RC-Thermal -- Test Suite

Physics sanity checks. Key distinctions from EDLC (EC041):
  - Higher E_a (12 kJ/mol vs 8 kJ/mol) -> stronger T-dependence of ESR
  - Faradaic entropic heat term (absent in EDLC)
  - Higher leakage current (faradaic side reactions)
  - Narrower T range (aqueous electrolyte: -30 to 60 degC)

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
    result = model.predict({"v_cap": 0.5, "current": 50.0, "temperature": 298.15})
    for key in ["terminal_voltage", "power", "heat_generation", "esr", "capacitance",
                "soc", "stored_energy", "leakage_current", "dvcap_dt"]:
        assert key in result


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC042"
    assert info["fidelity"] == "F1b"


# --- Arrhenius: ESR increases at lower temperatures ---

def test_esr_increases_at_low_temperature(model):
    """ESR must increase as temperature decreases (Arrhenius)."""
    esr_cold = model.predict({"v_cap": 0.5, "current": 0, "temperature": 243.15})["esr"]
    esr_ref = model.predict({"v_cap": 0.5, "current": 0, "temperature": 298.15})["esr"]
    esr_hot = model.predict({"v_cap": 0.5, "current": 0, "temperature": 333.15})["esr"]
    assert float(esr_cold) > float(esr_ref), "ESR at -30C must be > ESR at 25C"
    assert float(esr_ref) > float(esr_hot), "ESR at 25C must be > ESR at 60C"


def test_esr_at_reference(model):
    esr = model.predict({"v_cap": 0.5, "current": 0, "temperature": 298.15})["esr"]
    assert abs(float(esr) - 0.005) < 1e-8


def test_pseudocap_stronger_t_dep_than_edlc(model):
    """Pseudocap E_a=12 kJ/mol > EDLC E_a=8 kJ/mol -> higher ratio at extreme T."""
    esr_cold = float(model.predict({"v_cap": 0.5, "current": 0, "temperature": 243.15})["esr"])
    esr_ref = float(model.predict({"v_cap": 0.5, "current": 0, "temperature": 298.15})["esr"])
    ratio = esr_cold / esr_ref
    # RATIONALE: Pseudocapacitor E_a=12 kJ/mol (charge-transfer at RuO2 surface,
    # Sugimoto et al. 2006 Electrochim. Acta 52, 1742).
    # At -30C (243K) vs 25C (298K):
    # ratio = exp(12000/8.314*(1/243.15-1/298.15)) ~ exp(12000/8.314*0.000759) ~ exp(1.096) ~ 3.0
    # Threshold >2 confirms pseudocap E_a > EDLC; threshold <12 confirms aqueous system.
    assert ratio > 2.0, f"Pseudocap ESR(-30C)/ESR(25C)={ratio:.2f} must be >2 (E_a=12 kJ/mol)"
    assert ratio < 12.0, f"ESR ratio {ratio:.2f} should be <12 for aqueous pseudocap"


# --- Capacitance increases with temperature ---

def test_capacitance_increases_with_temperature(model):
    c_cold = model.predict({"v_cap": 0.5, "current": 0, "temperature": 243.15})["capacitance"]
    c_ref = model.predict({"v_cap": 0.5, "current": 0, "temperature": 298.15})["capacitance"]
    c_hot = model.predict({"v_cap": 0.5, "current": 0, "temperature": 333.15})["capacitance"]
    assert float(c_cold) < float(c_ref) < float(c_hot), "Capacitance must increase with T"


def test_capacitance_at_reference(model):
    c = model.predict({"v_cap": 0.5, "current": 0, "temperature": 298.15})["capacitance"]
    assert abs(float(c) - 200.0) < 1e-6


# --- Voltage drop ---

def test_terminal_voltage_drops_with_high_current(model):
    v_lo = model.predict({"v_cap": 0.5, "current": 20.0, "temperature": 298.15})["terminal_voltage"]
    v_hi = model.predict({"v_cap": 0.5, "current": 80.0, "temperature": 298.15})["terminal_voltage"]
    assert float(v_hi) < float(v_lo), "Higher current should give lower terminal voltage"


def test_terminal_voltage_lower_at_cold_high_current(model):
    v_cold = model.predict({"v_cap": 0.5, "current": 80.0, "temperature": 243.15})["terminal_voltage"]
    v_hot = model.predict({"v_cap": 0.5, "current": 80.0, "temperature": 333.15})["terminal_voltage"]
    assert float(v_cold) < float(v_hot), "V at -30C must be lower than at 60C (high current)"


# --- Heat generation: Joule + faradaic entropic term ---

def test_heat_generation_positive_discharge(model):
    q = model.predict({"v_cap": 0.5, "current": 50.0, "temperature": 298.15})["heat_generation"]
    assert float(q) > 0, "Heat generation must be positive during discharge"


def test_heat_generation_exceeds_joule_term(model):
    """Pseudocap heat > I^2*ESR alone due to faradaic entropic contribution (dOCV_dT < 0)."""
    I = 50.0
    T = 298.15
    result = model.predict({"v_cap": 0.5, "current": I, "temperature": T})
    q_total = float(result["heat_generation"])
    q_joule = I**2 * float(result["esr"])
    # RATIONALE: For RuO2 proton insertion, dOCV_dT < 0. At discharge (I>0),
    # q_entropic = I * V_cap * |dOCV_dT| > 0 -> total heat > Joule alone.
    # (Conway 1999; Trasatti & Buzzanca 1971). This is the key physics
    # distinguishing pseudocapacitors from ideal EDLC.
    assert q_total > q_joule, "Pseudocap heat must exceed I^2*ESR (faradaic entropic term present)"


def test_heat_generation_increases_with_current(model):
    q1 = model.predict({"v_cap": 0.5, "current": 10.0, "temperature": 298.15})["heat_generation"]
    q5 = model.predict({"v_cap": 0.5, "current": 50.0, "temperature": 298.15})["heat_generation"]
    q10 = model.predict({"v_cap": 0.5, "current": 100.0, "temperature": 298.15})["heat_generation"]
    assert float(q5) > float(q1)
    assert float(q10) > float(q5)


# --- Stored energy ---

def test_stored_energy_increases_with_v_cap(model):
    e_lo = model.predict({"v_cap": 0.1, "current": 0, "temperature": 298.15})["stored_energy"]
    e_hi = model.predict({"v_cap": 0.9, "current": 0, "temperature": 298.15})["stored_energy"]
    assert float(e_hi) > float(e_lo), "Stored energy must increase with V_cap"


def test_stored_energy_formula(model):
    """E = 0.5 * C(T) * V_cap^2."""
    v_cap = 0.8
    result = model.predict({"v_cap": v_cap, "current": 0, "temperature": 298.15})
    expected = 0.5 * float(result["capacitance"]) * v_cap**2
    assert abs(float(result["stored_energy"]) - expected) < 1e-6


# --- SOC ---

def test_soc_at_rated_voltage(model):
    """V_cap = V_max (1.0V) -> SOC = 1."""
    result = model.predict({"v_cap": 1.0, "current": 0, "temperature": 298.15})
    assert abs(float(result["soc"]) - 1.0) < 1e-6


def test_soc_at_zero_voltage(model):
    result = model.predict({"v_cap": 0.0, "current": 0, "temperature": 298.15})
    assert abs(float(result["soc"])) < 1e-6


# --- Leakage current ---

def test_leakage_current_positive_at_nonzero_voltage(model):
    """I_leak = V_cap / R_leak > 0 when V_cap > 0."""
    i_leak = model.predict({"v_cap": 0.5, "current": 0, "temperature": 298.15})["leakage_current"]
    assert float(i_leak) > 0


def test_leakage_current_zero_at_zero_voltage(model):
    i_leak = model.predict({"v_cap": 0.0, "current": 0, "temperature": 298.15})["leakage_current"]
    assert abs(float(i_leak)) < 1e-15


# --- dvcap_dt ---

def test_dvcap_dt_negative_during_discharge(model):
    """Discharging (I>0) must decrease V_cap."""
    result = model.predict({"v_cap": 0.5, "current": 50.0, "temperature": 298.15})
    assert float(result["dvcap_dt"]) < 0, "V_cap must decrease during discharge"


# --- Edge cases ---

def test_temperature_extremes(model):
    for T in [243.15, 333.15]:
        result = model.predict({"v_cap": 0.5, "current": 50.0, "temperature": T})
        assert np.isfinite(float(result["terminal_voltage"]))
        assert float(result["esr"]) > 0


def test_array_inputs(model):
    v_caps = np.array([0.1, 0.5, 0.9])
    currents = np.array([10.0, 50.0, 80.0])
    temps = np.array([253.15, 298.15, 318.15])
    result = model.predict({"v_cap": v_caps, "current": currents, "temperature": temps})
    assert result["terminal_voltage"].shape == (3,)


# --- Benchmark ---

def test_benchmark_1000_predictions(model):
    v_caps = np.random.uniform(0.01, 1.0, 1000)
    currents = np.random.uniform(-100.0, 100.0, 1000)
    temps = np.random.uniform(243.15, 333.15, 1000)
    start = time.perf_counter()
    model.predict({"v_cap": v_caps, "current": currents, "temperature": temps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
