"""
EC043 -- Hybrid Supercapacitor -- F1b RC-Thermal -- Test Suite
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
    r = model.predict({"v_cap": 3.0, "current": 50.0, "temperature": 298.15})
    for k in ["terminal_voltage_V", "power_W", "heat_W", "esr_Ohm",
              "capacitance_F", "soc", "stored_energy_J", "dvcap_dt_V_s"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC043"
    assert model.get_info()["fidelity"] == "F1b"


# --- Arrhenius ESR ---

def test_esr_increases_at_low_T(model):
    """ESR must increase at lower temperatures (Arrhenius)."""
    r_cold = float(model.predict({"v_cap": 3.0, "current": 0, "temperature": 233.15})["esr_Ohm"])
    r_ref  = float(model.predict({"v_cap": 3.0, "current": 0, "temperature": 298.15})["esr_Ohm"])
    r_hot  = float(model.predict({"v_cap": 3.0, "current": 0, "temperature": 333.15})["esr_Ohm"])
    assert r_cold > r_ref > r_hot, "ESR must decrease with increasing T"


def test_esr_at_T_ref(model):
    """ESR at T_ref must equal ESR_ref."""
    esr = float(model.predict({"v_cap": 3.0, "current": 0, "temperature": 298.15})["esr_Ohm"])
    assert abs(esr - 0.001) < 1e-9, f"ESR at T_ref should be 0.001 Ohm, got {esr:.6f}"


# --- Capacitance ---

def test_capacitance_increases_with_T(model):
    """Capacitance must increase with temperature."""
    c_cold = float(model.predict({"v_cap": 3.0, "current": 0, "temperature": 233.15})["capacitance_F"])
    c_ref  = float(model.predict({"v_cap": 3.0, "current": 0, "temperature": 298.15})["capacitance_F"])
    c_hot  = float(model.predict({"v_cap": 3.0, "current": 0, "temperature": 333.15})["capacitance_F"])
    assert c_cold < c_ref < c_hot, "Capacitance must increase with T"


def test_capacitance_at_T_ref(model):
    c = float(model.predict({"v_cap": 3.0, "current": 0, "temperature": 298.15})["capacitance_F"])
    assert abs(c - 3000.0) < 1e-6


# --- Voltage ---

def test_zero_current_terminal_equals_vcap(model):
    """At I=0, terminal voltage should equal V_cap (no ESR drop)."""
    for v in [2.0, 2.5, 3.0, 3.5]:
        r = model.predict({"v_cap": v, "current": 0.0, "temperature": 298.15})
        assert abs(float(r["terminal_voltage_V"]) - v) < 1e-9


def test_voltage_drop_at_high_current(model):
    """High discharge current -> terminal voltage < V_cap."""
    v_cap = 3.0
    r = model.predict({"v_cap": v_cap, "current": 200.0, "temperature": 298.15})
    v_term = float(r["terminal_voltage_V"])
    assert v_term < v_cap, f"V_term={v_term:.3f} must be < V_cap={v_cap} at high current"


def test_terminal_voltage_within_bounds(model):
    """Terminal voltage must stay within [v_min, v_max]."""
    for v in [2.0, 3.0, 3.8]:
        for I in [0, 100, -100]:
            r = model.predict({"v_cap": v, "current": I, "temperature": 298.15})
            V = float(r["terminal_voltage_V"])
            assert 1.8 <= V <= 3.8 + 1e-6, f"V_term={V:.3f} out of [1.8, 3.8] bounds"


# --- SOC ---

def test_soc_at_vmax(model):
    """SOC at V_max = 1.0."""
    soc = float(model.predict({"v_cap": 3.8, "current": 0, "temperature": 298.15})["soc"])
    assert abs(soc - 1.0) < 1e-6, f"SOC at V_max should be 1.0, got {soc:.6f}"


def test_soc_at_vmin(model):
    """SOC at V_min = 0.0."""
    soc = float(model.predict({"v_cap": 1.8, "current": 0, "temperature": 298.15})["soc"])
    assert abs(soc - 0.0) < 1e-6, f"SOC at V_min should be 0.0, got {soc:.6f}"


# --- Heat generation ---

def test_heat_generation_positive(model):
    """Joule heating must be positive for any non-zero current."""
    for I in [10.0, 100.0, -50.0]:
        r = model.predict({"v_cap": 3.0, "current": I, "temperature": 298.15})
        assert float(r["heat_W"]) >= 0, f"Heat must be >= 0 at I={I}"


def test_heat_increases_with_current(model):
    """Q = I^2 * ESR; must increase with |current|."""
    q10  = float(model.predict({"v_cap": 3.0, "current": 10.0, "temperature": 298.15})["heat_W"])
    q100 = float(model.predict({"v_cap": 3.0, "current": 100.0, "temperature": 298.15})["heat_W"])
    assert q100 > q10, "Heat must increase with current magnitude"


# --- Stored energy ---

def test_stored_energy_positive(model):
    for v in [2.0, 3.0, 3.8]:
        E = float(model.predict({"v_cap": v, "current": 0, "temperature": 298.15})["stored_energy_J"])
        assert E > 0, "Stored energy must be positive"


def test_stored_energy_increases_with_voltage(model):
    E_lo = float(model.predict({"v_cap": 2.0, "current": 0, "temperature": 298.15})["stored_energy_J"])
    E_hi = float(model.predict({"v_cap": 3.5, "current": 0, "temperature": 298.15})["stored_energy_J"])
    assert E_hi > E_lo, "Stored energy must increase with voltage"


# --- dVcap/dt ---

def test_dvcap_negative_during_discharge(model):
    """During discharge (I > 0), capacitor voltage decreases."""
    dv = float(model.predict({"v_cap": 3.0, "current": 100.0, "temperature": 298.15})["dvcap_dt_V_s"])
    assert dv < 0, "dV_cap/dt must be negative during discharge"


# --- Array inputs ---

def test_array_inputs(model):
    v_arr = np.array([2.0, 2.5, 3.0, 3.5])
    I_arr = np.array([0, 50.0, 100.0, 200.0])
    T_arr = np.array([253.15, 273.15, 298.15, 333.15])
    r = model.predict({"v_cap": v_arr, "current": I_arr, "temperature": T_arr})
    assert r["terminal_voltage_V"].shape == (4,)


# --- Benchmark ---

def test_benchmark_1000_predictions(model):
    v_arr = np.random.uniform(1.8, 3.8, 1000)
    I_arr = np.random.uniform(-500.0, 500.0, 1000)
    T_arr = np.random.uniform(233.15, 333.15, 1000)
    start = time.perf_counter()
    model.predict({"v_cap": v_arr, "current": I_arr, "temperature": T_arr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
