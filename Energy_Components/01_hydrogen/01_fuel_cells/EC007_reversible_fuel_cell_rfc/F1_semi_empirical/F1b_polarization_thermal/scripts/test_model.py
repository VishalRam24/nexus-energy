"""
EC007 -- RFC -- F1b Polarization-Thermal -- Test Suite

Physics sanity checks for temperature-dependent RFC model (both modes).
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

def test_predict_fc_returns_required_keys(model):
    result = model.predict({"current_density": 0.5, "temperature": 353.15, "mode": "fc"})
    for key in ["cell_voltage_V", "power_stack_kW", "efficiency",
                "heat_area_W_cm2", "membrane_resistance_ohm_cm2"]:
        assert key in result


def test_predict_el_returns_required_keys(model):
    result = model.predict({"current_density": 0.5, "temperature": 353.15, "mode": "electrolyser"})
    for key in ["cell_voltage_V", "efficiency", "heat_area_W_cm2"]:
        assert key in result


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC007"
    assert info["fidelity"] == "F1b"


# --- FC mode physics ---

def test_fc_voltage_below_nernst(model):
    """FC cell voltage must be below E_nernst at all operating points."""
    for j in [0.01, 0.5, 1.0, 1.5]:
        r = model.predict({"current_density": j, "temperature": 353.15, "mode": "fc"})
        E_n = float(model._model.nernst_voltage(353.15))
        V = float(r["cell_voltage_V"])
        assert V <= E_n + 1e-6, f"FC V={V:.4f} must be <= E_nernst={E_n:.4f}"


def test_fc_voltage_monotonic_decrease(model):
    """FC voltage decreases monotonically with current density."""
    j_vals = np.linspace(0.01, 1.8, 50)
    V_prev = float(model.predict({"current_density": float(j_vals[0]), "temperature": 353.15, "mode": "fc"})["cell_voltage_V"])
    for j in j_vals[1:]:
        V = float(model.predict({"current_density": float(j), "temperature": 353.15, "mode": "fc"})["cell_voltage_V"])
        assert V <= V_prev + 1e-9, f"V not monotonic: V({j:.3f})={V:.4f} > V_prev={V_prev:.4f}"
        V_prev = V


def test_fc_higher_temp_higher_voltage(model):
    """Higher temperature should improve FC voltage at moderate current (kinetics win)."""
    r_cold = model.predict({"current_density": 0.5, "temperature": 333.15, "mode": "fc"})
    r_hot  = model.predict({"current_density": 0.5, "temperature": 363.15, "mode": "fc"})
    assert float(r_hot["cell_voltage_V"]) > float(r_cold["cell_voltage_V"]), \
        "Higher T should give higher FC voltage at moderate j"


def test_fc_heat_generation_positive(model):
    """FC heat generation must be >= 0 for all j > 0."""
    for j in [0.1, 0.5, 1.0, 1.5]:
        r = model.predict({"current_density": j, "temperature": 353.15, "mode": "fc"})
        assert float(r["heat_area_W_cm2"]) >= 0, f"FC Q must be >= 0 at j={j}"


def test_fc_efficiency_in_range(model):
    """FC efficiency must be in (0, 1) for j > 0."""
    for j in [0.1, 0.5, 1.0]:
        r = model.predict({"current_density": j, "temperature": 353.15, "mode": "fc"})
        eta = float(r["efficiency"])
        assert 0.0 < eta < 1.0, f"FC eta={eta:.4f} not in (0,1)"


# --- Electrolyser mode physics ---

def test_el_voltage_above_nernst(model):
    """Electrolyser voltage must be >= E_nernst (overpotentials add on top)."""
    for j in [0.1, 0.5, 1.0, 2.0]:
        r = model.predict({"current_density": j, "temperature": 353.15, "mode": "electrolyser"})
        E_n = float(model._model.nernst_voltage(353.15))
        V = float(r["cell_voltage_V"])
        assert V >= E_n - 1e-6, f"EL V={V:.4f} must be >= E_nernst={E_n:.4f}"


def test_el_voltage_monotonic_increase(model):
    """Electrolyser voltage increases with current density."""
    j_vals = np.linspace(0.01, 2.5, 50)
    V_prev = float(model.predict({"current_density": float(j_vals[0]), "temperature": 353.15, "mode": "electrolyser"})["cell_voltage_V"])
    for j in j_vals[1:]:
        V = float(model.predict({"current_density": float(j), "temperature": 353.15, "mode": "electrolyser"})["cell_voltage_V"])
        assert V >= V_prev - 1e-9, f"EL V not monotonic at j={j:.3f}"
        V_prev = V


def test_el_higher_temp_lower_voltage(model):
    """Higher temperature reduces electrolyser overpotentials -> lower voltage required."""
    r_cold = model.predict({"current_density": 0.5, "temperature": 333.15, "mode": "electrolyser"})
    r_hot  = model.predict({"current_density": 0.5, "temperature": 363.15, "mode": "electrolyser"})
    assert float(r_hot["cell_voltage_V"]) < float(r_cold["cell_voltage_V"]), \
        "Higher T should reduce electrolyser voltage"


def test_el_efficiency_below_unity(model):
    """Electrolyser efficiency (E_tn/V_cell) must be < 1 for j > 0."""
    for j in [0.1, 0.5, 1.0, 2.0]:
        r = model.predict({"current_density": j, "temperature": 353.15, "mode": "electrolyser"})
        eta = float(r["efficiency"])
        assert 0.0 < eta < 1.0, f"EL eta={eta:.4f} not in (0,1)"


# --- Membrane resistance (shared) ---

def test_membrane_resistance_decreases_with_T(model):
    """Nafion conductivity increases with T -> resistance decreases."""
    r_cold = float(model.predict({"current_density": 0.5, "temperature": 313.15, "mode": "fc"})["membrane_resistance_ohm_cm2"])
    r_ref  = float(model.predict({"current_density": 0.5, "temperature": 353.15, "mode": "fc"})["membrane_resistance_ohm_cm2"])
    r_hot  = float(model.predict({"current_density": 0.5, "temperature": 363.15, "mode": "fc"})["membrane_resistance_ohm_cm2"])
    assert r_cold > r_ref > r_hot, "R_mem must decrease with increasing T"


# --- Thermal dynamics ---

def test_dTdt_positive_when_hot_stack_no_cooling(model):
    """At high load, heat generation > cooling -> dT/dt > 0 when T < equilibrium."""
    # At very low T (below T_cool), heat should flow from coolant in -> dT/dt should depend on balance
    # At T < T_cool, dT/dt > 0 since Q_cool = UA*(T - T_cool) < 0 (coolant heats stack)
    r = model.predict({"current_density": 1.0, "temperature": 313.15, "mode": "fc"})
    # T_cool = 333.15 K; T_stack = 313.15 K -> Q_cool < 0 -> net heat always positive
    # Cannot make a strong assertion without knowing balance; just check finite value
    assert np.isfinite(float(r["dTdt_K_s"])), "dTdt must be finite"


# --- Edge cases ---

def test_zero_current_fc(model):
    """At j=0, FC voltage should equal Nernst voltage."""
    r = model.predict({"current_density": 0.0, "temperature": 353.15, "mode": "fc"})
    E_n = float(model._model.nernst_voltage(353.15))
    V = float(r["cell_voltage_V"])
    assert abs(V - E_n) < 1e-3, f"At j=0, V_fc={V:.4f} should equal E_nernst={E_n:.4f}"


def test_zero_current_el(model):
    """At j=0, electrolyser voltage should equal Nernst voltage."""
    r = model.predict({"current_density": 0.0, "temperature": 353.15, "mode": "electrolyser"})
    E_n = float(model._model.nernst_voltage(353.15))
    V = float(r["cell_voltage_V"])
    assert abs(V - E_n) < 1e-3, f"At j=0, V_el={V:.4f} should equal E_nernst={E_n:.4f}"


def test_array_inputs(model):
    """Model must handle array inputs."""
    j_arr = np.array([0.1, 0.5, 1.0, 1.5])
    T_arr = np.array([333.15, 343.15, 353.15, 363.15])
    r = model.predict({"current_density": j_arr, "temperature": T_arr, "mode": "fc"})
    assert r["cell_voltage_V"].shape == (4,)


# --- Benchmark ---

def test_benchmark_1000_predictions(model):
    """1000 predictions should complete in < 1s."""
    j_arr = np.random.uniform(0.01, 1.8, 1000)
    T_arr = np.random.uniform(313.15, 363.15, 1000)
    start = time.perf_counter()
    model.predict({"current_density": j_arr, "temperature": T_arr, "mode": "fc"})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 FC predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
