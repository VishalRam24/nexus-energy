"""
EC017 -- PSA -- F1b Temperature-Pressure -- Test Suite
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

def test_predict_returns_keys(model):
    r = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75,
                       "feed_pressure_bar": 20.0})
    for k in ["recovery", "product_flow_kg_s", "tail_gas_flow_kg_s",
              "specific_energy_kWh_per_kg", "electric_power_kW",
              "pressure_ratio", "h2_yield_kg_s"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC017"
    assert info["fidelity"] == "F1b"


# --- Pressure effects ---

def test_recovery_increases_with_pressure(model):
    """Higher feed pressure -> better adsorption selectivity -> higher recovery."""
    r_lo = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75, "feed_pressure_bar": 10.0})
    r_hi = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75, "feed_pressure_bar": 40.0})
    assert float(r_hi["recovery"]) > float(r_lo["recovery"]), \
        "Recovery must increase with feed pressure"


def test_specific_energy_decreases_with_pressure(model):
    """
    Higher pressure -> lower specific energy: W = W_nom * (P_ref/P)^0.15.
    Per Sircar & Golden (2000): energy scales inversely with pressure ratio.
    """
    r_lo = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75, "feed_pressure_bar": 10.0})
    r_hi = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75, "feed_pressure_bar": 40.0})
    assert float(r_hi["specific_energy_kWh_per_kg"]) < float(r_lo["specific_energy_kWh_per_kg"]), \
        "Specific energy must decrease with pressure"


# --- Temperature effects ---

def test_recovery_decreases_with_temperature(model):
    """Higher T -> reduced adsorption capacity -> lower H2 recovery."""
    r_cold = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75,
                             "feed_pressure_bar": 20.0, "temperature_K": 263.15})
    r_hot  = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75,
                             "feed_pressure_bar": 20.0, "temperature_K": 333.15})
    assert float(r_cold["recovery"]) > float(r_hot["recovery"]), \
        "Recovery must decrease at higher temperature"


def test_specific_energy_increases_with_temperature(model):
    """Higher T -> more recompression needed -> higher specific energy."""
    r_cold = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75,
                             "feed_pressure_bar": 20.0, "temperature_K": 263.15})
    r_hot  = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75,
                             "feed_pressure_bar": 20.0, "temperature_K": 333.15})
    assert float(r_hot["specific_energy_kWh_per_kg"]) > float(r_cold["specific_energy_kWh_per_kg"]), \
        "Specific energy must increase with temperature"


# --- Mass balance ---

def test_mass_balance(model):
    """Product flow + tail gas flow must equal feed flow."""
    r = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75,
                       "feed_pressure_bar": 20.0, "temperature_K": 298.15})
    total = float(r["product_flow_kg_s"]) + float(r["tail_gas_flow_kg_s"])
    assert abs(total - 0.1) < 1e-10, f"Mass balance violated: {total:.6f} != 0.1"


# --- Recovery and energy bounds ---

def test_recovery_in_range(model):
    """Recovery must be in (0, 1)."""
    for P in [10, 20, 40]:
        r = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75,
                           "feed_pressure_bar": P})
        eta = float(r["recovery"])
        assert 0.0 < eta < 1.0, f"Recovery {eta:.3f} out of range at P={P} bar"


def test_specific_energy_in_physical_range(model):
    """Specific energy must be between 0.5 and 5.0 kWh/kg_H2."""
    for P in [10, 20, 40]:
        r = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75,
                           "feed_pressure_bar": P})
        W = float(r["specific_energy_kWh_per_kg"])
        assert 0.5 <= W <= 5.0, f"Specific energy {W:.3f} kWh/kg out of physical range"


# --- Pressure ratio ---

def test_pressure_ratio_positive(model):
    r = model.predict({"feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75, "feed_pressure_bar": 20.0})
    assert float(r["pressure_ratio"]) > 1.0, "Pressure ratio must be > 1"


# --- Array inputs ---

def test_array_inputs(model):
    """Model handles array inputs."""
    P_arr = np.array([10.0, 20.0, 40.0])
    F_arr = np.array([0.05, 0.1, 0.2])
    y_arr = np.array([0.7, 0.75, 0.8])
    r = model.predict({"feed_flow_kg_s": F_arr, "feed_h2_fraction": y_arr,
                       "feed_pressure_bar": P_arr})
    assert r["recovery"].shape == (3,)


# --- Benchmark ---

def test_benchmark_1000_predictions(model):
    P_arr = np.random.uniform(5.0, 80.0, 1000)
    F_arr = np.random.uniform(0.01, 1.0, 1000)
    y_arr = np.random.uniform(0.5, 0.95, 1000)
    T_arr = np.random.uniform(263.15, 333.15, 1000)
    start = time.perf_counter()
    model.predict({"feed_flow_kg_s": F_arr, "feed_h2_fraction": y_arr,
                   "feed_pressure_bar": P_arr, "temperature_K": T_arr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
