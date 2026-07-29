"""EC109 -- Simple Cycle Gas Turbine -- F1b Part-Load + Ambient -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Output key checks ---

def test_predict_keys(model):
    r = model.predict({"PLR": 1.0, "T_ambient": 288.15})
    for k in ["efficiency", "power_output_kw", "fuel_flow_kg_s",
              "exhaust_temp_K", "heat_rate_kj_kwh"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC109"
    assert info["fidelity"] == "F1b"


# --- Efficiency behaviour ---

def test_efficiency_peaks_near_rated_load(model):
    """Efficiency should peak near PLR=1.0 (or close), not at minimum PLR."""
    PLR = np.linspace(0.3, 1.0, 100)
    r = model.predict({"PLR": PLR, "T_ambient": 288.15})
    eta = r["efficiency"]
    idx_max = np.argmax(eta)
    assert PLR[idx_max] > 0.7, f"Peak efficiency at PLR={PLR[idx_max]:.2f}, expected >0.7"


def test_efficiency_drops_at_part_load(model):
    """Efficiency at PLR=0.3 should be lower than at PLR=1.0."""
    r_full = model.predict({"PLR": 1.0, "T_ambient": 288.15})
    r_part = model.predict({"PLR": 0.3, "T_ambient": 288.15})
    assert float(r_part["efficiency"]) < float(r_full["efficiency"])


def test_efficiency_bounded(model):
    """Efficiency must be in (0, 0.50] across all valid conditions."""
    PLR = np.linspace(0.3, 1.0, 50)
    T = np.linspace(253, 318, 50)
    r = model.predict({"PLR": PLR, "T_ambient": T})
    assert np.all(r["efficiency"] > 0)
    assert np.all(r["efficiency"] <= 0.50)


# --- Ambient corrections ---

def test_higher_temp_lowers_power(model):
    """Higher ambient temperature should reduce power output at same PLR."""
    r_cold = model.predict({"PLR": 1.0, "T_ambient": 263.15})   # -10C
    r_hot  = model.predict({"PLR": 1.0, "T_ambient": 313.15})   # 40C
    assert float(r_cold["power_output_kw"]) > float(r_hot["power_output_kw"])


def test_higher_temp_lowers_efficiency(model):
    """Higher ambient temperature should reduce efficiency."""
    r_cold = model.predict({"PLR": 1.0, "T_ambient": 263.15})
    r_hot  = model.predict({"PLR": 1.0, "T_ambient": 313.15})
    assert float(r_cold["efficiency"]) > float(r_hot["efficiency"])


def test_higher_pressure_increases_power(model):
    """Higher ambient pressure should increase power output."""
    r_low  = model.predict({"PLR": 1.0, "T_ambient": 288.15, "P_ambient": 85.0})
    r_high = model.predict({"PLR": 1.0, "T_ambient": 288.15, "P_ambient": 101.325})
    assert float(r_high["power_output_kw"]) > float(r_low["power_output_kw"])


def test_iso_conditions_near_rated(model):
    """At ISO conditions (15C, 101.325kPa, PLR=1), power ~ rated and eta ~ eta_rated."""
    r = model.predict({"PLR": 1.0, "T_ambient": 288.15, "P_ambient": 101.325})
    P_kw = float(r["power_output_kw"])
    eta  = float(r["efficiency"])
    # Power should be ~43 MW (43000 kW) within 5%
    assert abs(P_kw - 43000) / 43000 < 0.05, f"Power = {P_kw:.0f} kW, expected ~43000"
    # Efficiency should be ~0.41 within 5%
    assert abs(eta - 0.41) / 0.41 < 0.05, f"eta = {eta:.4f}, expected ~0.41"


# --- Heat rate ---

def test_heat_rate_increases_at_part_load(model):
    """Heat rate (kJ/kWh) should increase (worsen) at part-load."""
    r_full = model.predict({"PLR": 1.0, "T_ambient": 288.15})
    r_part = model.predict({"PLR": 0.4, "T_ambient": 288.15})
    assert float(r_part["heat_rate_kj_kwh"]) > float(r_full["heat_rate_kj_kwh"])


def test_heat_rate_consistent_with_efficiency(model):
    """HR = 3600 / eta."""
    r = model.predict({"PLR": 0.8, "T_ambient": 288.15})
    hr = float(r["heat_rate_kj_kwh"])
    eta = float(r["efficiency"])
    assert abs(hr - 3600 / eta) < 1.0


# --- Exhaust temperature ---

def test_exhaust_temp_higher_at_part_load(model):
    """Exhaust temperature slightly higher at part load (reduced expansion ratio)."""
    r_full = model.predict({"PLR": 1.0, "T_ambient": 288.15})
    r_part = model.predict({"PLR": 0.5, "T_ambient": 288.15})
    assert float(r_part["exhaust_temp_K"]) >= float(r_full["exhaust_temp_K"])


# --- Edge cases ---

def test_minimum_plr(model):
    """Model should produce valid results at minimum PLR."""
    r = model.predict({"PLR": 0.3, "T_ambient": 288.15})
    assert float(r["efficiency"]) > 0
    assert float(r["power_output_kw"]) > 0
    assert float(r["fuel_flow_kg_s"]) > 0


def test_extreme_cold(model):
    """Very cold ambient (-30C) should give high power but still valid efficiency."""
    r = model.predict({"PLR": 1.0, "T_ambient": 243.15, "P_ambient": 101.325})
    assert float(r["efficiency"]) <= 0.50
    assert float(r["power_output_kw"]) > 43000  # more than rated (cold boost)


def test_extreme_hot(model):
    """Very hot ambient (50C) should give reduced power."""
    r = model.predict({"PLR": 1.0, "T_ambient": 323.15, "P_ambient": 101.325})
    assert float(r["power_output_kw"]) < 43000  # less than rated


# --- Benchmark ---

def test_benchmark(model):
    PLR = np.random.uniform(0.3, 1.0, 1000)
    T = np.random.uniform(253, 318, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLR, "T_ambient": T})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
