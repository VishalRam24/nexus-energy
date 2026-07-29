"""EC107 -- Micro-CHP Stirling Engine -- F1b Part-Load + Ambient -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"PLR": 1.0})
    for k in ["efficiency_electrical", "efficiency_thermal", "efficiency_total",
              "power_electrical_kw", "heat_recovery_kw", "fuel_input_kw",
              "heat_to_power_ratio"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC107"
    assert info["fidelity"] == "F1b"


# --- Electrical efficiency ---

def test_eta_el_rated_range(model):
    """At full load 25C: Stirling eta_el in range 8-20%."""
    r = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    eta = float(r["efficiency_electrical"])
    assert 0.08 <= eta <= 0.20, f"Stirling eta_el = {eta:.3f}, expected 0.08-0.20"


def test_eta_el_drops_at_part_load(model):
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.4})
    assert float(r_part["efficiency_electrical"]) < float(r_full["efficiency_electrical"])


# --- Thermal efficiency ---

def test_eta_th_high(model):
    """Stirling thermal efficiency should be high (>0.70) at rated."""
    r = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    assert float(r["efficiency_thermal"]) > 0.70


def test_hpr_high_for_stirling(model):
    """Stirling HPR should be > 4 at rated (heat-dominant CHP)."""
    r = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    hpr = float(r["heat_to_power_ratio"])
    assert hpr > 4.0, f"Stirling HPR = {hpr:.2f}, expected > 4"


# --- Total efficiency ---

def test_total_efficiency_range(model):
    """Total CHP efficiency should be 0.80-0.97."""
    PLR = np.linspace(0.4, 1.0, 50)
    r = model.predict({"PLR": PLR, "T_ambient": 25.0})
    assert np.all(r["efficiency_total"] >= 0.75)
    assert np.all(r["efficiency_total"] <= 0.97)


def test_total_equals_sum(model):
    """eta_total = eta_el + eta_th."""
    PLR = np.linspace(0.4, 1.0, 50)
    r = model.predict({"PLR": PLR})
    diff = np.abs(r["efficiency_total"] - r["efficiency_electrical"] - r["efficiency_thermal"])
    assert np.all(diff < 1e-10)


# --- Ambient temperature ---

def test_higher_temp_reduces_power(model):
    r_cool = model.predict({"PLR": 1.0, "T_ambient": 10.0})
    r_hot  = model.predict({"PLR": 1.0, "T_ambient": 40.0})
    assert float(r_cool["power_electrical_kw"]) > float(r_hot["power_electrical_kw"])


def test_no_derating_below_25c(model):
    r_cold = model.predict({"PLR": 1.0, "T_ambient": 10.0})
    r_ref  = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    assert abs(float(r_cold["power_electrical_kw"]) - float(r_ref["power_electrical_kw"])) < 0.01


# --- Energy balance ---

def test_energy_balance(model):
    """P_el + Q_th <= fuel_input."""
    PLR = np.linspace(0.4, 1.0, 50)
    r = model.predict({"PLR": PLR})
    total = r["power_electrical_kw"] + r["heat_recovery_kw"]
    assert np.all(total <= r["fuel_input_kw"] * 1.001)


# --- Edge cases ---

def test_minimum_plr(model):
    r = model.predict({"PLR": 0.4})
    assert float(r["efficiency_electrical"]) > 0
    assert float(r["power_electrical_kw"]) > 0
    assert float(r["heat_recovery_kw"]) > 0


# --- Benchmark ---

def test_benchmark(model):
    PLR = np.random.uniform(0.4, 1.0, 1000)
    T   = np.random.uniform(-10, 45, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLR, "T_ambient": T})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
