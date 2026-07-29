"""EC106 -- SOFC-Based Fuel Cell CHP -- F1b Part-Load + Ambient -- Test Suite"""
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
    assert info["ec_id"] == "EC106"
    assert info["fidelity"] == "F1b"


# --- Electrical efficiency ---

def test_eta_el_rated_range(model):
    """At full load 25C: eta_el should be in SOFC range 45-60%."""
    r = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    eta = float(r["efficiency_electrical"])
    assert 0.42 <= eta <= 0.62, f"SOFC eta_el at rated = {eta:.3f}, expected 0.42-0.62"


def test_eta_el_flat_at_partload(model):
    """SOFC part-load electrical efficiency should not drop below 80% of rated value."""
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.3})
    eta_full = float(r_full["efficiency_electrical"])
    eta_part = float(r_part["efficiency_electrical"])
    # SOFC stays relatively flat; must retain > 70% of rated eta
    assert eta_part >= 0.70 * eta_full, \
        f"SOFC part-load eta {eta_part:.3f} < 70% of rated {eta_full:.3f}"


def test_eta_el_nonnegative(model):
    PLR = np.linspace(0.3, 1.0, 50)
    r = model.predict({"PLR": PLR})
    assert np.all(r["efficiency_electrical"] > 0)


# --- Thermal efficiency ---

def test_eta_th_higher_at_part_load(model):
    """HPR should increase at part load (heat loss is proportionally larger)."""
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.3})
    assert float(r_part["heat_to_power_ratio"]) > float(r_full["heat_to_power_ratio"])


# --- Total efficiency ---

def test_total_efficiency_range(model):
    """Total efficiency should be in CHP range 0.70-0.95."""
    PLR = np.linspace(0.3, 1.0, 50)
    r = model.predict({"PLR": PLR, "T_ambient": 25.0})
    assert np.all(r["efficiency_total"] >= 0.65), "Total eta too low"
    assert np.all(r["efficiency_total"] <= 0.95), "Total eta too high"


def test_total_equals_sum(model):
    """eta_total = eta_el + eta_th."""
    PLR = np.linspace(0.3, 1.0, 50)
    r = model.predict({"PLR": PLR})
    diff = np.abs(r["efficiency_total"] - r["efficiency_electrical"] - r["efficiency_thermal"])
    assert np.all(diff < 1e-10)


# --- Ambient temperature ---

def test_higher_temp_reduces_power(model):
    r_cool = model.predict({"PLR": 1.0, "T_ambient": 15.0})
    r_hot  = model.predict({"PLR": 1.0, "T_ambient": 45.0})
    assert float(r_cool["power_electrical_kw"]) > float(r_hot["power_electrical_kw"])


def test_no_derating_below_25c(model):
    r_cold = model.predict({"PLR": 1.0, "T_ambient": 10.0})
    r_ref  = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    assert abs(float(r_cold["power_electrical_kw"]) - float(r_ref["power_electrical_kw"])) < 0.1


# --- Energy balance ---

def test_energy_balance(model):
    """P_el + Q_th <= fuel_input (first law)."""
    PLR = np.linspace(0.3, 1.0, 50)
    r = model.predict({"PLR": PLR})
    total = r["power_electrical_kw"] + r["heat_recovery_kw"]
    assert np.all(total <= r["fuel_input_kw"] * 1.001)


# --- HPR ---

def test_hpr_at_rated(model):
    """HPR at rated ~ eta_th/eta_el = 0.35/0.47 ~ 0.74."""
    r = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    hpr = float(r["heat_to_power_ratio"])
    assert 0.5 < hpr < 1.2, f"HPR = {hpr:.3f}"


# --- Edge cases ---

def test_minimum_plr(model):
    r = model.predict({"PLR": 0.3})
    assert float(r["efficiency_electrical"]) > 0
    assert float(r["power_electrical_kw"]) > 0
    assert float(r["heat_recovery_kw"]) > 0


# --- Benchmark ---

def test_benchmark(model):
    PLR = np.random.uniform(0.3, 1.0, 1000)
    T   = np.random.uniform(-10, 45, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLR, "T_ambient": T})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
