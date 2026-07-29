"""EC104 -- Gas Engine CHP -- F1b Part-Load + Ambient -- Test Suite"""
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
    assert info["ec_id"] == "EC104"
    assert info["fidelity"] == "F1b"


# --- Electrical efficiency ---

def test_eta_el_peaks_near_full_load(model):
    PLR = np.linspace(0.5, 1.0, 100)
    r = model.predict({"PLR": PLR})
    eta = r["efficiency_electrical"]
    idx_max = np.argmax(eta)
    assert PLR[idx_max] > 0.7


def test_eta_el_drops_at_part_load(model):
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.5})
    assert float(r_part["efficiency_electrical"]) < float(r_full["efficiency_electrical"])


def test_eta_el_at_rated(model):
    """At full load, 25C: eta_el ~ 0.42."""
    r = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    eta = float(r["efficiency_electrical"])
    assert abs(eta - 0.42) / 0.42 < 0.05


# --- Thermal efficiency ---

def test_eta_th_higher_at_part_load(model):
    """Thermal efficiency should be proportionally higher at part load."""
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.5})
    # eta_th(0.5) / eta_th_rated should be > PLR relative to eta_el
    # i.e., heat-to-power ratio should increase at part load
    assert float(r_part["heat_to_power_ratio"]) > float(r_full["heat_to_power_ratio"])


# --- Total efficiency ---

def test_total_efficiency_range(model):
    """Total efficiency should be in reasonable range for a CHP unit.
    At part-load (PLR=0.5), total can drop to ~60% due to no-load losses;
    at full load it should be ~85%.
    """
    PLR = np.linspace(0.5, 1.0, 50)
    r = model.predict({"PLR": PLR, "T_ambient": 25.0})
    assert np.all(r["efficiency_total"] >= 0.55), "Total eta too low"
    assert np.all(r["efficiency_total"] <= 0.95), "Total eta too high"
    # At full load, total eta should be > 0.80
    r_fl = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    assert float(r_fl["efficiency_total"]) > 0.80


def test_total_equals_sum(model):
    """eta_total = eta_el + eta_th."""
    PLR = np.linspace(0.5, 1.0, 50)
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
    PLR = np.linspace(0.5, 1.0, 50)
    r = model.predict({"PLR": PLR})
    total = r["power_electrical_kw"] + r["heat_recovery_kw"]
    assert np.all(total <= r["fuel_input_kw"] * 1.001)  # small tolerance


# --- Heat-to-power ratio ---

def test_hpr_at_rated(model):
    """HPR at rated ~ eta_th/eta_el = 0.43/0.42 ~ 1.024."""
    r = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    hpr = float(r["heat_to_power_ratio"])
    assert 0.8 < hpr < 1.5, f"HPR = {hpr:.3f}"


def test_hpr_increases_at_part_load(model):
    """HPR should increase at part load (more heat relative to electricity)."""
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.5})
    assert float(r_part["heat_to_power_ratio"]) > float(r_full["heat_to_power_ratio"])


# --- Edge cases ---

def test_minimum_plr(model):
    r = model.predict({"PLR": 0.5})
    assert float(r["efficiency_electrical"]) > 0
    assert float(r["power_electrical_kw"]) > 0
    assert float(r["heat_recovery_kw"]) > 0


# --- Benchmark ---

def test_benchmark(model):
    PLR = np.random.uniform(0.5, 1.0, 1000)
    T = np.random.uniform(-10, 45, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLR, "T_ambient": T})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
