"""EC112 -- Micro Gas Turbine -- F1b Part-Load + Ambient + Altitude -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

ISO_T = 288.15   # 15 degC in Kelvin
ISO_P = 101.325  # kPa


@pytest.fixture
def model():
    return ComponentModel()


# --- Output key checks ---

def test_predict_keys(model):
    r = model.predict({"PLR": 1.0})
    for k in ["efficiency_electrical", "power_electrical_kw", "fuel_input_kw",
              "gas_mass_flow_kgs", "gas_volume_flow_m3h", "heat_rate_kj_kwh",
              "f_power_ambient", "f_eta_temperature", "f_altitude"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC112"
    assert info["fidelity"] == "F1b"


# --- Electrical efficiency ---

def test_eta_el_at_iso(model):
    """At ISO full load, eta_el should be near 0.30 (within 5%)."""
    r = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "P_ambient": ISO_P})
    eta = float(r["efficiency_electrical"])
    assert abs(eta - 0.30) / 0.30 < 0.05, f"eta_el = {eta:.4f}, expected ~0.30"


def test_eta_el_drops_at_part_load(model):
    """Efficiency at minimum PLR should be lower than at full load."""
    r_full = model.predict({"PLR": 1.0, "T_ambient": ISO_T})
    r_part = model.predict({"PLR": 0.3, "T_ambient": ISO_T})
    assert float(r_part["efficiency_electrical"]) < float(r_full["efficiency_electrical"])


def test_strong_ambient_T_sensitivity(model):
    """
    Micro gas turbine has strong T sensitivity ~0.01/K (1%/K).
    At 40C (313.15K) vs ISO 15C (288.15K): delta T = 25K.
    Expected efficiency drop: ~25% relative (0.01/K * 25K = 0.25 fraction).
    Test: efficiency at 40C should be at most 80% of ISO efficiency.
    """
    r_iso = model.predict({"PLR": 1.0, "T_ambient": ISO_T})
    r_hot = model.predict({"PLR": 1.0, "T_ambient": 313.15})
    eta_iso = float(r_iso["efficiency_electrical"])
    eta_hot = float(r_hot["efficiency_electrical"])
    # ratio should be <= 0.80 (25% drop at 25K above ISO)
    ratio = eta_hot / eta_iso
    assert ratio <= 0.80, \
        f"Efficiency ratio at 40C vs ISO = {ratio:.3f}, expected <= 0.80 (strong sensitivity)"


def test_higher_T_reduces_eta_monotonically(model):
    """Efficiency should decrease monotonically with increasing temperature."""
    T_arr = np.array([ISO_T, ISO_T + 5, ISO_T + 10, ISO_T + 20])
    eta_arr = np.array([
        float(model.predict({"PLR": 1.0, "T_ambient": T})["efficiency_electrical"])
        for T in T_arr
    ])
    assert np.all(np.diff(eta_arr) < 0), "Efficiency should decrease with temperature"


def test_eta_el_bounded(model):
    PLR = np.linspace(0.3, 1.0, 50)
    T = np.linspace(263.15, 313.15, 50)
    r = model.predict({"PLR": PLR, "T_ambient": T})
    assert np.all(r["efficiency_electrical"] > 0)
    assert np.all(r["efficiency_electrical"] <= 0.40)


# --- Power output ---

def test_power_at_iso_rated(model):
    """At ISO full load, power should be near P_rated (within 5%)."""
    r = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "P_ambient": ISO_P, "altitude_m": 0.0})
    P = float(r["power_electrical_kw"])
    assert abs(P - 200.0) / 200.0 < 0.05, f"P = {P:.1f} kW, expected ~200"


def test_higher_T_reduces_power(model):
    """Higher ambient T reduces air density -> lower power."""
    r_cold = model.predict({"PLR": 1.0, "T_ambient": 263.15})   # -10C
    r_hot  = model.predict({"PLR": 1.0, "T_ambient": 313.15})   # 40C
    assert float(r_cold["power_electrical_kw"]) > float(r_hot["power_electrical_kw"])


def test_altitude_reduces_power(model):
    """Higher altitude reduces power (lower air density)."""
    r_sea = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "altitude_m": 0.0})
    r_alt = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "altitude_m": 1000.0})
    assert float(r_alt["power_electrical_kw"]) < float(r_sea["power_electrical_kw"])


def test_altitude_derating_1pct_per_100m(model):
    """At 1000m, power should be ~10% lower (1%/100m)."""
    r_sea = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "altitude_m": 0.0})
    r_alt = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "altitude_m": 1000.0})
    P_sea = float(r_sea["power_electrical_kw"])
    P_alt = float(r_alt["power_electrical_kw"])
    derating_pct = (P_sea - P_alt) / P_sea * 100.0
    assert abs(derating_pct - 10.0) < 1.5, f"Derating = {derating_pct:.2f}%, expected ~10%"


def test_f_eta_temperature_at_iso(model):
    """f_eta_temperature should be 1.0 at ISO reference temperature."""
    r = model.predict({"PLR": 1.0, "T_ambient": ISO_T})
    f = float(r["f_eta_temperature"])
    assert abs(f - 1.0) < 1e-9


def test_f_altitude_at_sea_level(model):
    """f_altitude should be 1.0 at sea level."""
    r = model.predict({"PLR": 1.0, "altitude_m": 0.0})
    assert abs(float(r["f_altitude"]) - 1.0) < 1e-9


def test_f_power_ambient_at_iso(model):
    """f_power_ambient should be 1.0 at ISO conditions."""
    r = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "P_ambient": ISO_P})
    f = float(r["f_power_ambient"])
    assert abs(f - 1.0) < 0.001


# --- Energy balance ---

def test_energy_balance(model):
    """P_el <= fuel_input."""
    PLR = np.linspace(0.3, 1.0, 50)
    r = model.predict({"PLR": PLR, "T_ambient": ISO_T})
    assert np.all(r["power_electrical_kw"] <= r["fuel_input_kw"] * 1.001)


def test_heat_rate_consistent_with_eta(model):
    """Heat rate = 3600 / eta_el."""
    r = model.predict({"PLR": 0.8, "T_ambient": ISO_T})
    hr = float(r["heat_rate_kj_kwh"])
    eta = float(r["efficiency_electrical"])
    assert abs(hr - 3600.0 / eta) < 1.0


def test_heat_rate_worsens_at_part_load(model):
    r_full = model.predict({"PLR": 1.0, "T_ambient": ISO_T})
    r_part = model.predict({"PLR": 0.3, "T_ambient": ISO_T})
    assert float(r_part["heat_rate_kj_kwh"]) > float(r_full["heat_rate_kj_kwh"])


# --- Edge cases ---

def test_minimum_plr(model):
    r = model.predict({"PLR": 0.3, "T_ambient": ISO_T})
    assert float(r["efficiency_electrical"]) > 0
    assert float(r["power_electrical_kw"]) > 0


def test_cold_boost(model):
    """Very cold ambient (-20C=253.15K) should give higher power than ISO (air density)."""
    r_cold = model.predict({"PLR": 1.0, "T_ambient": 253.15, "P_ambient": ISO_P, "altitude_m": 0.0})
    assert float(r_cold["power_electrical_kw"]) > 200.0


# --- Benchmark ---

def test_benchmark(model):
    PLR = np.random.uniform(0.3, 1.0, 1000)
    T = np.random.uniform(263.15, 313.15, 1000)
    alt = np.random.uniform(0, 2000, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLR, "T_ambient": T, "altitude_m": alt})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
