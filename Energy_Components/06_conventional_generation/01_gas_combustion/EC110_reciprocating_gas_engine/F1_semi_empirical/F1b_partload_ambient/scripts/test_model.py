"""EC110 -- Reciprocating Gas Engine -- F1b Part-Load + Altitude + Ambient -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Output key checks ---

def test_predict_keys(model):
    r = model.predict({"PLR": 1.0})
    for k in ["efficiency_electrical", "power_electrical_kw", "fuel_input_kw",
              "gas_mass_flow_kgs", "gas_volume_flow_m3h", "sfc_g_kwh",
              "heat_rate_kj_kwh", "f_temperature", "f_altitude"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC110"
    assert info["fidelity"] == "F1b"


# --- Electrical efficiency ---

def test_eta_el_at_rated(model):
    """At PLR=1.0, eta_el should be near 0.42 (within 5%)."""
    r = model.predict({"PLR": 1.0, "T_ambient": 25.0, "altitude_m": 0.0})
    eta = float(r["efficiency_electrical"])
    assert abs(eta - 0.42) / 0.42 < 0.05, f"eta_el = {eta:.4f}, expected ~0.42"


def test_eta_el_drops_at_part_load(model):
    """Efficiency should be lower at minimum PLR than at full load."""
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.5})
    assert float(r_part["efficiency_electrical"]) < float(r_full["efficiency_electrical"])


def test_eta_el_peaks_near_full_load(model):
    """Efficiency maximum should occur near PLR=1.0 for lean-burn gas engine."""
    PLR = np.linspace(0.5, 1.0, 100)
    r = model.predict({"PLR": PLR})
    eta = r["efficiency_electrical"]
    idx_max = np.argmax(eta)
    assert PLR[idx_max] > 0.75, f"Peak eta at PLR={PLR[idx_max]:.2f}, expected > 0.75"


def test_eta_el_positive_bounded(model):
    PLR = np.linspace(0.5, 1.0, 50)
    r = model.predict({"PLR": PLR})
    assert np.all(r["efficiency_electrical"] > 0)
    assert np.all(r["efficiency_electrical"] <= 0.50)


# --- Temperature derating ---

def test_no_derating_below_T_ref(model):
    """No temperature derating below 25 degC (reference temperature)."""
    r_cold = model.predict({"PLR": 1.0, "T_ambient": 10.0})
    r_ref  = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    # Power should be identical (no cold boost for reciprocating engines)
    assert abs(float(r_cold["power_electrical_kw"]) - float(r_ref["power_electrical_kw"])) < 0.1


def test_higher_T_reduces_power(model):
    """Power should decrease above T_ref."""
    r_ref = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    r_hot = model.predict({"PLR": 1.0, "T_ambient": 45.0})
    assert float(r_hot["power_electrical_kw"]) < float(r_ref["power_electrical_kw"])


def test_temperature_derating_magnitude(model):
    """At 45C (20K above 25C ref), derating should be ~6% (0.3%/K * 20K)."""
    r_ref = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    r_hot = model.predict({"PLR": 1.0, "T_ambient": 45.0})
    P_ref = float(r_ref["power_electrical_kw"])
    P_hot = float(r_hot["power_electrical_kw"])
    derating_pct = (P_ref - P_hot) / P_ref * 100.0
    assert abs(derating_pct - 6.0) < 1.0, f"Derating = {derating_pct:.2f}%, expected ~6%"


def test_f_temperature_factor(model):
    """f_temperature should be 1.0 at or below T_ref."""
    r = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    assert abs(float(r["f_temperature"]) - 1.0) < 1e-9


# --- Altitude derating ---

def test_altitude_reduces_power(model):
    """Higher altitude reduces power (lower air pressure/density)."""
    r_sea = model.predict({"PLR": 1.0, "T_ambient": 25.0, "altitude_m": 0.0})
    r_alt = model.predict({"PLR": 1.0, "T_ambient": 25.0, "altitude_m": 1000.0})
    assert float(r_alt["power_electrical_kw"]) < float(r_sea["power_electrical_kw"])


def test_altitude_derating_magnitude(model):
    """At 1000m altitude, derating should be ~9% (0.9%/100m * 10 * 100m)."""
    r_sea = model.predict({"PLR": 1.0, "T_ambient": 25.0, "altitude_m": 0.0})
    r_alt = model.predict({"PLR": 1.0, "T_ambient": 25.0, "altitude_m": 1000.0})
    P_sea = float(r_sea["power_electrical_kw"])
    P_alt = float(r_alt["power_electrical_kw"])
    derating_pct = (P_sea - P_alt) / P_sea * 100.0
    assert abs(derating_pct - 9.0) < 1.0, f"Altitude derating = {derating_pct:.2f}%, expected ~9%"


def test_f_altitude_at_sea_level(model):
    """f_altitude at sea level should be 1.0."""
    r = model.predict({"PLR": 1.0, "altitude_m": 0.0})
    assert abs(float(r["f_altitude"]) - 1.0) < 1e-9


def test_altitude_and_temp_derating_combined(model):
    """Combined derating (hot + high altitude) should be worse than either alone."""
    r_sea_cool = model.predict({"PLR": 1.0, "T_ambient": 25.0, "altitude_m": 0.0})
    r_hot_alt  = model.predict({"PLR": 1.0, "T_ambient": 45.0, "altitude_m": 1500.0})
    P_sea = float(r_sea_cool["power_electrical_kw"])
    P_hot_alt = float(r_hot_alt["power_electrical_kw"])
    assert P_hot_alt < P_sea


# --- Energy balance ---

def test_energy_balance(model):
    """P_el <= fuel_input (first law, since efficiency < 1)."""
    PLR = np.linspace(0.5, 1.0, 50)
    r = model.predict({"PLR": PLR})
    assert np.all(r["power_electrical_kw"] <= r["fuel_input_kw"] * 1.001)


def test_sfc_consistent_with_eta(model):
    """
    SFC [g/kWh] should be consistent with eta_el.
    SFC * eta_el * LHV [kJ/g] = 3600 kJ/kWh
    LHV = 50 MJ/kg = 50 kJ/g
    """
    PLR = np.linspace(0.5, 1.0, 20)
    r = model.predict({"PLR": PLR})
    sfc = r["sfc_g_kwh"]
    eta = r["efficiency_electrical"]
    lhv_kj_per_g = 50.0  # MJ/kg = kJ/g
    # SFC * eta * LHV should equal 3600
    product = sfc * eta * lhv_kj_per_g
    assert np.allclose(product, 3600.0, rtol=0.01), \
        f"SFC check failed, max deviation: {np.max(np.abs(product - 3600)):.2f}"


def test_heat_rate_consistent_with_eta(model):
    """Heat rate = 3600 / eta_el."""
    r = model.predict({"PLR": 0.8})
    hr = float(r["heat_rate_kj_kwh"])
    eta = float(r["efficiency_electrical"])
    assert abs(hr - 3600.0 / eta) < 1.0


# --- Edge cases ---

def test_minimum_plr(model):
    r = model.predict({"PLR": 0.5})
    assert float(r["efficiency_electrical"]) > 0
    assert float(r["power_electrical_kw"]) > 0
    assert float(r["gas_mass_flow_kgs"]) > 0


def test_maximum_altitude(model):
    """At 3000m altitude, engine should still produce positive power."""
    r = model.predict({"PLR": 1.0, "T_ambient": 25.0, "altitude_m": 3000.0})
    assert float(r["power_electrical_kw"]) > 0


# --- Benchmark ---

def test_benchmark(model):
    PLR = np.random.uniform(0.5, 1.0, 1000)
    T = np.random.uniform(-10, 45, 1000)
    alt = np.random.uniform(0, 2000, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLR, "T_ambient": T, "altitude_m": alt})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
