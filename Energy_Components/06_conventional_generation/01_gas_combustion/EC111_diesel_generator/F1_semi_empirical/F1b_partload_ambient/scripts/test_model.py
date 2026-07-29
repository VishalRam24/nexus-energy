"""EC111 -- Diesel Generator -- F1b Part-Load + Ambient -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"PLR": 1.0})
    for k in ["efficiency", "power_output_kw", "fuel_consumption_l_h",
              "sfc_g_kwh", "exhaust_temp_degC"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC111"
    assert info["fidelity"] == "F1b"


# --- Efficiency ---

def test_efficiency_peaks_near_full_load(model):
    """Efficiency should be highest near full load (Willans line property)."""
    PLR = np.linspace(0.25, 1.0, 100)
    r = model.predict({"PLR": PLR})
    eta = r["efficiency"]
    idx_max = np.argmax(eta)
    assert PLR[idx_max] > 0.7, f"Peak eta at PLR={PLR[idx_max]:.2f}"


def test_efficiency_drops_at_part_load(model):
    """Efficiency at minimum PLR must be less than at full load."""
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.25})
    assert float(r_part["efficiency"]) < float(r_full["efficiency"])


def test_efficiency_bounded(model):
    """Efficiency in (0, 0.50]."""
    PLR = np.linspace(0.25, 1.0, 50)
    r = model.predict({"PLR": PLR})
    assert np.all(r["efficiency"] > 0)
    assert np.all(r["efficiency"] <= 0.50)


# --- SFC ---

def test_sfc_increases_at_part_load(model):
    """SFC should increase (worsen) at part-load."""
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.3})
    assert float(r_part["sfc_g_kwh"]) > float(r_full["sfc_g_kwh"])


def test_sfc_at_rated_reasonable(model):
    """SFC at rated should be approximately 200-250 g/kWh for diesel genset."""
    r = model.predict({"PLR": 1.0})
    sfc = float(r["sfc_g_kwh"])
    assert 180 < sfc < 300, f"SFC at rated = {sfc:.0f} g/kWh"


# --- Altitude derating ---

def test_altitude_derating_reduces_power(model):
    """Higher altitude should reduce available power."""
    r_sea = model.predict({"PLR": 1.0, "altitude_m": 0.0})
    r_alt = model.predict({"PLR": 1.0, "altitude_m": 3000.0})
    assert float(r_alt["power_output_kw"]) < float(r_sea["power_output_kw"])


def test_no_derating_below_1000m(model):
    """No altitude derating below 1000m."""
    r_0 = model.predict({"PLR": 1.0, "altitude_m": 0.0})
    r_500 = model.predict({"PLR": 1.0, "altitude_m": 500.0})
    assert abs(float(r_0["power_output_kw"]) - float(r_500["power_output_kw"])) < 0.01


def test_altitude_derating_magnitude(model):
    """At 4000m: 3000m above 1000m threshold -> 3000/300 * 3.5% = 35% derating."""
    r = model.predict({"PLR": 1.0, "altitude_m": 4000.0})
    P = float(r["power_output_kw"])
    expected = 500.0 * (1.0 - 0.35)
    assert abs(P - expected) / expected < 0.05, f"P={P:.0f}, expected ~{expected:.0f}"


# --- Temperature derating ---

def test_temp_derating_above_40c(model):
    """Power should decrease above 40C ambient."""
    r_cool = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    r_hot  = model.predict({"PLR": 1.0, "T_ambient": 50.0})
    assert float(r_hot["power_output_kw"]) < float(r_cool["power_output_kw"])


def test_no_temp_derating_below_40c(model):
    """No temperature derating below 40C."""
    r_25 = model.predict({"PLR": 1.0, "T_ambient": 25.0})
    r_35 = model.predict({"PLR": 1.0, "T_ambient": 35.0})
    assert abs(float(r_25["power_output_kw"]) - float(r_35["power_output_kw"])) < 0.01


# --- Exhaust temperature ---

def test_exhaust_temp_increases_with_load(model):
    """Exhaust temperature should increase with load."""
    r_part = model.predict({"PLR": 0.3})
    r_full = model.predict({"PLR": 1.0})
    assert float(r_full["exhaust_temp_degC"]) > float(r_part["exhaust_temp_degC"])


# --- Edge cases ---

def test_minimum_plr(model):
    r = model.predict({"PLR": 0.25})
    assert float(r["efficiency"]) > 0
    assert float(r["power_output_kw"]) > 0


def test_combined_derating(model):
    """Combined altitude + temperature derating should compound."""
    r = model.predict({"PLR": 1.0, "T_ambient": 50.0, "altitude_m": 4000.0})
    P = float(r["power_output_kw"])
    # altitude: 35% loss, temperature: 10*0.5%=5% loss => factor ~ 0.65 * 0.95 = 0.6175
    assert P < 500.0 * 0.70  # significant derating


# --- Benchmark ---

def test_benchmark(model):
    PLR = np.random.uniform(0.25, 1.0, 1000)
    T = np.random.uniform(-20, 55, 1000)
    alt = np.random.uniform(0, 4000, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLR, "T_ambient": T, "altitude_m": alt})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
