"""EC101 -- CCGT -- F1b Part-Load + Ambient -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"PLR": 1.0})
    for k in ["efficiency_combined", "efficiency_gt", "efficiency_st",
              "power_output_kw", "heat_rate_kj_kwh", "exhaust_temp_K"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC101"
    assert info["fidelity"] == "F1b"


# --- Combined efficiency ---

def test_combined_efficiency_at_iso_near_rated(model):
    """At ISO, PLR=1.0, combined eta should be ~62% within 5%."""
    r = model.predict({"PLR": 1.0, "T_ambient": 288.15})
    eta = float(r["efficiency_combined"])
    assert abs(eta - 0.62) / 0.62 < 0.05, f"eta_cc={eta:.4f}, expected ~0.62"


def test_combined_efficiency_at_50pct_load(model):
    """At 50% load, combined eta should be ~55% (typical CCGT)."""
    r = model.predict({"PLR": 0.5, "T_ambient": 288.15})
    eta = float(r["efficiency_combined"])
    assert 0.45 < eta < 0.62, f"eta_cc at 50% = {eta:.4f}"


def test_combined_greater_than_gt_alone(model):
    """Combined efficiency must exceed GT efficiency (bottoming cycle adds power)."""
    PLR = np.linspace(0.4, 1.0, 50)
    r = model.predict({"PLR": PLR, "T_ambient": 288.15})
    assert np.all(r["efficiency_combined"] > r["efficiency_gt"])


def test_efficiency_drops_at_part_load(model):
    """Combined efficiency at minimum PLR < full load."""
    r_full = model.predict({"PLR": 1.0, "T_ambient": 288.15})
    r_part = model.predict({"PLR": 0.4, "T_ambient": 288.15})
    assert float(r_part["efficiency_combined"]) < float(r_full["efficiency_combined"])


def test_efficiency_below_carnot(model):
    """Carnot limit for gas turbine: 1 - T_cold/T_hot.
    T_hot ~ 1600K (TIT), T_cold ~ 288K -> eta_carnot ~ 82%.
    Combined eta must be well below this.
    """
    r = model.predict({"PLR": 1.0, "T_ambient": 288.15})
    eta = float(r["efficiency_combined"])
    assert eta < 0.70


# --- Ambient effects ---

def test_higher_temp_lowers_efficiency(model):
    r_cold = model.predict({"PLR": 1.0, "T_ambient": 263.15})
    r_hot  = model.predict({"PLR": 1.0, "T_ambient": 313.15})
    assert float(r_cold["efficiency_combined"]) > float(r_hot["efficiency_combined"])


def test_higher_temp_lowers_power(model):
    r_cold = model.predict({"PLR": 1.0, "T_ambient": 263.15})
    r_hot  = model.predict({"PLR": 1.0, "T_ambient": 313.15})
    assert float(r_cold["power_output_kw"]) > float(r_hot["power_output_kw"])


def test_higher_pressure_increases_power(model):
    r_low  = model.predict({"PLR": 1.0, "T_ambient": 288.15, "P_ambient": 85.0})
    r_high = model.predict({"PLR": 1.0, "T_ambient": 288.15, "P_ambient": 101.325})
    assert float(r_high["power_output_kw"]) > float(r_low["power_output_kw"])


# --- Heat rate ---

def test_heat_rate_increases_at_part_load(model):
    r_full = model.predict({"PLR": 1.0, "T_ambient": 288.15})
    r_part = model.predict({"PLR": 0.5, "T_ambient": 288.15})
    assert float(r_part["heat_rate_kj_kwh"]) > float(r_full["heat_rate_kj_kwh"])


def test_heat_rate_consistent_with_efficiency(model):
    r = model.predict({"PLR": 0.8, "T_ambient": 288.15})
    hr = float(r["heat_rate_kj_kwh"])
    eta = float(r["efficiency_combined"])
    assert abs(hr - 3600 / eta) < 1.0


# --- Power output ---

def test_power_at_iso_near_rated(model):
    r = model.predict({"PLR": 1.0, "T_ambient": 288.15, "P_ambient": 101.325})
    P = float(r["power_output_kw"])
    assert abs(P - 571000) / 571000 < 0.05, f"P={P:.0f} kW, expected ~571000"


# --- Edge cases ---

def test_minimum_plr(model):
    r = model.predict({"PLR": 0.4, "T_ambient": 288.15})
    assert float(r["efficiency_combined"]) > 0
    assert float(r["power_output_kw"]) > 0


def test_extreme_hot(model):
    r = model.predict({"PLR": 1.0, "T_ambient": 323.15})
    assert float(r["power_output_kw"]) < 571000


# --- Benchmark ---

def test_benchmark(model):
    PLR = np.random.uniform(0.4, 1.0, 1000)
    T = np.random.uniform(260, 320, 1000)
    start = time.perf_counter()
    model.predict({"PLR": PLR, "T_ambient": T})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
