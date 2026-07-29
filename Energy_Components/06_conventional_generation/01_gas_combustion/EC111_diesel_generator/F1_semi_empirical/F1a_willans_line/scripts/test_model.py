"""EC111 — Diesel Generator — F1a Willans Line — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"power_output_kw": 250.0})
    for k in ["fuel_rate_lph", "sfc_gkwh", "efficiency", "co2_emissions_kgh"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC111"
    assert info["fidelity"] == "F1a"


def test_no_load_fuel_positive(model):
    """Fuel consumption must be positive even at no-load (engine idling)."""
    r = model.predict({"power_output_kw": 0.0})
    assert float(r["fuel_rate_lph"]) > 0.0, "No-load fuel must be > 0"


def test_efficiency_below_45_percent(model):
    """Diesel gen efficiency must be below 45% across load range."""
    P = np.linspace(125, 500, 50)  # above PLR_min
    r = model.predict({"power_output_kw": P})
    assert np.all(r["efficiency"] < 0.45), "Efficiency must be < 45%"


def test_efficiency_zero_at_no_load(model):
    """Efficiency is 0 at P=0 (no useful work output)."""
    r = model.predict({"power_output_kw": 0.0})
    assert float(r["efficiency"]) == 0.0


def test_sfc_decreases_toward_rated(model):
    """SFC should decrease (improve) as load increases from part-load to rated."""
    P = np.linspace(200, 500, 20)
    r = model.predict({"power_output_kw": P})
    sfc = r["sfc_gkwh"]
    valid = ~np.isnan(sfc)
    assert np.all(np.diff(sfc[valid]) < 0), "SFC must decrease with increasing load"


def test_fuel_rate_linear_with_power(model):
    """Willans line: fuel_rate = a + b*P, so residuals from linear fit should be near zero."""
    P = np.linspace(125, 500, 50)
    r = model.predict({"power_output_kw": P})
    fr = r["fuel_rate_lph"]
    coeffs = np.polyfit(P, fr, 1)
    residuals = fr - np.polyval(coeffs, P)
    assert np.max(np.abs(residuals)) < 1e-6, "Fuel rate must be exactly linear"


def test_co2_proportional_to_fuel(model):
    """CO2 = fuel_rate * co2_factor, so co2/fuel_rate must be constant."""
    P = np.linspace(125, 500, 20)
    r = model.predict({"power_output_kw": P})
    ratio = r["co2_emissions_kgh"] / r["fuel_rate_lph"]
    assert np.allclose(ratio, ratio[0], rtol=1e-6), "CO2/fuel ratio must be constant"


def test_ambient_derating(model):
    """Higher ambient temperature should reduce available rated power (more fuel at same P)."""
    r_hot = model.predict({"power_output_kw": 400.0, "ambient_temp_c": 50.0})
    r_cold = model.predict({"power_output_kw": 400.0, "ambient_temp_c": 25.0})
    # At high ambient, power is derated — same requested P is a higher fraction of derated rated
    # fuel rate should increase with ambient temp due to higher effective PLR
    assert float(r_hot["fuel_rate_lph"]) >= float(r_cold["fuel_rate_lph"])


def test_rated_sfc_approximately_correct(model):
    """SFC at rated load should be close to the nameplate 210 g/kWh."""
    r = model.predict({"power_output_kw": 500.0})
    sfc = float(r["sfc_gkwh"])
    assert 180 < sfc < 250, f"SFC at rated load = {sfc:.1f} g/kWh (expected ~210)"


def test_benchmark(model):
    P = np.random.uniform(125, 500, 1000)
    start = time.perf_counter()
    model.predict({"power_output_kw": P})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
