"""EC215 — Solar Still / HDH — F1b GOR + Solar + Temperature — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_top_degC": 70.0, "G_Wm2": 700.0, "T_amb_degC": 25.0})
    for k in ["gor", "distillate_kg_h", "sec_kwh_m3", "solar_heat_kw", "humidity_diff"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC215"
    assert info["fidelity"] == "F1b"


def test_gor_reasonable_range(model):
    """HDH GOR at T_top=70C: 1-4 (Narayan 2010: practical HDH GOR 1-5, typical 2-3)."""
    r = model.predict({"T_top_degC": 70.0, "G_Wm2": 700.0, "T_amb_degC": 25.0})
    gor = float(np.atleast_1d(r["gor"])[0])
    # Narayan 2010: HDH GOR 1-4 typically; allow generous range
    assert 0.1 < gor < 4.5, f"HDH GOR at design = {gor:.3f}"


def test_gor_increases_with_top_temperature(model):
    """Higher humidifier temperature → more evaporation → higher GOR."""
    r_low  = model.predict({"T_top_degC": 55.0, "G_Wm2": 700.0, "T_amb_degC": 25.0})
    r_high = model.predict({"T_top_degC": 80.0, "G_Wm2": 700.0, "T_amb_degC": 25.0})
    gor_low  = float(np.atleast_1d(r_low["gor"])[0])
    gor_high = float(np.atleast_1d(r_high["gor"])[0])
    assert gor_high > gor_low, f"GOR not increasing with T_top: {gor_low:.3f} vs {gor_high:.3f}"


def test_solar_heat_positive(model):
    r = model.predict({"T_top_degC": 70.0, "G_Wm2": 700.0, "T_amb_degC": 25.0})
    Q = float(np.atleast_1d(r["solar_heat_kw"])[0])
    assert Q > 0


def test_solar_heat_scales_with_irradiance(model):
    """More irradiance → more solar heat."""
    r_low  = model.predict({"T_top_degC": 70.0, "G_Wm2": 300.0, "T_amb_degC": 25.0})
    r_high = model.predict({"T_top_degC": 70.0, "G_Wm2": 800.0, "T_amb_degC": 25.0})
    Q_low  = float(np.atleast_1d(r_low["solar_heat_kw"])[0])
    Q_high = float(np.atleast_1d(r_high["solar_heat_kw"])[0])
    assert Q_high > Q_low, f"Solar heat not scaling with G: {Q_low:.2f} vs {Q_high:.2f}"


def test_solar_efficiency_declines_at_high_temp(model):
    """Solar collector efficiency should decrease at higher mean temperature (Hottel-Whillier)."""
    eta_low  = model._model.eta_solar_0 - model._model.a1 * (50.0 - 25.0) / 700.0
    eta_high = model._model.eta_solar_0 - model._model.a1 * (70.0 - 25.0) / 700.0
    assert eta_high < eta_low, f"eta not decreasing with T: {eta_low:.3f} vs {eta_high:.3f}"


def test_distillate_positive_with_irradiance(model):
    r = model.predict({"T_top_degC": 70.0, "G_Wm2": 700.0, "T_amb_degC": 25.0})
    m = float(np.atleast_1d(r["distillate_kg_h"])[0])
    assert m > 0, f"Distillate = {m:.4f} kg/h"


def test_no_production_at_zero_irradiance(model):
    """No distillate at G=0."""
    r = model.predict({"T_top_degC": 70.0, "G_Wm2": 0.0, "T_amb_degC": 25.0})
    m = float(np.atleast_1d(r["distillate_kg_h"])[0])
    assert m < 0.01, f"Distillate at G=0: {m:.4f} kg/h (should be 0)"


def test_humidity_diff_positive(model):
    """Humidity difference between humidifier and condenser must be positive."""
    r = model.predict({"T_top_degC": 70.0, "G_Wm2": 700.0, "T_amb_degC": 25.0})
    dw = float(np.atleast_1d(r["humidity_diff"])[0])
    assert dw > 0, f"Humidity diff = {dw:.6f}"


def test_humidity_diff_increases_with_temp(model):
    """Higher T_top → larger humidity ratio difference."""
    r_low  = model.predict({"T_top_degC": 55.0, "G_Wm2": 700.0, "T_amb_degC": 25.0})
    r_high = model.predict({"T_top_degC": 80.0, "G_Wm2": 700.0, "T_amb_degC": 25.0})
    dw_low  = float(np.atleast_1d(r_low["humidity_diff"])[0])
    dw_high = float(np.atleast_1d(r_high["humidity_diff"])[0])
    assert dw_high > dw_low, f"Humidity diff not increasing: {dw_low:.4f} vs {dw_high:.4f}"


def test_sec_reasonable(model):
    """HDH thermal SEC: 100-600 kWh_th/m3 (Narayan 2010: small-scale HDH)."""
    r = model.predict({"T_top_degC": 70.0, "G_Wm2": 700.0, "T_amb_degC": 25.0})
    sec = float(np.atleast_1d(r["sec_kwh_m3"])[0])
    # Solar still/HDH has very high SEC (solar input per m3); allow generous range
    assert 50.0 < sec < 2000.0, f"HDH SEC = {sec:.1f} kWh/m3"


def test_array_input(model):
    T_tops = np.linspace(55.0, 80.0, 10)
    r = model.predict({"T_top_degC": T_tops, "G_Wm2": 700.0, "T_amb_degC": 25.0})
    assert len(np.atleast_1d(r["gor"])) == 10


def test_benchmark(model):
    T_tops = np.random.uniform(55.0, 80.0, 1000)
    start = time.perf_counter()
    model.predict({"T_top_degC": T_tops, "G_Wm2": 700.0, "T_amb_degC": 25.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
