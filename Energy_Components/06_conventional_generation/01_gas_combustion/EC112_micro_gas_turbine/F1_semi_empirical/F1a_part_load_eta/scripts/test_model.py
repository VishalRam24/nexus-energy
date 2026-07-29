"""EC112 — Micro Gas Turbine — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 15.0})
    for k in ["electrical_power_kw", "fuel_input_kw", "eta_electrical",
              "gas_mass_flow_kgs", "gas_volume_flow_m3h", "heat_rate_kjkwh"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC112"


def test_eta_realistic_full_load(model):
    """Recuperated microturbine eta at ISO full load: 0.25-0.33."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 15.0})
    eta = float(r["eta_electrical"])
    assert 0.24 <= eta <= 0.35, f"eta_el={eta:.3f}"


def test_eta_drops_at_low_plr(model):
    r_full = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 15.0})
    r_part = model.predict({"part_load_ratio": 0.3, "ambient_temp_c": 15.0})
    assert float(r_part["eta_electrical"]) < float(r_full["eta_electrical"])


def test_eta_drops_at_high_ambient(model):
    r_cool = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 5.0})
    r_hot = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 40.0})
    assert float(r_hot["eta_electrical"]) < float(r_cool["eta_electrical"])


def test_full_load_power(model):
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 15.0})
    assert abs(float(r["electrical_power_kw"]) - 200.0) < 1.0


def test_eta_below_40_percent(model):
    plr = np.linspace(0.3, 1.0, 30)
    T = np.linspace(-20, 50, 30)
    r = model.predict({"part_load_ratio": plr, "ambient_temp_c": T})
    assert np.all(r["eta_electrical"] < 0.40)


def test_fuel_greater_than_power(model):
    plr = np.linspace(0.3, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    assert np.all(r["fuel_input_kw"] > r["electrical_power_kw"])


def test_fuel_positive_at_any_load(model):
    plr = np.linspace(0.3, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    assert np.all(r["fuel_input_kw"] > 0)
    assert np.all(r["gas_mass_flow_kgs"] > 0)


def test_heat_rate_consistent_with_eta(model):
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 15.0})
    hr_calc = 3600.0 / float(r["eta_electrical"])
    assert abs(hr_calc - float(r["heat_rate_kjkwh"])) < 1.0


def test_benchmark(model):
    plr = np.random.uniform(0.3, 1.0, 1000)
    T = np.random.uniform(-20, 50, 1000)
    start = time.perf_counter()
    model.predict({"part_load_ratio": plr, "ambient_temp_c": T})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
