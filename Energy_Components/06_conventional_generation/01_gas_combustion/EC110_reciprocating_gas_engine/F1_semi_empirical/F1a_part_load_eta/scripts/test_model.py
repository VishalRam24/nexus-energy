"""EC110 — Reciprocating Gas Engine — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"part_load_ratio": 1.0})
    for k in ["electrical_power_kw", "fuel_input_kw", "eta_electrical",
              "gas_mass_flow_kgs", "gas_volume_flow_m3h", "sfc_gkwh"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC110"


def test_eta_realistic_full_load(model):
    """eta_el for modern lean-burn NG engine: 0.38-0.45."""
    r = model.predict({"part_load_ratio": 1.0})
    eta = float(r["eta_electrical"])
    assert 0.38 <= eta <= 0.46, f"eta_el={eta:.3f}"


def test_eta_drops_at_part_load(model):
    r_full = model.predict({"part_load_ratio": 1.0})
    r_part = model.predict({"part_load_ratio": 0.5})
    assert float(r_part["eta_electrical"]) < float(r_full["eta_electrical"])


def test_full_load_power(model):
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 25.0})
    assert abs(float(r["electrical_power_kw"]) - 1000.0) < 1.0


def test_ambient_derating(model):
    """Higher ambient T -> lower power output."""
    r_cool = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 20.0})
    r_hot = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 45.0})
    assert float(r_hot["electrical_power_kw"]) < float(r_cool["electrical_power_kw"])


def test_fuel_greater_than_power(model):
    plr = np.linspace(0.5, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    assert np.all(r["fuel_input_kw"] > r["electrical_power_kw"])


def test_fuel_positive_at_any_load(model):
    plr = np.linspace(0.5, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    assert np.all(r["fuel_input_kw"] > 0)
    assert np.all(r["gas_mass_flow_kgs"] > 0)


def test_sfc_realistic(model):
    """SFC for NG engine ~150-200 g/kWh at full load."""
    r = model.predict({"part_load_ratio": 1.0})
    sfc = float(r["sfc_gkwh"])
    assert 130.0 <= sfc <= 220.0, f"SFC={sfc:.1f}"


def test_eta_below_50_percent(model):
    plr = np.linspace(0.5, 1.0, 30)
    r = model.predict({"part_load_ratio": plr})
    assert np.all(r["eta_electrical"] < 0.50)


def test_power_increases_with_plr(model):
    plr = np.linspace(0.5, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    assert np.all(np.diff(r["electrical_power_kw"]) > 0)


def test_benchmark(model):
    plr = np.random.uniform(0.5, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"part_load_ratio": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
