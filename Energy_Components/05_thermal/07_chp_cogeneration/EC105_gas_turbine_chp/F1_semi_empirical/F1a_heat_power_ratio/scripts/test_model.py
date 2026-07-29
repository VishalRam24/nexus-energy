"""EC105 — Gas Turbine CHP — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"part_load_ratio": 1.0})
    for k in ["electrical_power_kw", "thermal_power_kw", "fuel_input_kw",
              "eta_electrical", "eta_thermal", "eta_total", "heat_to_power_ratio"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC105"
    assert info["fidelity"] == "F1a"


def test_eta_total_less_than_095(model):
    plr = np.linspace(0.4, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    assert np.all(r["eta_total"] < 0.95)


def test_eta_total_in_realistic_range(model):
    """Total efficiency at full load should be ~0.75-0.90 for GT CHP."""
    r = model.predict({"part_load_ratio": 1.0})
    eta_tot = float(r["eta_total"])
    assert 0.70 <= eta_tot <= 0.90, f"eta_total={eta_tot:.3f} outside 0.70-0.90"


def test_eta_el_in_realistic_range(model):
    """GT electrical eff at full load 0.28-0.36."""
    r = model.predict({"part_load_ratio": 1.0})
    eta_el = float(r["eta_electrical"])
    assert 0.28 <= eta_el <= 0.36, f"eta_el={eta_el:.3f}"


def test_eta_th_greater_than_eta_el(model):
    """Gas turbine CHP: thermal recovery typically dominates electrical (HPR > 1)."""
    r = model.predict({"part_load_ratio": 1.0})
    assert float(r["eta_thermal"]) > float(r["eta_electrical"])


def test_heat_to_power_ratio_realistic(model):
    """HPR for gas turbine CHP typically ~1.2-2.0 at full load."""
    r = model.predict({"part_load_ratio": 1.0})
    hpr = float(r["heat_to_power_ratio"])
    assert 1.0 <= hpr <= 2.5, f"HPR={hpr:.2f}"


def test_both_efficiencies_positive(model):
    plr = np.linspace(0.4, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    assert np.all(r["eta_electrical"] > 0)
    assert np.all(r["eta_thermal"] > 0)


def test_fuel_conservation(model):
    """Fuel input must exceed P_el + Q_th."""
    plr = np.linspace(0.4, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    useful = r["electrical_power_kw"] + r["thermal_power_kw"]
    assert np.all(r["fuel_input_kw"] >= useful - 1e-9)


def test_electrical_power_at_full_load(model):
    r = model.predict({"part_load_ratio": 1.0})
    P_el = float(r["electrical_power_kw"])
    assert abs(P_el - 5000.0) < 1.0


def test_power_increases_with_plr(model):
    plr = np.linspace(0.4, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    assert np.all(np.diff(r["electrical_power_kw"]) > 0)
    assert np.all(np.diff(r["thermal_power_kw"]) > 0)


def test_eta_sum_equals_total(model):
    plr = np.linspace(0.4, 1.0, 10)
    r = model.predict({"part_load_ratio": plr})
    assert np.allclose(r["eta_total"], r["eta_electrical"] + r["eta_thermal"], rtol=1e-9)


def test_fuel_positive_at_any_load(model):
    plr = np.linspace(0.4, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    assert np.all(r["fuel_input_kw"] > 0)


def test_benchmark(model):
    plr = np.random.uniform(0.4, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"part_load_ratio": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
