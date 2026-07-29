"""EC104 — Gas Engine CHP — F1a — Test Suite"""
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
              "eta_electrical", "eta_thermal", "eta_total"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC104"
    assert info["fidelity"] == "F1a"


def test_eta_total_less_than_095(model):
    """Combined efficiency must be < 0.95 (thermodynamic limit for CHP)."""
    plr = np.linspace(0.5, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    assert np.all(r["eta_total"] < 0.95), "eta_total must be < 0.95"


def test_eta_total_in_realistic_range(model):
    """Total efficiency (el + th) should be 0.75-0.90 for a good gas engine CHP."""
    r = model.predict({"part_load_ratio": 1.0})
    eta_tot = float(r["eta_total"])
    assert 0.75 <= eta_tot <= 0.90, f"eta_total={eta_tot:.3f} outside 0.75-0.90"


def test_both_efficiencies_positive(model):
    """Both eta_el and eta_th must be > 0 across all PLRs."""
    plr = np.linspace(0.5, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    assert np.all(r["eta_electrical"] > 0), "eta_el must be positive"
    assert np.all(r["eta_thermal"] > 0), "eta_th must be positive"


def test_fuel_conservation(model):
    """Fuel input must exceed electrical + thermal output (some losses)."""
    plr = np.linspace(0.5, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    useful = r["electrical_power_kw"] + r["thermal_power_kw"]
    assert np.all(r["fuel_input_kw"] > useful), "fuel must exceed P_el + Q_th"


def test_electrical_power_at_full_load(model):
    """At PLR=1.0, electrical output should equal rated power (2 MW)."""
    r = model.predict({"part_load_ratio": 1.0})
    P_el = float(r["electrical_power_kw"])
    assert abs(P_el - 2000.0) < 1.0, f"P_el at PLR=1 = {P_el:.0f} kW, expected 2000 kW"


def test_power_increases_with_plr(model):
    """Both electrical and thermal output increase with PLR."""
    plr = np.linspace(0.5, 1.0, 20)
    r = model.predict({"part_load_ratio": plr})
    assert np.all(np.diff(r["electrical_power_kw"]) > 0)
    assert np.all(np.diff(r["thermal_power_kw"]) > 0)


def test_eta_el_eta_th_sum_equals_total(model):
    """eta_total == eta_el + eta_th by definition."""
    plr = np.linspace(0.5, 1.0, 10)
    r = model.predict({"part_load_ratio": plr})
    assert np.allclose(r["eta_total"], r["eta_electrical"] + r["eta_thermal"], rtol=1e-9)


def test_benchmark(model):
    plr = np.random.uniform(0.5, 1.0, 1000)
    start = time.perf_counter()
    model.predict({"part_load_ratio": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
