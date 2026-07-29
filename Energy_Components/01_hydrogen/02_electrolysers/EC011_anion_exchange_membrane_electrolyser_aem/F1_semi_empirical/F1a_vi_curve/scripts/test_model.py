"""EC011 — AEM Electrolyser — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"current_density": 5000.0, "temperature": 60.0})
    for k in ["cell_voltage", "stack_voltage", "hydrogen_rate_mols", "power_kw", "efficiency"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC011"
    assert info["fidelity"] == "F1a"


def test_voltage_above_e_rev(model):
    """V_cell must always exceed E_rev (thermodynamic minimum)."""
    j_arr = np.linspace(500, 15000, 50)
    r = model.predict({"current_density": j_arr, "temperature": 60.0})
    E_rev_60 = 1.229 - 0.0009 * (333.15 - 298.15)
    assert np.all(r["cell_voltage"] > E_rev_60)


def test_voltage_increases_with_current_density(model):
    j_arr = np.linspace(500, 15000, 100)
    r = model.predict({"current_density": j_arr, "temperature": 60.0})
    assert np.all(np.diff(r["cell_voltage"]) > 0)


def test_voltage_decreases_with_temperature(model):
    """Higher T => lower E_rev and lower ASR => lower V_cell."""
    j = 5000.0
    r_low = model.predict({"current_density": j, "temperature": 40.0})
    r_high = model.predict({"current_density": j, "temperature": 70.0})
    assert float(r_high["cell_voltage"]) < float(r_low["cell_voltage"])


def test_h2_proportional_to_current(model):
    """H2 rate is exactly linear in current density (constant Faraday efficiency)."""
    j1, j2 = 2000.0, 4000.0
    r1 = model.predict({"current_density": j1, "temperature": 60.0})
    r2 = model.predict({"current_density": j2, "temperature": 60.0})
    ratio = float(r2["hydrogen_rate_mols"]) / float(r1["hydrogen_rate_mols"])
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_efficiency_below_unity(model):
    j_arr = np.linspace(500, 15000, 50)
    r = model.predict({"current_density": j_arr, "temperature": 60.0})
    assert np.all(r["efficiency"] <= 1.0)
    assert np.all(r["efficiency"] >= 0.0)


def test_efficiency_reasonable(model):
    """AEM at 0.5 A/cm2 should be ~60-80% LHV efficient."""
    r = model.predict({"current_density": 5000.0, "temperature": 60.0})
    eta = float(r["efficiency"])
    assert 0.5 < eta < 0.9, f"Got eta={eta:.2f}"


def test_zero_current_density(model):
    r = model.predict({"current_density": 0.0, "temperature": 60.0})
    assert float(r["hydrogen_rate_mols"]) == pytest.approx(0.0, abs=1e-12)
    assert float(r["power_kw"]) == pytest.approx(0.0, abs=1e-9)


def test_stack_voltage_consistency(model):
    """Stack voltage = N_cells * cell voltage."""
    r = model.predict({"current_density": 5000.0, "temperature": 60.0})
    ratio = float(r["stack_voltage"]) / float(r["cell_voltage"])
    assert ratio == pytest.approx(10.0, rel=1e-6)


def test_voltage_in_realistic_range(model):
    """AEM cell voltage at 0.5 A/cm2 should be 1.7-2.2 V (literature range)."""
    r = model.predict({"current_density": 5000.0, "temperature": 60.0})
    V = float(r["cell_voltage"])
    assert 1.6 < V < 2.4, f"Got V={V:.3f} V"


def test_benchmark(model):
    j_arr = np.random.uniform(100, 15000, 1000)
    T_arr = np.random.uniform(30, 80, 1000)
    start = time.perf_counter()
    model.predict({"current_density": j_arr, "temperature": T_arr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
