"""EC002 — SOFC — F1a Polarization Curve — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"current_density": 0.5, "temperature": 800.0})
    for k in ["cell_voltage", "stack_voltage", "power_density", "stack_power_kw", "efficiency"]:
        assert k in r


def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC002"


def test_voltage_decreases_with_current(model):
    """Cell voltage must decrease monotonically with current density."""
    j = np.linspace(0.05, 1.5, 100)
    r = model.predict({"current_density": j, "temperature": 800.0})
    diffs = np.diff(r["cell_voltage"])
    assert np.all(diffs <= 0.0), "Cell voltage must be monotonically decreasing with j"


def test_voltage_below_nernst(model):
    """Cell voltage must always be below E_Nernst."""
    j = np.linspace(0.01, 1.8, 100)
    r = model.predict({"current_density": j, "temperature": 800.0})
    m = model._model
    T_K = 800.0 + 273.15
    EN = float(m.E_nernst(T_K))
    assert np.all(r["cell_voltage"] <= EN + 1e-6), \
        f"Cell voltage exceeds Nernst potential {EN:.4f} V"


def test_efficiency_below_one(model):
    """Thermodynamic efficiency must be < 1."""
    j = np.linspace(0.05, 1.8, 50)
    r = model.predict({"current_density": j, "temperature": 800.0})
    assert np.all(r["efficiency"] < 1.0), "Efficiency must be < 1"


def test_efficiency_positive(model):
    j = np.linspace(0.1, 1.5, 50)
    r = model.predict({"current_density": j, "temperature": 800.0})
    assert np.all(r["efficiency"] > 0.0)


def test_power_density_positive(model):
    j = np.linspace(0.1, 1.5, 50)
    r = model.predict({"current_density": j, "temperature": 800.0})
    assert np.all(r["power_density"] >= 0.0)


def test_high_j_near_jL_voltage_drops(model):
    """Near j_L the concentration loss should cause steep voltage drop."""
    m = model._model
    j_near_limit = m.j_L * 0.95
    j_mid        = m.j_L * 0.5
    r_limit = model.predict({"current_density": j_near_limit, "temperature": 800.0})
    r_mid   = model.predict({"current_density": j_mid,        "temperature": 800.0})
    assert float(r_limit["cell_voltage"]) < float(r_mid["cell_voltage"]), \
        "Voltage must be lower near j_L than at mid-range"


def test_higher_temperature_improves_voltage(model):
    """Higher operating temperature reduces activation losses, improving cell voltage at mid j."""
    j = 0.5
    r_hot  = model.predict({"current_density": j, "temperature": 900.0})
    r_cold = model.predict({"current_density": j, "temperature": 650.0})
    # At mid-range j, activation loss reduction dominates for SOFC
    assert float(r_hot["cell_voltage"]) > float(r_cold["cell_voltage"]), \
        "Higher T should improve voltage (lower activation overpotential)"


def test_stack_voltage_consistency(model):
    """stack_voltage = N_cells * cell_voltage."""
    r = model.predict({"current_density": 0.6, "temperature": 800.0})
    N = model._model.N
    assert abs(float(r["stack_voltage"]) - N * float(r["cell_voltage"])) < 1e-6


def test_zero_current_open_circuit(model):
    """At j=0 (or very small j), voltage should approach E_Nernst."""
    r = model.predict({"current_density": 1e-6, "temperature": 800.0})
    m = model._model
    EN = float(m.E_nernst(800.0 + 273.15))
    V  = float(r["cell_voltage"])
    assert abs(V - EN) < 0.05, f"OCV={V:.4f} V, expected ~{EN:.4f} V"


def test_benchmark(model):
    j     = np.random.uniform(0.05, 1.8, 1000)
    T     = np.random.uniform(600.0, 1000.0, 1000)
    start = time.perf_counter()
    model.predict({"current_density": j, "temperature": T})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
