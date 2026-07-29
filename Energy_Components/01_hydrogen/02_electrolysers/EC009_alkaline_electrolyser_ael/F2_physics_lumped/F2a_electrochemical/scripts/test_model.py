"""EC009 -- AEL -- F2a Electrochemical -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"current_density": 3000.0, "dt": 60.0, "duration_s": 60.0})
    for k in ["t", "voltage", "h2_production", "efficiency", "bubble_coverage"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC009"
    assert info["fidelity"] == "F2a"


def test_voltage_increases_with_current(model):
    """Cell voltage must increase with current density (electrolyser)."""
    m = model._model
    V_low = float(m.cell_voltage(1000.0, 353.0, 30.0))
    V_high = float(m.cell_voltage(4000.0, 353.0, 30.0))
    assert V_high > V_low


def test_voltage_above_reversible(model):
    """Cell voltage must be above reversible potential."""
    m = model._model
    V = float(m.cell_voltage(2000.0, 353.0, 30.0))
    E_rev = float(m.e_rev(353.0))
    assert V > E_rev


def test_bubble_coverage_increases(model):
    """Bubble coverage must increase with current density."""
    m = model._model
    t1 = float(m.bubble_coverage(1000.0))
    t2 = float(m.bubble_coverage(5000.0))
    assert t2 > t1


def test_bubble_coverage_bounded(model):
    """Bubble coverage must be in [0, 1)."""
    m = model._model
    for j in [100, 1000, 5000, 10000]:
        theta = float(m.bubble_coverage(j))
        assert 0.0 <= theta < 1.0


def test_h2_production_positive(model):
    """H2 rate must be positive for positive current."""
    m = model._model
    h2 = float(m.h2_production_rate(3000.0))
    assert h2 > 0


def test_efficiency_range(model):
    """Efficiency must be in (0, 1)."""
    m = model._model
    eff = float(m.efficiency(3000.0, 353.0, 30.0))
    assert 0.0 < eff < 1.0


def test_higher_temp_lower_voltage(model):
    """Higher temperature should reduce cell voltage (lower ohmic resistance)."""
    m = model._model
    V_cold = float(m.cell_voltage(3000.0, 323.0, 30.0))
    V_hot = float(m.cell_voltage(3000.0, 363.0, 30.0))
    assert V_hot < V_cold


def test_all_finite(model):
    r = model.predict({"current_density": 3000.0, "dt": 60.0, "duration_s": 300.0})
    for k, v in r.items():
        assert np.all(np.isfinite(v)), f"Non-finite in {k}"


def test_benchmark(model):
    start = time.perf_counter()
    model.predict({"current_density": 3000.0, "dt": 1.0, "duration_s": 3600.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 3600-step simulation in {elapsed*1000:.1f} ms")
    assert elapsed < 30.0
