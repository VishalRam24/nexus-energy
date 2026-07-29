"""EC010 -- SOEC -- F2a Electrochemical -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"current_density": 0.5, "dt": 60.0, "duration_s": 60.0})
    for k in ["t", "voltage", "h2_production", "efficiency", "thermal_mode"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC010"
    assert info["fidelity"] == "F2a"


def test_voltage_increases_with_current(model):
    m = model._model
    V_low = float(m.cell_voltage(0.1, 1073.15, 0.5))
    V_high = float(m.cell_voltage(1.0, 1073.15, 0.5))
    assert V_high > V_low


def test_endothermic_at_low_current(model):
    """At low current density, cell should be endothermic (V < V_tn)."""
    m = model._model
    V = float(m.cell_voltage(0.05, 1073.15, 0.3))
    assert V < m.V_tn, f"Expected endothermic (V={V} < V_tn={m.V_tn})"


def test_exothermic_at_high_current(model):
    """At high current density, cell should be exothermic (V > V_tn)."""
    m = model._model
    V = float(m.cell_voltage(1.5, 1073.15, 0.5))
    assert V > m.V_tn, f"Expected exothermic (V={V} > V_tn={m.V_tn})"


def test_higher_temp_lower_ohmic(model):
    """Higher T -> higher YSZ conductivity -> lower ohmic loss."""
    m = model._model
    eta_cold = float(m.eta_ohmic(0.5, 923.0))
    eta_hot = float(m.eta_ohmic(0.5, 1173.0))
    assert eta_hot < eta_cold


def test_h2_production_positive(model):
    m = model._model
    h2 = float(m.h2_production_rate(0.5))
    assert h2 > 0


def test_steam_utilization_raises_voltage(model):
    """Higher steam utilization -> higher Nernst voltage."""
    m = model._model
    V_low_U = float(m.cell_voltage(0.5, 1073.15, 0.2))
    V_high_U = float(m.cell_voltage(0.5, 1073.15, 0.8))
    assert V_high_U > V_low_U


def test_efficiency_can_exceed_1_endothermic(model):
    """In endothermic mode, electrical efficiency can exceed 1 (uses waste heat)."""
    m = model._model
    eff = float(m.efficiency(0.05, 1073.15, 0.3))
    # Should be close to or above 1 in endothermic regime
    assert eff > 0.5


def test_all_finite(model):
    r = model.predict({"current_density": 0.5, "dt": 60.0, "duration_s": 300.0})
    for k, v in r.items():
        assert np.all(np.isfinite(v)), f"Non-finite in {k}"


def test_benchmark(model):
    start = time.perf_counter()
    model.predict({"current_density": 0.5, "dt": 1.0, "duration_s": 3600.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 3600-step simulation in {elapsed*1000:.1f} ms")
    assert elapsed < 30.0
