"""EC216 — Thermoelectric Generator (TEG) — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_hot": 200.0, "T_cold": 30.0})
    for k in ["efficiency", "power_w", "heat_input_w", "voltage_v"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC216"
    assert info["fidelity"] == "F1a"


def test_efficiency_less_than_carnot(model):
    """TEG efficiency must be strictly less than Carnot efficiency."""
    T_h = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    T_c = 30.0
    r = model.predict({"T_hot": T_h, "T_cold": T_c})
    eta_carnot = 1.0 - (T_c + 273.15) / (T_h + 273.15)
    assert np.all(r["efficiency"] < eta_carnot), "TEG eta must be < Carnot"


def test_efficiency_less_than_10_percent(model):
    """Real Bi2Te3 TEG efficiency is below 10% for typical conditions."""
    r = model.predict({"T_hot": 200.0, "T_cold": 30.0})
    assert float(r["efficiency"]) < 0.10, f"eta={float(r['efficiency']):.3f} exceeds 10%"


def test_power_positive(model):
    """Power output must be positive when T_hot > T_cold."""
    r = model.predict({"T_hot": 100.0, "T_cold": 30.0})
    assert float(r["power_w"]) > 0.0


def test_power_zero_no_temperature_difference(model):
    """No temperature difference -> no power."""
    r = model.predict({"T_hot": 100.0, "T_cold": 100.0})
    assert float(r["power_w"]) == pytest.approx(0.0, abs=1e-9)


def test_power_increases_with_delta_T(model):
    """Power increases monotonically with temperature difference."""
    T_h = np.linspace(60.0, 300.0, 50)
    r = model.predict({"T_hot": T_h, "T_cold": 30.0})
    assert np.all(np.diff(r["power_w"]) > 0), "Power must increase with dT"


def test_efficiency_increases_with_delta_T(model):
    """Efficiency increases with larger temperature difference."""
    T_h = np.array([80.0, 120.0, 160.0, 200.0, 250.0])
    r = model.predict({"T_hot": T_h, "T_cold": 30.0})
    assert np.all(np.diff(r["efficiency"]) > 0), "Efficiency must increase with dT"


def test_energy_conservation(model):
    """P_output < Q_hot (first law: some heat goes to cold side)."""
    r = model.predict({"T_hot": 200.0, "T_cold": 30.0})
    assert float(r["power_w"]) < float(r["heat_input_w"])


def test_benchmark(model):
    T_h = np.random.uniform(50.0, 300.0, 1000)
    T_c = np.random.uniform(0.0, 50.0, 1000)
    # Ensure T_h > T_c
    T_h = np.maximum(T_h, T_c + 10.0)
    start = time.perf_counter()
    model.predict({"T_hot": T_h, "T_cold": T_c})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
