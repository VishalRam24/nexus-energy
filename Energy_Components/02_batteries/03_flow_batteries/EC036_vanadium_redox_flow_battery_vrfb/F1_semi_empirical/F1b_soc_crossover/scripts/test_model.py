"""EC036 — VRFB — F1b SOC+Crossover — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"soc": 0.5, "current": 50.0, "cycle_number": 0})
    for k in ["terminal_voltage", "power", "crossover_current_A",
              "capacity_fade_pct", "coulombic_efficiency"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC036"
    assert info["fidelity"] == "F1b"


def test_crossover_current_positive(model):
    r = model.predict({"soc": 0.7, "current": 50.0})
    assert float(r["crossover_current_A"]) > 0


def test_crossover_increases_with_soc_deviation(model):
    """Crossover should be higher at extreme SOC than at SOC=0.5."""
    r1 = model.predict({"soc": 0.5, "current": 50.0})
    r2 = model.predict({"soc": 0.9, "current": 50.0})
    assert float(r2["crossover_current_A"]) > float(r1["crossover_current_A"])


def test_capacity_fade_increases_with_cycles(model):
    r1 = model.predict({"soc": 0.5, "current": 50.0, "cycle_number": 100})
    r2 = model.predict({"soc": 0.5, "current": 50.0, "cycle_number": 1000})
    assert float(r2["capacity_fade_pct"]) > float(r1["capacity_fade_pct"])


def test_coulombic_efficiency_less_than_one(model):
    r = model.predict({"soc": 0.5, "current": 50.0})
    eta = float(r["coulombic_efficiency"])
    assert 0 < eta < 1.0


def test_voltage_reasonable(model):
    r = model.predict({"soc": 0.5, "current": 50.0})
    V = float(r["terminal_voltage"])
    assert 30 < V < 70, f"Stack voltage {V:.1f} V out of range"


def test_capacity_fade_zero_at_cycle_zero(model):
    r = model.predict({"soc": 0.5, "current": 50.0, "cycle_number": 0})
    assert abs(float(r["capacity_fade_pct"])) < 1e-10


def test_benchmark(model):
    soc = np.random.uniform(0.1, 0.9, 1000)
    current = np.random.uniform(10, 80, 1000)
    start = time.perf_counter()
    model.predict({"soc": soc, "current": current, "cycle_number": 100})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
