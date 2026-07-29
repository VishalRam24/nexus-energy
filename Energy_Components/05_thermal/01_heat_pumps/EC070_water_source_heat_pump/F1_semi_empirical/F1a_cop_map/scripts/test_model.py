"""EC070 — WSHP — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

@pytest.fixture
def model():
    return ComponentModel()

def test_predict_keys(model):
    r = model.predict({"T_source": 15.0, "T_sink": 45.0})
    for k in ["cop", "cooling_cop", "heating_capacity_kw", "cooling_capacity_kw", "electrical_input_kw"]:
        assert k in r

def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC070"

def test_cop_greater_than_one(model):
    """Heat pump COP must be > 1."""
    temps = np.linspace(5, 30, 50)
    r = model.predict({"T_source": temps, "T_sink": 45.0})
    assert np.all(r["cop"] > 1.0)

def test_cop_decreases_with_temp_lift(model):
    """COP decreases as T_sink - T_source increases."""
    sources = np.array([25.0, 20.0, 15.0, 10.0, 5.0])
    r = model.predict({"T_source": sources, "T_sink": 45.0})
    assert np.all(np.diff(r["cop"]) < 0), "COP must decrease with increasing temp lift"

def test_cop_at_rating_conditions(model):
    """COP at W15/W45 should be ~4-6 (rated for WSHP)."""
    r = model.predict({"T_source": 15.0, "T_sink": 45.0})
    cop = float(r["cop"])
    assert 4.0 < cop < 7.0, f"COP at W15/W45 = {cop:.2f}"

def test_wshp_higher_than_ashp(model):
    """WSHP COP should be higher than typical ASHP at same lift due to stable source."""
    r = model.predict({"T_source": 10.0, "T_sink": 45.0})
    assert float(r["cop"]) > 3.0

def test_energy_balance(model):
    """Q_heating / W_electrical <= COP (aux makes effective COP slightly lower)."""
    r = model.predict({"T_source": 15.0, "T_sink": 45.0})
    q = float(r["heating_capacity_kw"])
    w = float(r["electrical_input_kw"])
    cop = float(r["cop"])
    assert q / w <= cop + 0.1

def test_cooling_capacity_positive(model):
    """Cooling capacity Q_c = Q_h - W must be positive."""
    r = model.predict({"T_source": 15.0, "T_sink": 45.0})
    assert float(r["cooling_capacity_kw"]) > 0

def test_cooling_cop_relationship(model):
    """Cooling COP = Heating COP - 1."""
    r = model.predict({"T_source": 15.0, "T_sink": 45.0})
    assert abs(float(r["cooling_cop"]) - (float(r["cop"]) - 1.0)) < 1e-6

def test_electrical_increases_with_temp_lift(model):
    """More electrical input needed at higher temperature lifts."""
    sources = np.array([25.0, 20.0, 15.0, 10.0, 5.0])
    r = model.predict({"T_source": sources, "T_sink": 55.0})
    assert np.all(np.diff(r["electrical_input_kw"]) > 0)

def test_benchmark(model):
    Ts = np.random.uniform(5, 25, 1000)
    Tk = np.random.uniform(30, 55, 1000)
    start = time.perf_counter()
    model.predict({"T_source": Ts, "T_sink": Tk})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
