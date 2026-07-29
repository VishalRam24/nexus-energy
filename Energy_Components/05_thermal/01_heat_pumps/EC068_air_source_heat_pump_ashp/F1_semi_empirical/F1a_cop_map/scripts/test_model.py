"""EC068 — ASHP — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

@pytest.fixture
def model():
    return ComponentModel()

def test_predict_keys(model):
    r = model.predict({"T_source": 7.0, "T_sink": 35.0})
    for k in ["cop", "heating_capacity_kw", "electrical_input_kw"]:
        assert k in r

def test_get_info(model):
    assert model.get_info()["ec_id"] == "EC068"

def test_cop_greater_than_one(model):
    """Heat pump COP must be > 1 (otherwise it's just resistance heating)."""
    temps = np.linspace(-15, 35, 50)
    r = model.predict({"T_source": temps, "T_sink": 35.0})
    assert np.all(r["cop"] > 1.0)

def test_cop_decreases_with_temp_lift(model):
    """COP decreases as T_sink - T_source increases."""
    sources = np.array([15.0, 7.0, 0.0, -7.0, -15.0])
    r = model.predict({"T_source": sources, "T_sink": 35.0})
    assert np.all(np.diff(r["cop"]) < 0), "COP must decrease with increasing temp lift"

def test_cop_at_rating_conditions(model):
    """COP at A7/W35 should be ~3.5 (rated)."""
    r = model.predict({"T_source": 7.0, "T_sink": 35.0})
    assert 2.5 < float(r["cop"]) < 5.0, f"COP at A7/W35 = {float(r['cop']):.2f}"

def test_energy_balance(model):
    """Q_heating = COP * W_electrical (approximately, ignoring aux)."""
    r = model.predict({"T_source": 7.0, "T_sink": 35.0})
    q = float(r["heating_capacity_kw"])
    w = float(r["electrical_input_kw"])
    cop = float(r["cop"])
    # w includes aux so cop_actual = q/w <= cop
    assert q / w <= cop + 0.1

def test_electrical_increases_with_temp_lift(model):
    """More electrical input needed at higher temperature lifts."""
    sources = np.array([15.0, 7.0, 0.0, -10.0])
    r = model.predict({"T_source": sources, "T_sink": 45.0})
    assert np.all(np.diff(r["electrical_input_kw"]) > 0)

def test_benchmark(model):
    Ts = np.random.uniform(-15, 30, 1000)
    Tk = np.random.uniform(30, 55, 1000)
    start = time.perf_counter()
    model.predict({"T_source": Ts, "T_sink": Tk})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
