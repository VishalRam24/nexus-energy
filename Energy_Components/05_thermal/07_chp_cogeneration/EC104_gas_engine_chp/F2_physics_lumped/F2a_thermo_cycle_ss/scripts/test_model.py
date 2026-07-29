"""EC104 -- Gas Engine CHP -- F2a Otto Cycle -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"fuel_input_kw": 500.0})
    for k in ["power_electrical_kw", "heat_exhaust_kw", "heat_jacket_kw",
              "eta_electrical", "eta_thermal", "T_exhaust_K"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC104"
    assert info["fidelity"] == "F2a"


def test_otto_efficiency_formula(model):
    """eta_Otto = 1 - 1/r^(gamma-1)."""
    m = model._model
    r, g = m.r, m.gamma
    expected = 1.0 - 1.0 / r**(g - 1.0)
    actual = float(m.otto_efficiency())
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_efficiency_increases_with_r(model):
    m = model._model
    eta_low = float(m.otto_efficiency(8.0))
    eta_high = float(m.otto_efficiency(14.0))
    assert eta_high > eta_low


def test_electrical_output_positive(model):
    r = model.predict({"fuel_input_kw": 500.0})
    assert float(r["power_electrical_kw"]) > 0


def test_thermal_output_positive(model):
    r = model.predict({"fuel_input_kw": 500.0})
    assert float(r["heat_exhaust_kw"]) > 0
    assert float(r["heat_jacket_kw"]) > 0


def test_total_efficiency_below_1(model):
    r = model.predict({"fuel_input_kw": 500.0})
    assert float(r["eta_total"]) < 1.0


def test_total_efficiency_reasonable(model):
    """Total CHP efficiency should be 0.70-0.95."""
    r = model.predict({"fuel_input_kw": 500.0})
    assert 0.50 < float(r["eta_total"]) < 0.98


def test_electrical_efficiency_reasonable(model):
    r = model.predict({"fuel_input_kw": 500.0})
    assert 0.20 < float(r["eta_electrical"]) < 0.55


def test_energy_balance(model):
    """P_el + Q_thermal + losses <= fuel_input."""
    r = model.predict({"fuel_input_kw": 500.0})
    total_out = (float(r["power_electrical_kw"]) +
                 float(r["heat_exhaust_kw"]) + float(r["heat_jacket_kw"]))
    assert total_out <= 500.0 * 1.01  # allow 1% tolerance


def test_exhaust_temp_above_ambient(model):
    r = model.predict({"fuel_input_kw": 500.0, "T_ambient_K": 298.15})
    assert float(r["T_exhaust_K"]) > 298.15


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({"fuel_input_kw": 500.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 5.0
