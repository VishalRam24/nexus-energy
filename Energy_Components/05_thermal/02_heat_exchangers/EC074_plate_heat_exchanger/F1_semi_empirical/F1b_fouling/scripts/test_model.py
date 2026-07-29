"""EC074 — Plate Heat Exchanger — F1b Fouling — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


BASE = {"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 1.0}


def test_predict_keys(model):
    r = model.predict(BASE)
    for k in ["Q_kw", "T_h_out", "T_c_out", "effectiveness", "ntu", "U_fouled", "effectiveness_reduction"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC074"
    assert info["fidelity"] == "F1b"


def test_fouling_reduces_heat_transfer(model):
    """More fouling must reduce Q."""
    r_clean = model.predict({**BASE, "fouling_resistance_hot": 0.0, "fouling_resistance_cold": 0.0})
    r_fouled = model.predict({**BASE, "fouling_resistance_hot": 0.002, "fouling_resistance_cold": 0.002})
    assert float(r_clean["Q_kw"]) > float(r_fouled["Q_kw"])


def test_fouling_reduces_effectiveness(model):
    r_clean = model.predict({**BASE, "fouling_resistance_hot": 0.0, "fouling_resistance_cold": 0.0})
    r_fouled = model.predict({**BASE, "fouling_resistance_hot": 0.001, "fouling_resistance_cold": 0.001})
    assert float(r_clean["effectiveness"]) > float(r_fouled["effectiveness"])


def test_U_fouled_less_than_U_clean(model):
    """Fouled U must be less than clean U."""
    r = model.predict({**BASE, "fouling_resistance_hot": 0.0005, "fouling_resistance_cold": 0.0005})
    assert float(r["U_fouled"]) < 3000.0  # U_clean = 3000


def test_zero_fouling_matches_clean(model):
    """With Rf=0, U_fouled should equal U_clean."""
    r = model.predict({**BASE, "fouling_resistance_hot": 0.0, "fouling_resistance_cold": 0.0})
    np.testing.assert_allclose(float(r["U_fouled"]), 3000.0, rtol=1e-10)


def test_effectiveness_reduction_zero_when_clean(model):
    r = model.predict({**BASE, "fouling_resistance_hot": 0.0, "fouling_resistance_cold": 0.0})
    np.testing.assert_allclose(float(r["effectiveness_reduction"]), 0.0, atol=1e-10)


def test_effectiveness_reduction_positive_with_fouling(model):
    r = model.predict({**BASE, "fouling_resistance_hot": 0.001, "fouling_resistance_cold": 0.001})
    assert float(r["effectiveness_reduction"]) > 0.0


def test_energy_balance(model):
    """Hot side loss = cold side gain."""
    r = model.predict(BASE)
    Q_hot = 1.0 * 4186.0 * (80.0 - float(r["T_h_out"])) / 1000.0
    Q_cold = 1.0 * 4186.0 * (float(r["T_c_out"]) - 20.0) / 1000.0
    np.testing.assert_allclose(Q_hot, Q_cold, rtol=1e-6)
    np.testing.assert_allclose(Q_hot, float(r["Q_kw"]), rtol=1e-6)


def test_outlet_temps_physical(model):
    """T_h_out < T_h_in and T_c_out > T_c_in."""
    r = model.predict(BASE)
    assert float(r["T_h_out"]) < 80.0
    assert float(r["T_c_out"]) > 20.0


def test_heavy_fouling_significant_reduction(model):
    """Heavy fouling (Rf=0.005 each side) should cause >10% effectiveness reduction."""
    r = model.predict({**BASE, "fouling_resistance_hot": 0.005, "fouling_resistance_cold": 0.005})
    assert float(r["effectiveness_reduction"]) > 0.10


def test_benchmark(model):
    n = 1000
    inputs = {
        "T_h_in": np.random.uniform(60, 100, n),
        "T_c_in": np.random.uniform(10, 30, n),
        "m_dot_hot": np.random.uniform(0.5, 3.0, n),
        "m_dot_cold": np.random.uniform(0.5, 3.0, n),
        "fouling_resistance_hot": np.random.uniform(0, 0.002, n),
        "fouling_resistance_cold": np.random.uniform(0, 0.002, n),
    }
    start = time.perf_counter()
    model.predict(inputs)
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
