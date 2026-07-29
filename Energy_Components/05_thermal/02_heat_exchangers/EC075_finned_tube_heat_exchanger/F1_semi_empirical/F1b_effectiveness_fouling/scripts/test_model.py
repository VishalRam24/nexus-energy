"""EC075 — Finned-Tube Heat Exchanger — F1b Fouling + Property Corrections — Test Suite"""

import sys
import time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


BASE = {"T_h_in": 70.0, "T_c_in": 20.0, "m_dot_hot": 2.0, "m_dot_cold": 5.0}


# --- Interface ---

def test_predict_keys(model):
    r = model.predict(BASE)
    for k in ["Q_kw", "T_h_out", "T_c_out", "effectiveness", "ntu",
              "U_fouled", "U_effective_clean", "effectiveness_reduction",
              "cleanliness_factor"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC075"
    assert info["fidelity"] == "F1b"


# --- Fouling physics ---

def test_fouling_reduces_heat_transfer(model):
    r_clean = model.predict({**BASE, "fouling_resistance_tube": 0.0, "fouling_resistance_air": 0.0})
    r_fouled = model.predict({**BASE, "fouling_resistance_tube": 0.002, "fouling_resistance_air": 0.002})
    assert float(r_clean["Q_kw"]) > float(r_fouled["Q_kw"])


def test_fouling_reduces_effectiveness(model):
    r_clean = model.predict({**BASE, "fouling_resistance_tube": 0.0, "fouling_resistance_air": 0.0})
    r_fouled = model.predict({**BASE, "fouling_resistance_tube": 0.001, "fouling_resistance_air": 0.001})
    assert float(r_clean["effectiveness"]) > float(r_fouled["effectiveness"])


def test_U_fouled_less_than_U_clean(model):
    r = model.predict({**BASE, "fouling_resistance_tube": 0.001, "fouling_resistance_air": 0.001})
    assert float(r["U_fouled"]) < float(r["U_effective_clean"])


def test_zero_fouling_CF_equals_one(model):
    r = model.predict({**BASE, "fouling_resistance_tube": 0.0, "fouling_resistance_air": 0.0})
    np.testing.assert_allclose(float(r["cleanliness_factor"]), 1.0, rtol=1e-6)


def test_zero_fouling_eps_reduction_zero(model):
    r = model.predict({**BASE, "fouling_resistance_tube": 0.0, "fouling_resistance_air": 0.0})
    np.testing.assert_allclose(float(r["effectiveness_reduction"]), 0.0, atol=1e-10)


def test_eps_reduction_positive_with_fouling(model):
    r = model.predict({**BASE, "fouling_resistance_tube": 0.001, "fouling_resistance_air": 0.001})
    assert float(r["effectiveness_reduction"]) > 0.0


def test_heavy_fouling_significant_reduction(model):
    r = model.predict({**BASE, "fouling_resistance_tube": 0.005, "fouling_resistance_air": 0.002})
    assert float(r["effectiveness_reduction"]) > 0.05


# --- Property / temperature corrections ---

def test_U_increases_with_flow_rate(model):
    """Higher flow rate -> higher Re -> higher U_effective."""
    r_low = model.predict({**BASE, "m_dot_hot": 0.5})
    r_high = model.predict({**BASE, "m_dot_hot": 5.0})
    assert float(r_high["U_effective_clean"]) > float(r_low["U_effective_clean"])


def test_Q_increases_with_flow_rate(model):
    r_low = model.predict({**BASE, "m_dot_hot": 0.5})
    r_high = model.predict({**BASE, "m_dot_hot": 5.0})
    assert float(r_high["Q_kw"]) > float(r_low["Q_kw"])


def test_higher_temp_more_heat_transfer(model):
    """Hotter inlet -> larger driving temperature difference -> more Q."""
    r_lo = model.predict({**BASE, "T_h_in": 40.0})
    r_hi = model.predict({**BASE, "T_h_in": 90.0})
    assert float(r_hi["Q_kw"]) > float(r_lo["Q_kw"])


# --- Energy balance ---

def test_energy_balance(model):
    r = model.predict(BASE)
    Q_hot = 2.0 * 4186.0 * (70.0 - float(r["T_h_out"])) / 1000.0
    Q_cold = 5.0 * 1006.0 * (float(r["T_c_out"]) - 20.0) / 1000.0
    np.testing.assert_allclose(Q_hot, Q_cold, rtol=1e-5)
    np.testing.assert_allclose(Q_hot, float(r["Q_kw"]), rtol=1e-5)


# --- Physical outlet temps ---

def test_outlet_temps_physical(model):
    r = model.predict(BASE)
    assert float(r["T_h_out"]) < 70.0
    assert float(r["T_c_out"]) > 20.0
    assert float(r["T_h_out"]) > float(r["T_c_in"])  # 2nd law


# --- Effectiveness bounds ---

def test_effectiveness_in_0_1(model):
    v = np.linspace(0.5, 5.0, 20)
    for m in v:
        r = model.predict({**BASE, "m_dot_hot": float(m)})
        eps = float(r["effectiveness"])
        assert 0.0 < eps <= 1.0, f"Effectiveness={eps:.4f} out of range at m_dot={m:.2f}"


# --- Array inputs ---

def test_array_inputs(model):
    n = 10
    inp = {
        "T_h_in": np.random.uniform(50, 90, n),
        "T_c_in": np.random.uniform(10, 25, n),
        "m_dot_hot": np.random.uniform(0.5, 4.0, n),
        "m_dot_cold": np.random.uniform(1.0, 10.0, n),
    }
    r = model.predict(inp)
    assert r["Q_kw"].shape == (n,)


# --- Benchmark ---

def test_benchmark(model):
    n = 1000
    inp = {
        "T_h_in": np.random.uniform(50, 90, n),
        "T_c_in": np.random.uniform(10, 30, n),
        "m_dot_hot": np.random.uniform(0.5, 5.0, n),
        "m_dot_cold": np.random.uniform(1.0, 15.0, n),
        "fouling_resistance_tube": np.random.uniform(0, 0.002, n),
        "fouling_resistance_air": np.random.uniform(0, 0.001, n),
    }
    start = time.perf_counter()
    model.predict(inp)
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
