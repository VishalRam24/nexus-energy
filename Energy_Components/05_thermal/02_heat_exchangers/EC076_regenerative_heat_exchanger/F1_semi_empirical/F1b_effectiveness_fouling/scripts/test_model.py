"""EC076 — Regenerative Heat Exchanger — F1b Fouling + Carryover — Test Suite"""

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


BASE = {"T_h_in": 250.0, "T_c_in": 20.0, "m_dot_hot": 3.0, "m_dot_cold": 3.0}


# --- Interface ---

def test_predict_keys(model):
    r = model.predict(BASE)
    for k in ["Q_kw", "T_h_out", "T_c_out", "effectiveness", "ntu",
              "U_fouled", "effectiveness_reduction", "carryover_penalty",
              "cleanliness_factor"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC076"
    assert info["fidelity"] == "F1b"


# --- Fouling physics ---

def test_fouling_reduces_heat_transfer(model):
    r_clean = model.predict({**BASE, "fouling_resistance_hot": 0.0, "fouling_resistance_cold": 0.0})
    r_fouled = model.predict({**BASE, "fouling_resistance_hot": 0.002, "fouling_resistance_cold": 0.002})
    assert float(r_clean["Q_kw"]) > float(r_fouled["Q_kw"])


def test_fouling_reduces_effectiveness(model):
    r_clean = model.predict({**BASE, "fouling_resistance_hot": 0.0, "fouling_resistance_cold": 0.0})
    r_fouled = model.predict({**BASE, "fouling_resistance_hot": 0.001, "fouling_resistance_cold": 0.001})
    assert float(r_clean["effectiveness"]) > float(r_fouled["effectiveness"])


def test_U_fouled_less_than_U_clean(model):
    r = model.predict({**BASE, "fouling_resistance_hot": 0.001, "fouling_resistance_cold": 0.001})
    assert float(r["U_fouled"]) < 80.0  # U_clean


def test_zero_fouling_CF_equals_one(model):
    r = model.predict({**BASE, "fouling_resistance_hot": 0.0, "fouling_resistance_cold": 0.0,
                       "carryover_leakage": 0.0})
    np.testing.assert_allclose(float(r["cleanliness_factor"]), 1.0, rtol=1e-6)


# --- Carryover/leakage physics ---

def test_carryover_reduces_heat_transfer(model):
    r_low = model.predict({**BASE, "carryover_leakage": 0.0})
    r_high = model.predict({**BASE, "carryover_leakage": 0.08})
    assert float(r_low["Q_kw"]) > float(r_high["Q_kw"])


def test_zero_carryover_higher_eps(model):
    r_0 = model.predict({**BASE, "carryover_leakage": 0.0})
    r_5 = model.predict({**BASE, "carryover_leakage": 0.05})
    assert float(r_0["effectiveness"]) > float(r_5["effectiveness"])


def test_carryover_penalty_matches_input(model):
    co = 0.04
    r = model.predict({**BASE, "carryover_leakage": co})
    np.testing.assert_allclose(float(r["carryover_penalty"]), co, rtol=1e-6)


# --- Cr* correction ---

def test_high_crstar_near_ideal_recuperator(model):
    """Very high Cr* -> regenerator approaches ideal recuperator (eps_regen ~ eps_cf)."""
    r_high = model.predict({**BASE, "Cr_star": 50.0, "carryover_leakage": 0.0,
                              "fouling_resistance_hot": 0.0, "fouling_resistance_cold": 0.0})
    r_low = model.predict({**BASE, "Cr_star": 1.5, "carryover_leakage": 0.0,
                             "fouling_resistance_hot": 0.0, "fouling_resistance_cold": 0.0})
    assert float(r_high["effectiveness"]) > float(r_low["effectiveness"])


# --- Energy balance ---

def test_energy_balance(model):
    r = model.predict({**BASE, "carryover_leakage": 0.0})
    Q_hot = 3.0 * 1010.0 * (250.0 - float(r["T_h_out"])) / 1000.0
    Q_cold = 3.0 * 1010.0 * (float(r["T_c_out"]) - 20.0) / 1000.0
    np.testing.assert_allclose(Q_hot, Q_cold, rtol=1e-5)
    np.testing.assert_allclose(Q_hot, float(r["Q_kw"]), rtol=1e-5)


# --- Physical outlet temps ---

def test_outlet_temps_physical(model):
    r = model.predict(BASE)
    assert float(r["T_h_out"]) < 250.0
    assert float(r["T_c_out"]) > 20.0


def test_2nd_law_outlet_temps(model):
    """T_h_out must not be colder than T_c_in; T_c_out must not exceed T_h_in."""
    r = model.predict(BASE)
    assert float(r["T_h_out"]) >= float(r["T_c_in"]) - 0.1
    assert float(r["T_c_out"]) <= float(r["T_h_in"]) + 0.1


# --- Effectiveness bounds ---

def test_effectiveness_in_0_1(model):
    r = model.predict(BASE)
    eps = float(r["effectiveness"])
    assert 0.0 < eps <= 1.0, f"eps={eps:.4f}"


# --- Array inputs ---

def test_array_inputs(model):
    n = 10
    inp = {
        "T_h_in": np.random.uniform(100, 400, n),
        "T_c_in": np.random.uniform(10, 30, n),
        "m_dot_hot": np.random.uniform(1.0, 8.0, n),
        "m_dot_cold": np.random.uniform(1.0, 8.0, n),
    }
    r = model.predict(inp)
    assert r["Q_kw"].shape == (n,)


# --- Benchmark ---

def test_benchmark(model):
    n = 1000
    inp = {
        "T_h_in": np.random.uniform(80, 400, n),
        "T_c_in": np.random.uniform(5, 30, n),
        "m_dot_hot": np.random.uniform(0.5, 10.0, n),
        "m_dot_cold": np.random.uniform(0.5, 10.0, n),
        "fouling_resistance_hot": np.random.uniform(0, 0.002, n),
        "fouling_resistance_cold": np.random.uniform(0, 0.002, n),
    }
    start = time.perf_counter()
    model.predict(inp)
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
