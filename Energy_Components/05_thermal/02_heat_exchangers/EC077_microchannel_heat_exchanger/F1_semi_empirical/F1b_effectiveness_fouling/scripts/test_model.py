"""EC077 — Microchannel HX — F1b Fouling + Part-Load LMTD — Test Suite"""

import sys, time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_returns_all_keys(model):
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                        "m_dot_hot": 0.5, "m_dot_cold": 0.3})
    for key in ["Q_kw", "T_h_out", "T_c_out", "effectiveness", "ntu",
                 "U_fouled", "effectiveness_reduction", "F_lmtd"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC077"
    assert info["fidelity"] == "F1b"


# --- Fouling physics ---

def test_fouling_reduces_heat_transfer(model):
    """More fouling -> less heat transfer."""
    r_clean = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                              "m_dot_hot": 0.5, "m_dot_cold": 0.3,
                              "Rf_hot": 0.0, "Rf_cold": 0.0})
    r_dirty = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                              "m_dot_hot": 0.5, "m_dot_cold": 0.3,
                              "Rf_hot": 0.001, "Rf_cold": 0.001})
    assert float(r_dirty["Q_kw"]) < float(r_clean["Q_kw"])


def test_fouling_reduces_u_value(model):
    r_clean = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                              "m_dot_hot": 0.5, "m_dot_cold": 0.3,
                              "Rf_hot": 0.0, "Rf_cold": 0.0})
    r_dirty = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                              "m_dot_hot": 0.5, "m_dot_cold": 0.3,
                              "Rf_hot": 0.001, "Rf_cold": 0.001})
    assert float(r_dirty["U_fouled"]) < float(r_clean["U_fouled"])


def test_fouling_increases_with_rf(model):
    """Effectiveness reduction should increase as fouling resistance increases."""
    Rf = np.array([0.0, 0.0001, 0.001, 0.003])
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                        "m_dot_hot": 0.5, "m_dot_cold": 0.3,
                        "Rf_hot": Rf, "Rf_cold": Rf})
    assert np.all(np.diff(r["effectiveness_reduction"]) > 0)


def test_zero_fouling_zero_reduction(model):
    """With zero fouling, effectiveness_reduction = 0."""
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                        "m_dot_hot": 0.5, "m_dot_cold": 0.3,
                        "Rf_hot": 0.0, "Rf_cold": 0.0})
    assert float(r["effectiveness_reduction"]) == pytest.approx(0.0, abs=1e-6)


# --- Part-load LMTD correction ---

def test_F_lmtd_at_full_load(model):
    """F_LMTD at PLR=1.0 should equal f_lmtd_a (~0.92)."""
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                        "m_dot_hot": 0.5, "m_dot_cold": 0.3, "PLR": 1.0})
    F = float(r["F_lmtd"])
    assert 0.88 < F <= 1.0, f"F_LMTD = {F:.4f}"


def test_F_lmtd_decreases_at_part_load(model):
    """F_LMTD should decrease (or stay same) at part load vs full load."""
    r_full = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                             "m_dot_hot": 0.5, "m_dot_cold": 0.3, "PLR": 1.0})
    r_part = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                             "m_dot_hot": 0.5, "m_dot_cold": 0.3, "PLR": 0.6})
    assert float(r_part["F_lmtd"]) <= float(r_full["F_lmtd"])


def test_part_load_reduces_heat_transfer(model):
    """Lower PLR reduces effective NTU -> less heat transfer."""
    r_full = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                             "m_dot_hot": 0.5, "m_dot_cold": 0.3, "PLR": 1.0})
    r_part = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                             "m_dot_hot": 0.5, "m_dot_cold": 0.3, "PLR": 0.5})
    assert float(r_part["Q_kw"]) <= float(r_full["Q_kw"])


# --- Basic heat exchanger physics ---

def test_positive_heat_transfer(model):
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                        "m_dot_hot": 0.5, "m_dot_cold": 0.3})
    assert float(r["Q_kw"]) > 0


def test_hot_outlet_below_inlet(model):
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                        "m_dot_hot": 0.5, "m_dot_cold": 0.3})
    assert float(r["T_h_out"]) < 80.0


def test_cold_outlet_above_inlet(model):
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                        "m_dot_hot": 0.5, "m_dot_cold": 0.3})
    assert float(r["T_c_out"]) > 20.0


def test_no_second_law_violation(model):
    """T_h_out must be > T_c_in (2nd law for counter/cross flow)."""
    T_c_in = 20.0
    r = model.predict({"T_h_in": 80.0, "T_c_in": T_c_in,
                        "m_dot_hot": 0.5, "m_dot_cold": 0.3})
    assert float(r["T_h_out"]) > T_c_in


def test_energy_balance(model):
    """Energy balance: Q = C_h*(T_h_in - T_h_out) = C_c*(T_c_out - T_c_in)."""
    m_hot, m_cold = 0.5, 0.3
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                        "m_dot_hot": m_hot, "m_dot_cold": m_cold})
    cp_h, cp_c = 1006.0, 4186.0
    Q_h = m_hot * cp_h * (80.0 - float(r["T_h_out"]))
    Q_c = m_cold * cp_c * (float(r["T_c_out"]) - 20.0)
    Q_kw = float(r["Q_kw"])
    assert abs(Q_h / 1000.0 - Q_kw) < 0.01 * Q_kw, f"Hot-side balance fail"
    assert abs(Q_c / 1000.0 - Q_kw) < 0.01 * Q_kw, f"Cold-side balance fail"


def test_effectiveness_range(model):
    """Effectiveness must be [0, 1]."""
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                        "m_dot_hot": 0.5, "m_dot_cold": 0.3})
    eps = float(r["effectiveness"])
    assert 0.0 < eps <= 1.0, f"Effectiveness = {eps}"


def test_zero_flow_gives_zero_Q(model):
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                        "m_dot_hot": 0.0, "m_dot_cold": 0.3})
    assert float(r["Q_kw"]) == 0.0


def test_high_u_clean_high_effectiveness(model):
    """Microchannel HX should achieve high effectiveness at rated flow."""
    r = model.predict({"T_h_in": 60.0, "T_c_in": 20.0,
                        "m_dot_hot": 0.3, "m_dot_cold": 0.3,
                        "Rf_hot": 0.0, "Rf_cold": 0.0})
    eps = float(r["effectiveness"])
    assert eps > 0.80, f"Expected high eps for microchannel, got {eps:.4f}"


# --- Array inputs ---

def test_array_inputs(model):
    T_h = np.array([60.0, 70.0, 80.0])
    r = model.predict({"T_h_in": T_h, "T_c_in": 20.0,
                        "m_dot_hot": 0.5, "m_dot_cold": 0.3})
    assert r["Q_kw"].shape == (3,)


# --- Benchmark ---

def test_benchmark(model):
    m = np.random.uniform(0.1, 1.0, 1000)
    Rf = np.random.uniform(0.0, 0.002, 1000)
    start = time.perf_counter()
    model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                    "m_dot_hot": m, "m_dot_cold": 0.3, "Rf_hot": Rf, "Rf_cold": Rf})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
