"""EC096 — Magnetic Refrigeration — F1b COP + Part-Load — Test Suite"""

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
    r = model.predict({"T_hot_degC": 35.0, "T_cold_degC": 15.0, "PLR": 1.0})
    for key in ["cop", "cop_carnot", "eta_vs_carnot", "cooling_kw",
                 "electrical_kw", "heat_rejection_kw", "delta_T_span_K"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC096"
    assert info["fidelity"] == "F1b"


# --- COP physics ---

def test_cop_less_than_carnot(model):
    """Actual COP must be below Carnot limit."""
    r = model.predict({"T_hot_degC": 35.0, "T_cold_degC": 15.0, "PLR": 1.0})
    assert float(r["cop"]) < float(r["cop_carnot"])


def test_cop_greater_than_1(model):
    """Magnetic refrigerators should have COP > 1 at reasonable conditions."""
    r = model.predict({"T_hot_degC": 35.0, "T_cold_degC": 15.0, "PLR": 1.0})
    assert float(r["cop"]) > 1.0


def test_cop_positive(model):
    PLR = np.linspace(0.3, 1.0, 30)
    r = model.predict({"T_hot_degC": 35.0, "T_cold_degC": 15.0, "PLR": PLR})
    assert np.all(r["cop"] > 0)


# --- Temperature effects ---

def test_higher_hot_temp_lowers_cop(model):
    """Higher reject temperature -> lower COP."""
    r_cool = model.predict({"T_hot_degC": 28.0, "T_cold_degC": 15.0, "PLR": 1.0})
    r_hot  = model.predict({"T_hot_degC": 45.0, "T_cold_degC": 15.0, "PLR": 1.0})
    assert float(r_cool["cop"]) > float(r_hot["cop"])


def test_wider_span_lowers_cop(model):
    """Larger temperature span -> lower Carnot -> lower actual COP."""
    r_narrow = model.predict({"T_hot_degC": 30.0, "T_cold_degC": 20.0, "PLR": 1.0})
    r_wide   = model.predict({"T_hot_degC": 40.0, "T_cold_degC": 10.0, "PLR": 1.0})
    assert float(r_narrow["cop"]) > float(r_wide["cop"])


def test_cop_carnot_increases_with_cold_temp(model):
    """Carnot COP = T_cold/(T_hot-T_cold): higher T_cold -> higher COP."""
    T_cold = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    r = model.predict({"T_hot_degC": 35.0, "T_cold_degC": T_cold, "PLR": 1.0})
    assert np.all(np.diff(r["cop_carnot"]) > 0)


# --- Part-load ---

def test_cop_drops_at_part_load(model):
    """COP should be lower at part load due to frequency mismatch."""
    r_full = model.predict({"T_hot_degC": 35.0, "T_cold_degC": 15.0, "PLR": 1.0})
    r_half = model.predict({"T_hot_degC": 35.0, "T_cold_degC": 15.0, "PLR": 0.5})
    assert float(r_half["cop"]) < float(r_full["cop"])


def test_cooling_scales_with_plr(model):
    """Cooling capacity should scale with PLR."""
    PLR = np.array([0.3, 0.5, 0.75, 1.0])
    r = model.predict({"T_hot_degC": 35.0, "T_cold_degC": 15.0, "PLR": PLR})
    assert np.all(np.diff(r["cooling_kw"]) > 0)


# --- Energy balance (1st law) ---

def test_energy_balance(model):
    """Q_hot = Q_cold + W_in (1st law)."""
    r = model.predict({"T_hot_degC": 35.0, "T_cold_degC": 15.0, "PLR": 1.0})
    Q_cold = float(r["cooling_kw"])
    W_in   = float(r["electrical_kw"])
    Q_hot  = float(r["heat_rejection_kw"])
    assert abs((Q_cold + W_in) - Q_hot) < 0.001 * Q_hot, (
        f"1st law fail: Q_cold+W_in={Q_cold+W_in:.4f}, Q_hot={Q_hot:.4f}")


# --- COP definition consistency ---

def test_cop_definition(model):
    """COP = Q_cold / W_in."""
    r = model.predict({"T_hot_degC": 35.0, "T_cold_degC": 15.0, "PLR": 0.8})
    cop_check = float(r["cooling_kw"]) / float(r["electrical_kw"])
    assert abs(cop_check - float(r["cop"])) < 0.01 * float(r["cop"])


# --- eta_vs_carnot ---

def test_eta_vs_carnot_between_0_and_1(model):
    """eta_vs_carnot = COP / COP_Carnot should be in (0, 1)."""
    r = model.predict({"T_hot_degC": 35.0, "T_cold_degC": 15.0, "PLR": 1.0})
    eta = float(r["eta_vs_carnot"])
    assert 0.0 < eta < 1.0, f"eta_vs_carnot = {eta:.4f}"


# --- Temperature span ---

def test_delta_T_span(model):
    r = model.predict({"T_hot_degC": 35.0, "T_cold_degC": 15.0})
    assert float(r["delta_T_span_K"]) == pytest.approx(20.0, abs=0.1)


# --- Array inputs ---

def test_array_inputs(model):
    T_hot = np.array([30.0, 35.0, 40.0])
    r = model.predict({"T_hot_degC": T_hot, "T_cold_degC": 15.0, "PLR": 1.0})
    assert r["cop"].shape == (3,)


# --- Benchmark ---

def test_benchmark(model):
    PLR   = np.random.uniform(0.3, 1.0, 1000)
    T_hot = np.random.uniform(28, 48, 1000)
    start = time.perf_counter()
    model.predict({"T_hot_degC": T_hot, "T_cold_degC": 15.0, "PLR": PLR})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
