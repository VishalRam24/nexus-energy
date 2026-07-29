"""EC102 — Kalina Cycle — F1b Part-Load + Condenser T + NH3 Fraction — Test Suite"""

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
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0, "PLR": 1.0})
    for key in ["efficiency", "eta_carnot", "power_output_kw",
                 "heat_rejection_kw", "f_composition", "f_condenser"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC102"
    assert info["fidelity"] == "F1b"


# --- Efficiency physics ---

def test_efficiency_below_carnot(model):
    """Kalina efficiency must be below Carnot limit."""
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0, "PLR": 1.0})
    assert float(r["efficiency"]) < float(r["eta_carnot"])


def test_efficiency_positive(model):
    PLR = np.linspace(0.3, 1.0, 20)
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0, "PLR": PLR})
    assert np.all(r["efficiency"] > 0)


def test_efficiency_drops_at_part_load(model):
    r_full = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0, "PLR": 1.0})
    r_half = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0, "PLR": 0.5})
    assert float(r_half["efficiency"]) < float(r_full["efficiency"])


def test_higher_heat_source_improves_efficiency(model):
    r_low  = model.predict({"T_heat_source": 100.0, "T_condenser": 32.0, "PLR": 1.0})
    r_high = model.predict({"T_heat_source": 180.0, "T_condenser": 32.0, "PLR": 1.0})
    assert float(r_high["efficiency"]) > float(r_low["efficiency"])


# --- Condenser temperature sensitivity ---

def test_higher_condenser_temp_lowers_efficiency(model):
    """Higher condenser T -> lower Carnot -> lower actual efficiency."""
    r_cool = model.predict({"T_heat_source": 150.0, "T_condenser": 20.0, "PLR": 1.0})
    r_hot  = model.predict({"T_heat_source": 150.0, "T_condenser": 45.0, "PLR": 1.0})
    assert float(r_cool["efficiency"]) > float(r_hot["efficiency"])


def test_condenser_sensitivity_significant(model):
    """15K rise in condenser should reduce efficiency by >15% (NH3 high dp/dT)."""
    r_32 = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0, "PLR": 1.0})
    r_47 = model.predict({"T_heat_source": 150.0, "T_condenser": 47.0, "PLR": 1.0})
    drop = 1.0 - float(r_47["efficiency"]) / float(r_32["efficiency"])
    assert drop > 0.15, f"Only {drop*100:.1f}% drop for 15K condenser rise"


def test_f_condenser_unity_at_design(model):
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0})
    assert float(r["f_condenser"]) == pytest.approx(1.0, abs=0.001)


# --- Ammonia fraction ---

def test_f_composition_unity_at_design(model):
    """f_composition should be 1.0 at design NH3 fraction."""
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0,
                        "x_NH3": 0.85})
    assert float(r["f_composition"]) == pytest.approx(1.0, abs=0.001)


def test_higher_x_NH3_improves_efficiency_near_design(model):
    """Small increase in NH3 above design improves efficiency (k_x > 0)."""
    r_lo = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0,
                           "PLR": 1.0, "x_NH3": 0.80})
    r_hi = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0,
                           "PLR": 1.0, "x_NH3": 0.90})
    assert float(r_hi["efficiency"]) > float(r_lo["efficiency"])


# --- Energy balance ---

def test_energy_balance(model):
    """P_out + Q_reject = Q_hot (1st law)."""
    Q_hot = 1000.0  # kW
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0,
                        "PLR": 0.8, "heat_input_kw": Q_hot})
    P_out = float(r["power_output_kw"])
    Q_rej = float(r["heat_rejection_kw"])
    assert abs((P_out + Q_rej) - Q_hot) < 0.1, (
        f"1st law: P+Q_rej={P_out+Q_rej:.2f} != Q_hot={Q_hot:.2f}")


# --- Power output at rated ---

def test_power_at_rated(model):
    """At design conditions PLR=1, power should be ~100 kW."""
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0, "PLR": 1.0})
    P = float(r["power_output_kw"])
    assert abs(P - 100.0) / 100.0 < 0.20, f"P={P:.1f} kW expected ~100 kW"


# --- Heat rate ---

def test_heat_rate_increases_at_part_load(model):
    r_full = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0, "PLR": 1.0})
    r_part = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0, "PLR": 0.5})
    hr_full = 3600 / float(r_full["efficiency"])
    hr_part = 3600 / float(r_part["efficiency"])
    assert hr_part > hr_full


# --- Carnot ---

def test_eta_carnot_increases_with_temp_diff(model):
    T_hot = np.array([100.0, 130.0, 160.0, 200.0])
    r = model.predict({"T_heat_source": T_hot, "T_condenser": 32.0})
    assert np.all(np.diff(r["eta_carnot"]) > 0)


# --- Array inputs ---

def test_array_inputs(model):
    T_cond = np.array([20.0, 32.0, 45.0])
    r = model.predict({"T_heat_source": 150.0, "T_condenser": T_cond, "PLR": 1.0})
    assert r["efficiency"].shape == (3,)


# --- Benchmark ---

def test_benchmark(model):
    PLR    = np.random.uniform(0.3, 1.0, 1000)
    T_cond = np.random.uniform(20, 50, 1000)
    start  = time.perf_counter()
    model.predict({"T_heat_source": 150.0, "T_condenser": T_cond, "PLR": PLR})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
