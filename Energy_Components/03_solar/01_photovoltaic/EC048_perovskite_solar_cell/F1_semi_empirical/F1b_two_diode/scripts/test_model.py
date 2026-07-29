"""EC048 — Perovskite Solar Cell — F1b Two-Diode + Hysteresis — Test Suite"""

import sys, time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_returns_dict(model):
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0})
    for key in ["i_mp", "v_mp", "p_mp", "i_sc", "v_oc", "efficiency", "hysteresis_index"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC048"
    assert info["fidelity"] == "F1b"


def test_stc_power(model):
    """At STC, 1cm2 perovskite at ~20% should produce ~20 mW."""
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0})
    p_mw = float(r["p_mp"]) * 1000
    assert 10.0 < p_mw < 30.0, f"Pmp at STC = {p_mw:.2f} mW"


def test_efficiency_range(model):
    """Perovskite efficiency should be 12-25% at STC."""
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0})
    eff = float(r["efficiency"])
    assert 0.10 < eff < 0.28, f"Efficiency = {eff:.3f}"


def test_voc_range(model):
    """Perovskite Voc should be 0.9-1.3 V."""
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0})
    voc = float(r["v_oc"])
    assert 0.8 < voc < 1.4, f"Voc = {voc:.3f} V"


def test_power_scales_with_irradiance(model):
    irr = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"irradiance_w_m2": irr, "temperature_cell_degC": 25.0})
    assert np.all(np.diff(r["p_mp"]) > 0)


def test_power_decreases_with_temperature(model):
    temps = np.array([10.0, 25.0, 40.0, 55.0, 70.0])
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": temps})
    assert np.all(np.diff(r["p_mp"]) < 0)


def test_zero_irradiance(model):
    r = model.predict({"irradiance_w_m2": 0.0, "temperature_cell_degC": 25.0})
    assert float(r["p_mp"]) < 1e-6


def test_hysteresis_reduces_power(model):
    """Non-zero irradiance rate should reduce power output."""
    r_ss = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0,
                           "irradiance_rate_w_m2_s": 0.0})
    r_hyst = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0,
                             "irradiance_rate_w_m2_s": 500.0})
    assert float(r_hyst["p_mp"]) < float(r_ss["p_mp"])


def test_hysteresis_index_zero_at_steady_state(model):
    """At steady state (dG/dt=0), hysteresis index should be 0."""
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0,
                        "irradiance_rate_w_m2_s": 0.0})
    assert float(r["hysteresis_index"]) == pytest.approx(0.0, abs=1e-10)


def test_hysteresis_symmetric(model):
    """Positive and negative irradiance rates should give same hysteresis."""
    r_pos = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0,
                            "irradiance_rate_w_m2_s": 300.0})
    r_neg = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0,
                            "irradiance_rate_w_m2_s": -300.0})
    assert float(r_pos["p_mp"]) == pytest.approx(float(r_neg["p_mp"]), rel=1e-6)


def test_array_inputs(model):
    irr = np.array([200.0, 500.0, 800.0])
    temps = np.array([15.0, 25.0, 45.0])
    r = model.predict({"irradiance_w_m2": irr, "temperature_cell_degC": temps})
    assert r["p_mp"].shape == (3,)


def test_benchmark(model):
    start = time.perf_counter()
    for _ in range(5):
        model.predict({"irradiance_w_m2": 800.0, "temperature_cell_degC": 30.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 5 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 30.0
