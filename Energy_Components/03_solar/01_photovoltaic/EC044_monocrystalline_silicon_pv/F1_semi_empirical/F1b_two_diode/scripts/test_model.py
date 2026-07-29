"""EC044 — Mono-Si PV — F1b Two-Diode — Test Suite"""

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
    for key in ["i_mp", "v_mp", "p_mp", "i_sc", "v_oc", "fill_factor", "efficiency"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC044"
    assert info["fidelity"] == "F1b"


def test_stc_power(model):
    """At STC, Pmp should be close to 280W for a 60-cell mono-Si module."""
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0})
    assert 230.0 < float(r["p_mp"]) < 370.0, f"Pmp at STC = {float(r['p_mp']):.1f}W"


def test_power_scales_with_irradiance(model):
    """Power should increase with irradiance."""
    irr = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"irradiance_w_m2": irr, "temperature_cell_degC": 25.0})
    assert np.all(np.diff(r["p_mp"]) > 0), "Power must increase with irradiance"


def test_power_decreases_with_temperature(model):
    """Power should decrease with higher cell temperature."""
    temps = np.array([10.0, 25.0, 40.0, 55.0, 70.0])
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": temps})
    assert np.all(np.diff(r["p_mp"]) < 0), "Power must decrease with temperature"


def test_zero_irradiance(model):
    """No sun = no power."""
    r = model.predict({"irradiance_w_m2": 0.0, "temperature_cell_degC": 25.0})
    assert float(r["p_mp"]) < 1.0


def test_efficiency_reasonable(model):
    """Mono-Si efficiency should be 14-23% at STC."""
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0})
    eff = float(r["efficiency"])
    assert 0.14 < eff < 0.23, f"Efficiency = {eff:.3f}"


def test_fill_factor_range(model):
    """Fill factor should be 0.65-0.85 for Si at STC."""
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0})
    ff = float(r["fill_factor"])
    assert 0.60 < ff < 0.90, f"Fill factor = {ff:.3f}"


def test_voc_decreases_with_temperature(model):
    """Voc should decrease with temperature."""
    temps = np.array([10.0, 25.0, 50.0])
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": temps})
    assert np.all(np.diff(r["v_oc"]) < 0), "Voc must decrease with temperature"


def test_isc_increases_with_irradiance(model):
    """Isc should increase with irradiance."""
    irr = np.array([200.0, 500.0, 1000.0])
    r = model.predict({"irradiance_w_m2": irr, "temperature_cell_degC": 25.0})
    assert np.all(np.diff(r["i_sc"]) > 0)


def test_two_diode_low_irradiance(model):
    """Two-diode model should still produce reasonable results at low irradiance."""
    r = model.predict({"irradiance_w_m2": 100.0, "temperature_cell_degC": 25.0})
    assert float(r["p_mp"]) > 10.0, "Should produce some power at 100 W/m2"
    assert float(r["fill_factor"]) > 0.5, "Fill factor should remain reasonable at low G"


def test_array_inputs(model):
    irr = np.array([200.0, 500.0, 800.0])
    temps = np.array([15.0, 25.0, 45.0])
    r = model.predict({"irradiance_w_m2": irr, "temperature_cell_degC": temps})
    assert r["p_mp"].shape == (3,)


def test_benchmark(model):
    """Benchmark: 10 predictions (two-diode is slower due to numerical solver)."""
    irr = np.array([200.0, 400.0, 600.0, 800.0, 1000.0,
                    200.0, 400.0, 600.0, 800.0, 1000.0])
    temps = np.array([15.0, 20.0, 25.0, 30.0, 35.0,
                      40.0, 45.0, 50.0, 55.0, 60.0])
    start = time.perf_counter()
    model.predict({"irradiance_w_m2": irr, "temperature_cell_degC": temps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 10 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 30.0, "Two-diode solver should complete in reasonable time"
