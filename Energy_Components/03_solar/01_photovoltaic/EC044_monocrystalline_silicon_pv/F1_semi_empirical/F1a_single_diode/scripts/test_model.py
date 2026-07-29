"""EC044 — Mono-Si PV — F1a — Test Suite"""

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
    r = model.predict({"irradiance": 1000.0, "cell_temperature": 25.0})
    for key in ["v_mp", "i_mp", "p_mp", "v_oc", "i_sc", "efficiency"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC044"


def test_stc_power(model):
    """At STC, Pmp should be close to 280W."""
    r = model.predict({"irradiance": 1000.0, "cell_temperature": 25.0})
    assert 250.0 < float(r["p_mp"]) < 310.0, f"Pmp at STC = {float(r['p_mp']):.1f}W"


def test_power_scales_with_irradiance(model):
    """Power should increase with irradiance."""
    irr = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"irradiance": irr, "cell_temperature": 25.0})
    assert np.all(np.diff(r["p_mp"]) > 0), "Power must increase with irradiance"


def test_power_decreases_with_temperature(model):
    """Power should decrease with higher cell temperature."""
    temps = np.array([10.0, 25.0, 40.0, 55.0, 70.0])
    r = model.predict({"irradiance": 1000.0, "cell_temperature": temps})
    assert np.all(np.diff(r["p_mp"]) < 0), "Power must decrease with temperature"


def test_zero_irradiance(model):
    """No sun = no power."""
    r = model.predict({"irradiance": 0.0, "cell_temperature": 25.0})
    assert float(r["p_mp"]) < 1.0, "Power should be ~0 at zero irradiance"


def test_efficiency_reasonable(model):
    """Mono-Si efficiency should be 15-22% at STC."""
    r = model.predict({"irradiance": 1000.0, "cell_temperature": 25.0})
    eff = float(r["efficiency"])
    assert 0.10 < eff < 0.25, f"Efficiency = {eff:.3f}, expected 0.15-0.22"


def test_voc_decreases_with_temperature(model):
    """Voc should decrease with temperature (negative temp coeff)."""
    temps = np.array([10.0, 25.0, 50.0])
    r = model.predict({"irradiance": 1000.0, "cell_temperature": temps})
    assert np.all(np.diff(r["v_oc"]) < 0), "Voc must decrease with temperature"


def test_isc_increases_with_irradiance(model):
    """Isc should increase linearly with irradiance."""
    irr = np.array([200.0, 500.0, 1000.0])
    r = model.predict({"irradiance": irr, "cell_temperature": 25.0})
    assert np.all(np.diff(r["i_sc"]) > 0)


def test_array_inputs(model):
    irr = np.array([200.0, 500.0, 800.0])
    temps = np.array([15.0, 25.0, 45.0])
    r = model.predict({"irradiance": irr, "cell_temperature": temps})
    assert r["p_mp"].shape == (3,)


def test_benchmark(model):
    irr = np.random.uniform(100.0, 1000.0, 1000)
    temps = np.random.uniform(10.0, 60.0, 1000)
    start = time.perf_counter()
    model.predict({"irradiance": irr, "cell_temperature": temps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 5.0, "pvlib single-diode should be fast"
