"""EC045 — Poly-Si PV — F1a — Test Suite"""

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
    assert info["ec_id"] == "EC045"
    assert info["fidelity"] == "F1a"


def test_stc_power(model):
    """At STC, Pmp should be ~200-300W for a 60-cell poly-Si module."""
    r = model.predict({"irradiance": 1000.0, "cell_temperature": 25.0})
    assert 200.0 < float(r["p_mp"]) < 320.0, f"Pmp at STC = {float(r['p_mp']):.1f}W"


def test_power_scales_with_irradiance(model):
    irr = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"irradiance": irr, "cell_temperature": 25.0})
    assert np.all(np.diff(r["p_mp"]) > 0), "Power must increase with irradiance"


def test_power_decreases_with_temperature(model):
    temps = np.array([10.0, 25.0, 40.0, 55.0, 70.0])
    r = model.predict({"irradiance": 1000.0, "cell_temperature": temps})
    assert np.all(np.diff(r["p_mp"]) < 0), "Power must decrease with temperature"


def test_zero_irradiance(model):
    r = model.predict({"irradiance": 0.0, "cell_temperature": 25.0})
    assert float(r["p_mp"]) < 1.0


def test_efficiency_reasonable(model):
    """Poly-Si efficiency should be 13-18% at STC."""
    r = model.predict({"irradiance": 1000.0, "cell_temperature": 25.0})
    eff = float(r["efficiency"])
    assert 0.12 < eff < 0.20, f"Efficiency = {eff:.3f}, expected 0.13-0.18"


def test_voc_decreases_with_temperature(model):
    temps = np.array([10.0, 25.0, 50.0])
    r = model.predict({"irradiance": 1000.0, "cell_temperature": temps})
    assert np.all(np.diff(r["v_oc"]) < 0), "Voc must decrease with temperature"


def test_isc_increases_with_irradiance(model):
    irr = np.array([200.0, 500.0, 1000.0])
    r = model.predict({"irradiance": irr, "cell_temperature": 25.0})
    assert np.all(np.diff(r["i_sc"]) > 0)


def test_array_inputs(model):
    irr = np.array([200.0, 500.0, 800.0])
    temps = np.array([15.0, 25.0, 45.0])
    r = model.predict({"irradiance": irr, "cell_temperature": temps})
    assert r["p_mp"].shape == (3,)


def test_pmp_below_voc_isc(model):
    """P_mp must be less than V_oc * I_sc."""
    r = model.predict({"irradiance": 1000.0, "cell_temperature": 25.0})
    assert float(r["p_mp"]) < float(r["v_oc"]) * float(r["i_sc"])


def test_benchmark(model):
    irr = np.random.uniform(100.0, 1000.0, 1000)
    temps = np.random.uniform(10.0, 60.0, 1000)
    start = time.perf_counter()
    model.predict({"irradiance": irr, "cell_temperature": temps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 10.0
