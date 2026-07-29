"""EC052 — Bifacial PV Module — F1a — Test Suite"""

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
    r = model.predict({"irradiance_front": 1000.0, "cell_temperature": 25.0})
    for key in ["v_mp", "i_mp", "p_mp", "v_oc", "i_sc",
                "efficiency", "bifacial_gain", "G_effective", "G_rear_used"]:
        assert key in r, f"missing {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC052"
    assert info["fidelity"] == "F1a"


def test_stc_front_only_power(model):
    """At STC with no rear/albedo, Pmp ~ 360-440W (LG NeON 2 BiFacial 400W)."""
    r = model.predict({"irradiance_front": 1000.0, "cell_temperature": 25.0})
    p = float(r["p_mp"])
    assert 350.0 < p < 450.0, f"Pmp front-only at STC = {p:.1f}W"


def test_bifacial_gain_positive(model):
    """Adding rear irradiance must increase power."""
    r0 = model.predict({"irradiance_front": 1000.0, "cell_temperature": 25.0})
    r1 = model.predict({"irradiance_front": 1000.0, "cell_temperature": 25.0,
                        "irradiance_rear": 200.0})
    assert float(r1["p_mp"]) > float(r0["p_mp"]), "rear irradiance must boost Pmp"
    assert float(r1["bifacial_gain"]) > 0


def test_albedo_increases_power(model):
    """Higher albedo -> more rear irradiance -> more power."""
    albedos = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    r = model.predict({"irradiance_front": 1000.0, "cell_temperature": 25.0,
                       "albedo": albedos})
    p = np.asarray(r["p_mp"])
    assert np.all(np.diff(p) > 0), f"Power must increase with albedo: {p}"


def test_bifacial_gain_typical_range(model):
    """At albedo=0.25 (grass), bifacial gain typically 5-15%."""
    r = model.predict({"irradiance_front": 1000.0, "cell_temperature": 25.0,
                       "albedo": 0.25})
    bg = float(r["bifacial_gain"])
    assert 0.03 < bg < 0.20, f"Bifacial gain at albedo=0.25 = {bg:.3f}, expected 5-15%"


def test_zero_irradiance(model):
    r = model.predict({"irradiance_front": 0.0, "cell_temperature": 25.0,
                       "albedo": 0.3})
    assert float(r["p_mp"]) < 1.0


def test_power_decreases_with_temperature(model):
    temps = np.array([10.0, 25.0, 40.0, 55.0, 70.0])
    r = model.predict({"irradiance_front": 1000.0, "cell_temperature": temps,
                       "albedo": 0.3})
    assert np.all(np.diff(r["p_mp"]) < 0)


def test_voc_decreases_with_temperature(model):
    temps = np.array([10.0, 25.0, 50.0])
    r = model.predict({"irradiance_front": 1000.0, "cell_temperature": temps})
    assert np.all(np.diff(r["v_oc"]) < 0)


def test_efficiency_reasonable(model):
    """Front-side efficiency at STC ~ 18-22% for n-type mono-Si bifacial."""
    r = model.predict({"irradiance_front": 1000.0, "cell_temperature": 25.0})
    eff = float(r["efficiency"])
    assert 0.15 < eff < 0.24, f"Front-side eff = {eff:.3f}"


def test_G_effective_consistency(model):
    """G_effective should equal G_front + phi * G_rear_used."""
    phi = model._model.phi
    r = model.predict({"irradiance_front": 1000.0, "cell_temperature": 25.0,
                       "irradiance_rear": 150.0})
    expected = 1000.0 + phi * 150.0
    assert abs(float(r["G_effective"]) - expected) < 1e-6


def test_array_inputs(model):
    g = np.array([400.0, 700.0, 1000.0])
    t = np.array([15.0, 25.0, 45.0])
    r = model.predict({"irradiance_front": g, "cell_temperature": t, "albedo": 0.25})
    assert r["p_mp"].shape == (3,)


def test_pmp_below_voc_isc(model):
    r = model.predict({"irradiance_front": 1000.0, "cell_temperature": 25.0,
                       "albedo": 0.3})
    assert float(r["p_mp"]) < float(r["v_oc"]) * float(r["i_sc"])


def test_benchmark(model):
    g = np.random.uniform(100.0, 1000.0, 1000)
    t = np.random.uniform(10.0, 60.0, 1000)
    a = np.random.uniform(0.1, 0.6, 1000)
    start = time.perf_counter()
    model.predict({"irradiance_front": g, "cell_temperature": t, "albedo": a})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 15.0
