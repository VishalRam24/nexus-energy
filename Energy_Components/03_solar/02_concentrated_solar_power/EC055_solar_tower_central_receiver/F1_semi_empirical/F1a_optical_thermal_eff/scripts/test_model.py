"""EC055 — Solar Tower CSP — F1a — Test Suite"""

import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"dni": 900.0, "solar_zenith": 20.0,
                       "T_receiver": 565.0, "T_ambient": 25.0})
    for k in ["Q_field_kw", "useful_heat_kw", "thermal_loss_kw",
              "optical_efficiency", "receiver_efficiency", "overall_efficiency"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC055"
    assert info["fidelity"] == "F1a"


def test_zero_dni_zero_useful(model):
    r = model.predict({"dni": 0.0, "solar_zenith": 30.0,
                       "T_receiver": 565.0, "T_ambient": 25.0})
    assert float(r["useful_heat_kw"]) == 0.0
    assert float(r["optical_efficiency"]) == 0.0


def test_thermal_loss_increases_with_T(model):
    T = np.array([300.0, 400.0, 500.0, 600.0, 700.0])
    r = model.predict({"dni": 900.0, "solar_zenith": 0.0,
                       "T_receiver": T, "T_ambient": 25.0})
    q = np.asarray(r["thermal_loss_kw"])
    assert np.all(np.diff(q) > 0), f"Thermal loss must rise with T_recv: {q}"


def test_optical_decreases_with_zenith(model):
    z = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0])
    r = model.predict({"dni": 900.0, "solar_zenith": z,
                       "T_receiver": 565.0, "T_ambient": 25.0})
    eta = np.asarray(r["optical_efficiency"])
    assert np.all(np.diff(eta) <= 0), f"Optical eff should drop with zenith: {eta}"


def test_useful_heat_increases_with_dni(model):
    dnis = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"dni": dnis, "solar_zenith": 20.0,
                       "T_receiver": 565.0, "T_ambient": 25.0})
    q = np.asarray(r["useful_heat_kw"])
    assert np.all(np.diff(q) > 0)


def test_useful_heat_non_negative(model):
    """Even at 0 DNI and high T_recv, useful_heat clamped at 0."""
    r = model.predict({"dni": 0.0, "solar_zenith": 0.0,
                       "T_receiver": 700.0, "T_ambient": 10.0})
    assert float(r["useful_heat_kw"]) >= 0.0


def test_overall_below_optical(model):
    r = model.predict({"dni": 900.0, "solar_zenith": 20.0,
                       "T_receiver": 565.0, "T_ambient": 25.0})
    assert float(r["overall_efficiency"]) <= float(r["optical_efficiency"]) + 1e-9


def test_optical_peak_at_zenith_zero(model):
    """At zenith=0, optical efficiency should be ~ eta_field_peak."""
    r = model.predict({"dni": 1000.0, "solar_zenith": 0.0,
                       "T_receiver": 565.0, "T_ambient": 25.0})
    eta = float(r["optical_efficiency"])
    assert abs(eta - 0.65) < 1e-6, f"At zenith=0, eta_optical = {eta:.4f}"


def test_array_inputs(model):
    dni = np.array([400.0, 700.0, 900.0])
    z = np.array([10.0, 30.0, 50.0])
    Tr = np.array([400.0, 500.0, 600.0])
    Ta = np.array([10.0, 25.0, 35.0])
    r = model.predict({"dni": dni, "solar_zenith": z,
                       "T_receiver": Tr, "T_ambient": Ta})
    assert np.asarray(r["useful_heat_kw"]).shape == (3,)


def test_benchmark(model):
    n = 1000
    dni = np.random.uniform(0, 1000, n)
    z = np.random.uniform(0, 80, n)
    Tr = np.random.uniform(400, 700, n)
    Ta = np.random.uniform(0, 40, n)
    start = time.perf_counter()
    model.predict({"dni": dni, "solar_zenith": z, "T_receiver": Tr, "T_ambient": Ta})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
