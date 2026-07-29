"""EC059 — Evacuated Tube Solar Collector — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"irradiance": 800.0, "T_inlet": 60.0, "T_ambient": 10.0})
    for k in ["useful_heat_w", "efficiency", "T_outlet_approx"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC059"
    assert info["fidelity"] == "F1a"


def test_zero_irradiance_zero_heat(model):
    r = model.predict({"irradiance": 0.0, "T_inlet": 60.0, "T_ambient": 20.0})
    assert float(r["useful_heat_w"]) == 0.0


def test_efficiency_below_FR_tau_alpha(model):
    G = np.linspace(100, 1200, 50)
    r = model.predict({"irradiance": G, "T_inlet": 25.0, "T_ambient": 25.0})
    eta = np.asarray(r["efficiency"])
    assert np.all(eta <= 0.72 + 1e-9)


def test_low_UL_advantage_at_high_dT(model):
    """ETC should still produce useful heat at high (T_in - T_amb), where flat plate would stagnate.

    With F_R*U_L = 1.4 W/m2K and G=400 W/m2, even (T_in - T_amb)=80K gives:
        eta = 0.72 - 1.4*80/400 = 0.72 - 0.28 = 0.44 (still positive).
    A flat plate (F_R*U_L=4.5) would have eta = 0.75 - 4.5*80/400 = -0.15 (stagnated).
    """
    r = model.predict({"irradiance": 400.0, "T_inlet": 90.0, "T_ambient": 10.0})
    assert float(r["useful_heat_w"]) > 0.0, "ETC should still extract heat at high dT/G"
    assert float(r["efficiency"]) > 0.30


def test_eta_decreases_with_T_in(model):
    G = 800.0
    T_in = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
    r = model.predict({"irradiance": G, "T_inlet": T_in, "T_ambient": 20.0})
    eta = np.asarray(r["efficiency"])
    active = eta > 0
    if np.sum(active) > 1:
        assert np.all(np.diff(eta[active]) <= 0)


def test_useful_heat_increases_with_irradiance(model):
    G = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"irradiance": G, "T_inlet": 60.0, "T_ambient": 20.0})
    Q_u = np.asarray(r["useful_heat_w"])
    assert np.all(np.diff(Q_u) > 0)


def test_outlet_above_inlet(model):
    r = model.predict({"irradiance": 800.0, "T_inlet": 60.0, "T_ambient": 10.0})
    assert float(r["T_outlet_approx"]) >= 60.0


def test_outlet_equals_inlet_at_zero_Q(model):
    r = model.predict({"irradiance": 0.0, "T_inlet": 70.0, "T_ambient": 25.0})
    assert abs(float(r["T_outlet_approx"]) - 70.0) < 1e-9


def test_efficiency_non_negative(model):
    T_in = np.linspace(10, 110, 30)
    r = model.predict({"irradiance": 500.0, "T_inlet": T_in, "T_ambient": 20.0})
    assert np.all(np.asarray(r["efficiency"]) >= 0.0)


def test_benchmark(model):
    G = np.random.uniform(0, 1200, 1000)
    T_in = np.random.uniform(10, 110, 1000)
    T_amb = np.random.uniform(-10, 40, 1000)
    start = time.perf_counter()
    model.predict({"irradiance": G, "T_inlet": T_in, "T_ambient": T_amb})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
