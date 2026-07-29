"""EC058 — Flat Plate Solar Collector — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"irradiance": 800.0, "T_inlet": 40.0, "T_ambient": 20.0})
    for k in ["useful_heat_w", "efficiency", "T_outlet_approx"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC058"
    assert info["fidelity"] == "F1a"


def test_zero_irradiance_gives_zero_heat(model):
    """No solar radiation -> no useful heat gain (or heat loss, capped at 0)."""
    r = model.predict({"irradiance": 0.0, "T_inlet": 40.0, "T_ambient": 20.0})
    assert float(r["useful_heat_w"]) == 0.0


def test_high_inlet_temp_stagnation(model):
    """When T_inlet is very high relative to irradiance, Q_u should be 0 (stagnation)."""
    r = model.predict({"irradiance": 100.0, "T_inlet": 90.0, "T_ambient": 10.0})
    assert float(r["useful_heat_w"]) == 0.0, \
        f"Expected 0 at stagnation, got {float(r['useful_heat_w']):.2f}W"


def test_efficiency_below_FR_tau_alpha(model):
    """Instantaneous eta must be <= F_R * tau_alpha = 0.75 (optical limit)."""
    G = np.linspace(100, 1200, 50)
    r = model.predict({"irradiance": G, "T_inlet": 20.0, "T_ambient": 20.0})
    eta = np.asarray(r["efficiency"])
    assert np.all(eta <= 0.75 + 1e-9), f"Efficiency exceeded F_R_tau_alpha: max={eta.max():.4f}"


def test_eta_decreases_with_temperature_parameter(model):
    """Efficiency decreases as (T_in - T_amb)/G increases (HWB characteristic)."""
    G = 800.0
    T_amb = 20.0
    T_ins = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    r = model.predict({"irradiance": G, "T_inlet": T_ins, "T_ambient": T_amb})
    eta = np.asarray(r["efficiency"])
    active = eta > 0
    if np.sum(active) > 1:
        assert np.all(np.diff(eta[active]) <= 0), f"Efficiency should decrease with T_in: {eta}"


def test_useful_heat_increases_with_irradiance(model):
    """More solar -> more useful heat."""
    G = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"irradiance": G, "T_inlet": 40.0, "T_ambient": 20.0})
    Q_u = np.asarray(r["useful_heat_w"])
    assert np.all(np.diff(Q_u) > 0), f"Useful heat must increase with G: {Q_u}"


def test_outlet_above_inlet(model):
    """Outlet temperature must be >= inlet temperature when Q_u > 0."""
    r = model.predict({"irradiance": 800.0, "T_inlet": 40.0, "T_ambient": 20.0})
    assert float(r["T_outlet_approx"]) >= 40.0


def test_outlet_equals_inlet_at_zero_Q(model):
    """At zero irradiance, T_outlet should equal T_inlet."""
    r = model.predict({"irradiance": 0.0, "T_inlet": 50.0, "T_ambient": 25.0})
    assert abs(float(r["T_outlet_approx"]) - 50.0) < 1e-9


def test_efficiency_non_negative(model):
    """Efficiency must be >= 0 (clamped by max(0, Q_u))."""
    T_ins = np.linspace(10, 90, 30)
    r = model.predict({"irradiance": 500.0, "T_inlet": T_ins, "T_ambient": 20.0})
    assert np.all(np.asarray(r["efficiency"]) >= 0.0)


def test_benchmark(model):
    G = np.random.uniform(0, 1200, 1000)
    T_in = np.random.uniform(10, 85, 1000)
    T_amb = np.random.uniform(-5, 40, 1000)
    start = time.perf_counter()
    model.predict({"irradiance": G, "T_inlet": T_in, "T_ambient": T_amb})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
