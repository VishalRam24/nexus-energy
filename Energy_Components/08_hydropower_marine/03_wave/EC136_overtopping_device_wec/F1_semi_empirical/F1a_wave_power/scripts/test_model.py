"""EC136 — Overtopping Device WEC — F1a — Test Suite"""

import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

_G = 9.81


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    for k in ["wave_power_per_m_kw", "power_kw", "overall_efficiency"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC136"
    assert info["fidelity"] == "F1a"


def test_zero_power_at_zero_height(model):
    r = model.predict({"H_s": 0.0, "T_e": 10.0})
    assert float(r["power_kw"]) == 0.0


def test_power_quadratic_with_H_s(model):
    """P ∝ H_s^2."""
    r1 = model.predict({"H_s": 1.0, "T_e": 10.0})
    r2 = model.predict({"H_s": 2.0, "T_e": 10.0})
    ratio = float(r2["power_kw"]) / float(r1["power_kw"])
    assert abs(ratio - 4.0) < 0.01, f"Expected P ratio 4.0, got {ratio:.4f}"


def test_power_linear_with_T_e(model):
    """P ∝ T_e."""
    r1 = model.predict({"H_s": 2.0, "T_e": 8.0})
    r2 = model.predict({"H_s": 2.0, "T_e": 16.0})
    ratio = float(r2["power_kw"]) / float(r1["power_kw"])
    assert abs(ratio - 2.0) < 0.01, f"Expected P ratio 2.0, got {ratio:.4f}"


def test_efficiency_in_range(model):
    """Overall efficiency should be in realistic 0.10-0.25 range for overtopping device."""
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    eta = float(r["overall_efficiency"])
    # eta_ramp(0.20) * eta_turbine(0.80) * eta_gen(0.92) = 0.1472
    assert 0.10 <= eta <= 0.25, f"Overall efficiency {eta:.4f} outside expected 0.10-0.25"


def test_power_positive(model):
    H_s = np.linspace(0.5, 6.0, 20)
    T_e = np.linspace(5.0, 20.0, 20)
    r = model.predict({"H_s": H_s, "T_e": T_e})
    assert np.all(np.asarray(r["power_kw"]) >= 0.0)


def test_wave_power_formula(model):
    """Cross-check wave_power_per_m vs analytical formula."""
    H_s, T_e, rho = 2.0, 10.0, 1025.0
    J_expected = (rho * _G**2 * H_s**2 * T_e) / (64.0 * np.pi) / 1e3
    r = model.predict({"H_s": H_s, "T_e": T_e})
    assert abs(float(r["wave_power_per_m_kw"]) - J_expected) < 1e-6


def test_power_increases_with_H_s(model):
    H_s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = model.predict({"H_s": H_s, "T_e": 10.0})
    assert np.all(np.diff(r["power_kw"]) > 0)


def test_benchmark(model):
    H_s = np.random.uniform(0.5, 6.0, 1000)
    T_e = np.random.uniform(5.0, 20.0, 1000)
    start = time.perf_counter()
    model.predict({"H_s": H_s, "T_e": T_e})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
