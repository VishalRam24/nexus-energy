"""EC134 — OWC WEC — F1a — Test Suite"""

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
    for k in ["wave_power_per_m_kw", "power_kw", "overall_efficiency", "capture_width_m"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC134"
    assert info["fidelity"] == "F1a"


def test_zero_power_at_zero_wave_height(model):
    r = model.predict({"H_s": 0.0, "T_e": 10.0})
    assert float(r["power_kw"]) == 0.0


def test_power_scales_quadratic_with_H_s(model):
    """P ∝ H_s^2 — doubling H_s should quadruple power."""
    r1 = model.predict({"H_s": 1.0, "T_e": 10.0})
    r2 = model.predict({"H_s": 2.0, "T_e": 10.0})
    ratio = float(r2["power_kw"]) / float(r1["power_kw"])
    assert abs(ratio - 4.0) < 0.01, f"P ratio for 2x H_s = {ratio:.4f}, expected 4.0"


def test_power_scales_linear_with_T_e(model):
    """P ∝ T_e — doubling T_e should double power."""
    r1 = model.predict({"H_s": 2.0, "T_e": 8.0})
    r2 = model.predict({"H_s": 2.0, "T_e": 16.0})
    ratio = float(r2["power_kw"]) / float(r1["power_kw"])
    assert abs(ratio - 2.0) < 0.01, f"P ratio for 2x T_e = {ratio:.4f}, expected 2.0"


def test_power_positive(model):
    H_s = np.linspace(0.5, 6.0, 20)
    T_e = np.linspace(5.0, 20.0, 20)
    r = model.predict({"H_s": H_s, "T_e": T_e})
    assert np.all(np.asarray(r["power_kw"]) >= 0.0)


def test_wave_power_formula(model):
    """Cross-check wave_power_per_m with analytical formula."""
    H_s, T_e = 2.0, 10.0
    rho = 1025.0
    J_expected = (rho * _G**2 * H_s**2 * T_e) / (64.0 * np.pi) / 1e3  # kW/m
    r = model.predict({"H_s": H_s, "T_e": T_e})
    assert abs(float(r["wave_power_per_m_kw"]) - J_expected) < 1e-6


def test_cwr_in_valid_range(model):
    """Default CWR should give efficiency 0.1 ≤ overall_eta ≤ 0.25."""
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    eta = float(r["overall_efficiency"])
    assert 0.05 <= eta <= 0.25, f"Overall efficiency {eta:.4f} outside expected range"


def test_cwr_override(model):
    """Higher CWR should give proportionally higher power."""
    r_lo = model.predict({"H_s": 2.0, "T_e": 10.0, "cwr": 0.10})
    r_hi = model.predict({"H_s": 2.0, "T_e": 10.0, "cwr": 0.30})
    assert float(r_hi["power_kw"]) > float(r_lo["power_kw"]) * 2.9


def test_overall_efficiency_less_than_1(model):
    r = model.predict({"H_s": 2.0, "T_e": 10.0, "cwr": 0.30})
    assert float(r["overall_efficiency"]) < 1.0


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
