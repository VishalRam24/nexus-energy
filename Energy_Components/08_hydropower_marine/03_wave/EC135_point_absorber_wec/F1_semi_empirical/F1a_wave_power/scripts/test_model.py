"""EC135 — Point Absorber WEC — F1a — Test Suite"""

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
    for k in ["wave_power_per_m_kw", "power_kw", "capture_width_ratio", "overall_efficiency"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC135"
    assert info["fidelity"] == "F1a"


def test_zero_power_at_zero_height(model):
    r = model.predict({"H_s": 0.0, "T_e": 10.0})
    assert float(r["power_kw"]) == 0.0


def test_power_quadratic_with_H_s(model):
    """P ∝ H_s^2."""
    r1 = model.predict({"H_s": 1.0, "T_e": 10.0})
    r2 = model.predict({"H_s": 2.0, "T_e": 10.0})
    ratio = float(r2["power_kw"]) / float(r1["power_kw"])
    assert abs(ratio - 4.0) < 0.01, f"P ratio for 2x H_s = {ratio:.4f}, expected 4.0"


def test_resonance_peak(model):
    """CWR must be maximum at T_n (10s) vs off-resonance."""
    cwr_resonance = float(model.predict({"H_s": 2.0, "T_e": 10.0})["capture_width_ratio"])
    cwr_off1 = float(model.predict({"H_s": 2.0, "T_e": 5.0})["capture_width_ratio"])
    cwr_off2 = float(model.predict({"H_s": 2.0, "T_e": 18.0})["capture_width_ratio"])
    assert cwr_resonance > cwr_off1, "CWR at resonance not greater than off-resonance (T_e=5s)"
    assert cwr_resonance > cwr_off2, "CWR at resonance not greater than off-resonance (T_e=18s)"


def test_cwr_peak_not_exceeded(model):
    """CWR should never exceed cwr_peak (0.25)."""
    T_e = np.linspace(4.0, 20.0, 100)
    r = model.predict({"H_s": 2.0, "T_e": T_e})
    assert np.all(np.asarray(r["capture_width_ratio"]) <= 0.26), "CWR exceeds peak"


def test_overall_efficiency_less_than_1(model):
    T_e = np.linspace(4.0, 20.0, 50)
    r = model.predict({"H_s": 2.0, "T_e": T_e})
    assert np.all(np.asarray(r["overall_efficiency"]) <= 1.0)


def test_power_positive(model):
    H_s = np.linspace(0.5, 6.0, 20)
    T_e = np.linspace(5.0, 18.0, 20)
    r = model.predict({"H_s": H_s, "T_e": T_e})
    assert np.all(np.asarray(r["power_kw"]) >= 0.0)


def test_wave_power_formula(model):
    """Cross-check wave_power_per_m vs analytical formula."""
    H_s, T_e, rho = 2.0, 10.0, 1025.0
    J_expected = (rho * _G**2 * H_s**2 * T_e) / (64.0 * np.pi) / 1e3
    r = model.predict({"H_s": H_s, "T_e": T_e})
    assert abs(float(r["wave_power_per_m_kw"]) - J_expected) < 1e-6


def test_benchmark(model):
    H_s = np.random.uniform(0.5, 6.0, 1000)
    T_e = np.random.uniform(4.0, 20.0, 1000)
    start = time.perf_counter()
    model.predict({"H_s": H_s, "T_e": T_e})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
