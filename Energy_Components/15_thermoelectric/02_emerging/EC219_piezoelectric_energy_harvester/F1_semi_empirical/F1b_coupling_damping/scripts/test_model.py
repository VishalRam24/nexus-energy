"""EC219 — Piezoelectric Harvester — F1b Coupling+Damping — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


@pytest.fixture
def R_opt(model):
    return float(model._model.compute(9.81, model._model.f_n, 1e4)["optimal_R_ohm"])


def test_predict_keys(model, R_opt):
    r = model.predict({"acceleration_ms2": 9.81, "frequency_hz": 100.0, "R_load_ohm": R_opt})
    for k in ["power_w", "power_uw", "voltage_v", "frequency_ratio",
              "zeta_electrical", "zeta_total", "optimal_R_ohm", "at_resonance_power_w"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC219"
    assert info["fidelity"] == "F1b"


def test_power_positive_at_resonance(model, R_opt):
    """Power must be positive at resonance with optimal load."""
    f_n = model._model.f_n
    r = model.predict({"acceleration_ms2": 9.81, "frequency_hz": f_n, "R_load_ohm": R_opt})
    P = float(np.atleast_1d(r["power_w"])[0])
    assert P > 0, f"Power at resonance = {P:.4e} W"


def test_power_scales_with_acceleration_squared(model, R_opt):
    """Power ~ a^2 (fundamental piezoelectric scaling)."""
    f_n = model._model.f_n
    r1 = model.predict({"acceleration_ms2": 1.0, "frequency_hz": f_n, "R_load_ohm": R_opt})
    r2 = model.predict({"acceleration_ms2": 2.0, "frequency_hz": f_n, "R_load_ohm": R_opt})
    P1 = float(np.atleast_1d(r1["power_w"])[0])
    P2 = float(np.atleast_1d(r2["power_w"])[0])
    ratio = P2 / (P1 + 1e-30)
    assert abs(ratio - 4.0) < 0.1, f"P(2a)/P(a) = {ratio:.4f}, expected ~4.0"


def test_power_peaks_at_resonance(model, R_opt):
    """Power should be maximum at resonance frequency."""
    f_n = model._model.f_n
    freqs = [f_n * 0.5, f_n * 0.8, f_n, f_n * 1.2, f_n * 2.0]
    powers = []
    for f in freqs:
        r = model.predict({"acceleration_ms2": 9.81, "frequency_hz": f, "R_load_ohm": R_opt})
        powers.append(float(np.atleast_1d(r["power_w"])[0]))
    # Power at resonance should be the maximum
    idx_fn = 2  # f_n is 3rd entry
    assert powers[idx_fn] == max(powers), \
        f"Peak not at resonance. Powers: {powers}"


def test_off_resonance_power_lower(model, R_opt):
    """Off-resonance power must be < resonance power."""
    f_n = model._model.f_n
    r_res = model.predict({"acceleration_ms2": 9.81, "frequency_hz": f_n, "R_load_ohm": R_opt})
    r_off = model.predict({"acceleration_ms2": 9.81, "frequency_hz": f_n * 2.0, "R_load_ohm": R_opt})
    P_res = float(np.atleast_1d(r_res["power_w"])[0])
    P_off = float(np.atleast_1d(r_off["power_w"])[0])
    assert P_res > P_off, f"Resonance power {P_res:.4e} <= off-resonance {P_off:.4e}"


def test_zeta_total_greater_than_zeta_mech(model, R_opt):
    """Total damping > mechanical damping (electrical extraction adds damping)."""
    f_n = model._model.f_n
    r = model.predict({"acceleration_ms2": 9.81, "frequency_hz": f_n, "R_load_ohm": R_opt})
    zeta_total = float(r["zeta_total"])
    zeta_mech = model._model.zeta_mech
    assert zeta_total > zeta_mech, f"zeta_total={zeta_total:.4f} <= zeta_mech={zeta_mech:.4f}"


def test_voltage_positive(model, R_opt):
    f_n = model._model.f_n
    r = model.predict({"acceleration_ms2": 9.81, "frequency_hz": f_n, "R_load_ohm": R_opt})
    V = float(np.atleast_1d(r["voltage_v"])[0])
    assert V > 0


def test_frequency_ratio_at_resonance(model, R_opt):
    f_n = model._model.f_n
    r = model.predict({"acceleration_ms2": 9.81, "frequency_hz": f_n, "R_load_ohm": R_opt})
    ratio = float(np.atleast_1d(r["frequency_ratio"])[0])
    assert abs(ratio - 1.0) < 1e-6


def test_optimal_R_reasonable(model, R_opt):
    """Optimal load ~1/(omega_n * C_p) should be in reasonable range."""
    f_n = model._model.f_n
    C_p = model._model.C_p
    R_expected = 1.0 / (2.0 * np.pi * f_n * C_p)
    r = model.predict({"acceleration_ms2": 9.81, "frequency_hz": f_n, "R_load_ohm": R_opt})
    R_opt_model = float(r["optimal_R_ohm"])
    assert 0.1 * R_expected < R_opt_model < 10.0 * R_expected, \
        f"R_opt={R_opt_model:.1f} vs expected {R_expected:.1f}"


def test_benchmark(model, R_opt):
    freqs = np.random.uniform(50, 200, 100)
    start = time.perf_counter()
    for f in freqs:
        model.predict({"acceleration_ms2": 9.81, "frequency_hz": float(f), "R_load_ohm": R_opt})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 100 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 5.0
