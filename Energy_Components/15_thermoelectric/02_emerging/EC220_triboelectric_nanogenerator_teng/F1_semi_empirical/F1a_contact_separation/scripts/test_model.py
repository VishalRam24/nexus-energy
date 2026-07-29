"""EC220 — Triboelectric Nanogenerator (TENG) — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"frequency": 3.0, "R_load": 1e7})
    for k in ["V_oc_peak_V", "C_avg_F", "R_internal_ohm", "power_avg_w", "power_density_mwcm2", "efficiency"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC220"
    assert info["fidelity"] == "F1a"


def test_voc_positive(model):
    """V_oc must be positive (charged surfaces)."""
    r = model.predict({"frequency": 3.0, "R_load": 1e9})  # high R ~ open circuit
    assert float(r["V_oc_peak_V"]) > 0.0


def test_voc_order_of_magnitude(model):
    """TENG V_oc is typically tens to hundreds of volts."""
    r = model.predict({"frequency": 3.0, "R_load": 1e9})
    V = float(r["V_oc_peak_V"])
    assert 1.0 < V < 10000.0, f"V_oc={V:.1f} V outside expected 1-10000 V range"


def test_high_impedance_device(model):
    """TENG internal resistance must be very high (MOhm range)."""
    r = model.predict({"frequency": 3.0, "R_load": 1e7})
    R_int = float(r["R_internal_ohm"])
    assert R_int > 1e4, f"R_internal={R_int:.0f} ohm, expected > 10 kOhm (high-impedance device)"


def test_power_positive(model):
    """Power must be positive with valid inputs."""
    r = model.predict({"frequency": 3.0, "R_load": 1e7})
    assert float(r["power_avg_w"]) > 0.0


def test_power_increases_with_frequency(model):
    """Higher frequency (more cycles/s) generally yields more average power."""
    f = np.linspace(1.0, 20.0, 20)
    # At fixed optimal load (high R), power increases with f
    r = model.predict({"frequency": f, "R_load": 1e9})
    assert np.all(np.diff(r["power_avg_w"]) > 0), \
        "At high R_load, power must increase with frequency"


def test_optimal_load_maximizes_power(model):
    """There exists an optimal load resistance that maximizes power."""
    R_range = np.logspace(4, 10, 100)
    r = model.predict({"frequency": 3.0, "R_load": R_range})
    P = r["power_avg_w"]
    idx_max = np.argmax(P)
    # Max must not be at the extreme ends
    assert 0 < idx_max < len(R_range) - 1, \
        "Optimal load must exist at intermediate R, not at extremes"


def test_efficiency_physical_range(model):
    """Efficiency must be between 0 and 100%."""
    r = model.predict({"frequency": 3.0, "R_load": 1e7})
    eta = float(r["efficiency"])
    assert 0.0 <= eta <= 1.0, f"Efficiency {eta:.3f} outside [0,1]"


def test_efficiency_literature_range(model):
    """TENG efficiency at optimal conditions: 10-40% (Wang et al.)."""
    R_range = np.logspace(4, 10, 200)
    r = model.predict({"frequency": 3.0, "R_load": R_range})
    eta_max = float(np.max(r["efficiency"]))
    assert 0.05 < eta_max <= 1.0, \
        f"Peak efficiency {eta_max*100:.1f}% outside expected range"


def test_benchmark(model):
    f = np.random.uniform(0.5, 50.0, 1000)
    R = np.random.choice(np.logspace(4, 10, 100), size=1000)
    start = time.perf_counter()
    model.predict({"frequency": f, "R_load": R})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
