"""EC219 — Piezoelectric Energy Harvester — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"acceleration": 9.81, "frequency": 100.0})
    for k in ["power_w", "power_uw", "voltage_v", "frequency_ratio", "at_resonance_power_w"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC219"
    assert info["fidelity"] == "F1a"


def test_power_scales_with_acceleration_squared(model):
    """Fundamental piezo physics: P ~ a^2 at resonance."""
    f_n = model._model.f_n
    a1, a2 = 5.0, 10.0  # ratio = 2
    r1 = model.predict({"acceleration": a1, "frequency": f_n})
    r2 = model.predict({"acceleration": a2, "frequency": f_n})
    ratio = float(r2["power_w"]) / float(r1["power_w"])
    # P ~ a^2: doubling a quadruples P
    assert abs(ratio - 4.0) < 0.1, \
        f"P/a^2 scaling violated: ratio={ratio:.3f}, expected 4.0"


def test_power_maximum_at_resonance(model):
    """Power must be maximum at resonant frequency."""
    f_n = model._model.f_n
    f_off_low = f_n * 0.7
    f_off_high = f_n * 1.3
    a = 9.81
    r_res = model.predict({"acceleration": a, "frequency": f_n})
    r_low = model.predict({"acceleration": a, "frequency": f_off_low})
    r_high = model.predict({"acceleration": a, "frequency": f_off_high})
    assert float(r_res["power_w"]) > float(r_low["power_w"]), \
        "Power at resonance must exceed off-resonance (low freq)"
    assert float(r_res["power_w"]) > float(r_high["power_w"]), \
        "Power at resonance must exceed off-resonance (high freq)"


def test_power_increases_with_acceleration(model):
    """Power must increase monotonically with acceleration."""
    f_n = model._model.f_n
    a = np.linspace(1.0, 20.0, 20)
    r = model.predict({"acceleration": a, "frequency": f_n})
    assert np.all(np.diff(r["power_w"]) > 0), \
        "Power must increase monotonically with acceleration"


def test_power_positive(model):
    """Power must be non-negative for positive acceleration."""
    r = model.predict({"acceleration": 9.81, "frequency": 100.0})
    assert float(r["power_w"]) > 0.0


def test_power_zero_at_zero_acceleration(model):
    """Zero acceleration gives zero power."""
    r = model.predict({"acceleration": 0.0, "frequency": 100.0})
    assert float(r["power_w"]) == pytest.approx(0.0, abs=1e-18)


def test_power_scale_microwatt_to_milliwatt(model):
    """At 1g, 100Hz: power in uW-mW range (not kW or pW)."""
    r = model.predict({"acceleration": 9.81, "frequency": 100.0})
    P_uw = float(r["power_uw"])
    assert 0.1 < P_uw < 100000.0, \
        f"Power {P_uw:.2f} uW out of expected uW-mW range"


def test_frequency_ratio_correct(model):
    """frequency_ratio must equal f / f_n."""
    f_n = model._model.f_n
    test_f = 150.0
    r = model.predict({"acceleration": 9.81, "frequency": test_f})
    expected_ratio = test_f / f_n
    assert float(r["frequency_ratio"]) == pytest.approx(expected_ratio, rel=1e-6)


def test_benchmark(model):
    a = np.random.uniform(0.5, 30.0, 1000)
    f = np.random.uniform(10.0, 500.0, 1000)
    start = time.perf_counter()
    model.predict({"acceleration": a, "frequency": f})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
