"""EC168 — MPPT Controller — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"irradiance": 1000.0, "p_mpp_input": 10000.0})
    for k in ["p_output", "tracking_efficiency", "power_loss"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC168"
    assert "p_output" in info["outputs"]


def test_eta_never_exceeds_one(model):
    """Tracking efficiency must never exceed 1."""
    G = np.linspace(0, 1200, 200)
    p_in = G * 10.0
    r = model.predict({"irradiance": G, "p_mpp_input": p_in})
    assert np.all(r["tracking_efficiency"] <= 1.0 + 1e-9), "eta > 1 detected"


def test_eta_zero_at_zero_irradiance(model):
    """At G=0, no power available, efficiency must be 0."""
    r = model.predict({"irradiance": 0.0, "p_mpp_input": 0.0})
    assert float(r["tracking_efficiency"]) == 0.0


def test_eta_increases_with_irradiance(model):
    """Tracking efficiency must increase with irradiance."""
    G = np.linspace(10, 1200, 50)
    r = model.predict({"irradiance": G, "p_mpp_input": G * 10.0})
    assert np.all(np.diff(r["tracking_efficiency"]) > 0), "eta not monotonically increasing"


def test_eta_high_at_rated_irradiance(model):
    """At G=1000 W/m2 (STC), eta should be > 0.98 (near eta_max=0.99 with k=5)."""
    r = model.predict({"irradiance": 1000.0, "p_mpp_input": 10000.0})
    assert float(r["tracking_efficiency"]) > 0.98, f"eta={float(r['tracking_efficiency']):.4f} too low at G=1000"


def test_eta_low_at_low_irradiance(model):
    """At G < 100 W/m2, eta should be noticeably below 0.90 (roll-off visible)."""
    r = model.predict({"irradiance": 50.0, "p_mpp_input": 500.0})
    assert float(r["tracking_efficiency"]) < 0.90, f"eta={float(r['tracking_efficiency']):.4f} too high at G=50"


def test_p_out_less_than_or_equal_p_in(model):
    """Output power must never exceed input power."""
    G = np.random.uniform(0, 1200, 500)
    p_in = G * 10.0
    r = model.predict({"irradiance": G, "p_mpp_input": p_in})
    assert np.all(r["p_output"] <= p_in + 1e-6), "P_out > P_in detected"


def test_power_loss_non_negative(model):
    """Power loss must be non-negative."""
    G = np.linspace(0, 1200, 100)
    p_in = G * 10.0
    r = model.predict({"irradiance": G, "p_mpp_input": p_in})
    assert np.all(r["power_loss"] >= -1e-9), "Negative power loss detected"


def test_power_balance(model):
    """P_out + power_loss == P_in."""
    G = np.linspace(50, 1000, 100)
    p_in = G * 9.5
    r = model.predict({"irradiance": G, "p_mpp_input": p_in})
    residual = np.abs(r["p_output"] + r["power_loss"] - p_in)
    assert np.all(residual < 1e-6), "Power balance violated"


def test_benchmark(model):
    G = np.random.uniform(0, 1200, 1000)
    p_in = G * 10.0
    start = time.perf_counter()
    model.predict({"irradiance": G, "p_mpp_input": p_in})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
