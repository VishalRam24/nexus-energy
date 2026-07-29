"""EC183 — Circuit Breaker — F1a Loss Model — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"I_A": 400.0, "state": "closed"})
    for k in ["P_loss_W", "is_overloaded", "can_interrupt", "E_fault_J", "thermal_rating_ok"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC183"
    assert info["fidelity"] == "F1a"


def test_conduction_loss_formula(model):
    """P_loss = I^2 * R_cb exactly."""
    I = 400.0
    R_cb = model._model.R_cb
    r = model.predict({"I_A": I, "state": "closed"})
    expected = I**2 * R_cb
    assert abs(float(r["P_loss_W"]) - expected) < 1e-12


def test_open_state_zero_loss(model):
    """Open breaker carries no current → zero loss."""
    r = model.predict({"I_A": 500.0, "state": "open"})
    assert float(r["P_loss_W"]) == 0.0


def test_loss_scales_with_I_squared(model):
    """Doubling current quadruples loss."""
    r1 = model.predict({"I_A": 200.0, "state": "closed"})
    r2 = model.predict({"I_A": 400.0, "state": "closed"})
    ratio = float(r2["P_loss_W"]) / float(r1["P_loss_W"])
    assert abs(ratio - 4.0) < 1e-9, f"Loss ratio={ratio:.6f}, expected 4.0"


def test_overload_detection(model):
    """Current above rated → is_overloaded = True."""
    r = model.predict({"I_A": 700.0, "state": "closed"})
    assert bool(r["is_overloaded"]) is True


def test_no_overload_at_rated(model):
    """Exactly at rated current → not overloaded."""
    r = model.predict({"I_A": 630.0, "state": "closed"})
    assert bool(r["is_overloaded"]) is False


def test_can_interrupt_below_rating(model):
    """Fault current <= rating → can_interrupt = True."""
    r = model.predict({"I_A": 400.0, "state": "closed", "I_fault_kA": 20.0})
    assert bool(r["can_interrupt"]) is True


def test_cannot_interrupt_above_rating(model):
    """Fault current > rating → can_interrupt = False."""
    r = model.predict({"I_A": 400.0, "state": "closed", "I_fault_kA": 30.0})
    assert bool(r["can_interrupt"]) is False


def test_fault_energy_positive(model):
    """Fault energy must be positive for nonzero fault current."""
    r = model.predict({"I_A": 0.0, "state": "closed", "I_fault_kA": 15.0})
    assert float(r["E_fault_J"]) > 0.0


def test_zero_current_zero_loss(model):
    r = model.predict({"I_A": 0.0, "state": "closed"})
    assert float(r["P_loss_W"]) == 0.0


def test_vectorized_current(model):
    I = np.linspace(0, 630, 50)
    r = model.predict({"I_A": I, "state": "closed"})
    assert r["P_loss_W"].shape == (50,)
    # Loss must be monotonically increasing with I
    assert np.all(np.diff(r["P_loss_W"]) >= 0)


def test_benchmark(model):
    I = np.random.uniform(0, 630, 1000)
    start = time.perf_counter()
    model.predict({"I_A": I, "state": "closed"})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
