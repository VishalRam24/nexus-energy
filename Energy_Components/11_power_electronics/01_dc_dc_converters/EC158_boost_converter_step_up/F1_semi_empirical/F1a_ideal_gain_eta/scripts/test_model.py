"""EC158 — Boost Converter — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 5.0})
    for k in ["duty_cycle", "v_out", "efficiency", "p_loss_w", "i_input"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC158"
    assert info["fidelity"] == "F1a"


def test_duty_cycle_correct(model):
    """D = 1 - V_in/V_out for ideal boost."""
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 2.0})
    D = float(r["duty_cycle"])
    D_expected = 1.0 - 12.0 / 48.0  # = 0.75
    assert abs(D - D_expected) < 1e-9, f"D={D:.6f}, expected {D_expected:.6f}"


def test_v_out_equals_vin_over_one_minus_d(model):
    """V_out = V_in / (1-D) (ideal)."""
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 2.0})
    v_out = float(r["v_out"])
    assert abs(v_out - 48.0) < 1e-6, f"V_out={v_out:.6f}, expected 48.0"


def test_efficiency_less_than_one(model):
    """Efficiency < 1 always."""
    i_range = np.linspace(0.5, 10.0, 30)
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": i_range})
    assert np.all(r["efficiency"] < 1.0)


def test_efficiency_positive(model):
    """Efficiency > 0."""
    i_range = np.linspace(0.5, 10.0, 30)
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": i_range})
    assert np.all(r["efficiency"] > 0.0)


def test_losses_positive(model):
    """Losses > 0."""
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 5.0})
    assert float(r["p_loss_w"]) > 0.0


def test_input_current_greater_than_output(model):
    """
    For boost converter, I_in > I_out (current is amplified).
    I_in = I_out * V_out / V_in  =>  for 12V->48V, I_in = 4 * I_out
    """
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 5.0})
    i_in = float(r["i_input"])
    i_out = 5.0
    assert i_in > i_out, f"I_in ({i_in:.2f}A) should be > I_out ({i_out:.2f}A)"


def test_input_current_ratio(model):
    """I_in / I_out ≈ V_out / V_in = 4 for 12V->48V (ideal)."""
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 5.0})
    i_in = float(r["i_input"])
    i_out = 5.0
    ratio = i_in / i_out
    assert abs(ratio - 4.0) < 1e-6, f"I_in/I_out = {ratio:.6f}, expected 4.0 (ideal)"


def test_duty_cycle_increases_with_conversion_ratio(model):
    """Higher step-up ratio requires higher D."""
    r1 = model.predict({"v_in": 12.0, "v_out_target": 24.0, "i_load": 2.0})
    r2 = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 2.0})
    r3 = model.predict({"v_in": 12.0, "v_out_target": 96.0, "i_load": 2.0})
    D1, D2, D3 = float(r1["duty_cycle"]), float(r2["duty_cycle"]), float(r3["duty_cycle"])
    assert D1 < D2 < D3, f"D should increase with conversion ratio: {D1:.3f} < {D2:.3f} < {D3:.3f}"


def test_efficiency_reasonable(model):
    """Efficiency should be in reasonable range (85-99%) at nominal conditions."""
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 5.0})
    eta = float(r["efficiency"])
    assert 0.80 <= eta <= 0.999, f"eta={eta:.4f} outside expected [0.80, 0.999]"


def test_vectorized_input(model):
    """Array inputs must work."""
    i_range = np.linspace(0.5, 10.0, 30)
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": i_range})
    assert len(r["efficiency"]) == 30
    assert len(r["i_input"]) == 30


def test_benchmark(model):
    v_in = np.random.uniform(8, 30, 1000)
    i_load = np.random.uniform(0.5, 10.0, 1000)
    start = time.perf_counter()
    model.predict({"v_in": v_in, "v_out_target": 48.0, "i_load": i_load})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
