"""EC166 -- AC-DC Diode Bridge Rectifier -- F1b Loss Model -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"v_ac_rms": 400.0, "i_dc": 30.0})
    expected = ["v_dc", "efficiency", "p_loss_w", "p_conduction_w", "p_recovery_w", "t_j_degc"]
    for k in expected:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC166"
    assert info["fidelity"] == "F1b"


def test_dc_voltage_3phase(model):
    """3-phase: V_dc = 1.3505 * V_LL_rms (ideal)."""
    v_ll = 400.0
    r = model.predict({"v_ac_rms": v_ll, "i_dc": 1.0})
    v_dc_expected = 1.3505 * v_ll
    # Allow small deviation (losses shift operating point slightly, but dc_voltage is ideal)
    assert abs(float(r["v_dc"]) - v_dc_expected) < 1.0, \
        f"V_dc={float(r['v_dc']):.2f}, expected ~{v_dc_expected:.2f}"


def test_efficiency_in_bounds(model):
    """Efficiency must be (0.9, 1.0) for typical operating point."""
    r = model.predict({"v_ac_rms": 400.0, "i_dc": 30.0})
    eta = float(r["efficiency"])
    assert 0.90 < eta < 1.0, f"Efficiency {eta:.4f} outside expected range"


def test_losses_increase_with_current(model):
    """Total losses must increase with DC current (I^2 dominant)."""
    r1 = model.predict({"v_ac_rms": 400.0, "i_dc": 10.0})
    r2 = model.predict({"v_ac_rms": 400.0, "i_dc": 20.0})
    assert float(r2["p_loss_w"]) > float(r1["p_loss_w"]), \
        "Losses must increase with current"


def test_conduction_loss_scales_with_i_squared(model):
    """r_d * I_rms^2 part: quadratic in I_dc; at low V_f contribution, check ratio."""
    # At high current the I^2 term dominates. Check ratio of conduction losses at 2x current.
    # If V_f*I term and I^2 term coexist, ratio is between 2 and 4.
    r1 = model.predict({"v_ac_rms": 400.0, "i_dc": 20.0})
    r2 = model.predict({"v_ac_rms": 400.0, "i_dc": 40.0})
    ratio = float(r2["p_conduction_w"]) / float(r1["p_conduction_w"])
    assert 2.0 < ratio < 4.5, f"Conduction loss ratio {ratio:.3f}, expected between 2 and 4.5"


def test_zero_current_zero_losses(model):
    """Zero DC current → zero losses."""
    r = model.predict({"v_ac_rms": 400.0, "i_dc": 0.0})
    assert abs(float(r["p_loss_w"])) < 1e-9


def test_junction_temperature_above_ambient(model):
    T_a = model.params["unit"]["T_a"]["value"]
    r = model.predict({"v_ac_rms": 400.0, "i_dc": 30.0})
    assert float(r["t_j_degc"]) > T_a


def test_loss_breakdown_sums_to_total(model):
    r = model.predict({"v_ac_rms": 400.0, "i_dc": 30.0})
    total = float(r["p_conduction_w"]) + float(r["p_recovery_w"])
    assert abs(total - float(r["p_loss_w"])) < 1e-9


def test_vectorized(model):
    i_dc = np.linspace(5, 50, 20)
    r = model.predict({"v_ac_rms": 400.0, "i_dc": i_dc})
    assert len(r["efficiency"]) == 20


def test_benchmark(model):
    i_dc = np.random.uniform(5, 100, 1000)
    start = time.perf_counter()
    model.predict({"v_ac_rms": 400.0, "i_dc": i_dc})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
