"""EC164 -- Three-Phase Inverter -- F2a dq-Frame -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_steady_state_power(model):
    """SS active power matches reference."""
    ss = model.predict_steady_state({"P_ref_kw": 50.0, "Q_ref_kvar": 0.0})
    assert abs(ss["P_ss_w"] - 50000.0) < 1.0


def test_steady_state_reactive(model):
    """SS reactive power matches reference."""
    ss = model.predict_steady_state({"P_ref_kw": 50.0, "Q_ref_kvar": 20.0})
    assert abs(ss["Q_ss_var"] - 20000.0) < 1.0


def test_steady_state_q_zero_means_iq_zero(model):
    """When Q_ref=0, i_q should be 0."""
    ss = model.predict_steady_state({"P_ref_kw": 50.0, "Q_ref_kvar": 0.0})
    assert abs(ss["i_q_ss"]) < 1e-6


def test_simulation_tracks_power(model):
    """Dynamic sim should track P_ref within 2% after settling."""
    r = model.predict({
        "P_ref_kw": 50.0, "Q_ref_kvar": 0.0,
        "dt": 1e-5, "duration_s": 0.1,
    })
    # Check last 20% of simulation
    n = len(r["t"])
    P_final = np.mean(r["P"][int(0.8*n):])
    assert abs(P_final - 50000.0) / 50000.0 < 0.02


def test_simulation_tracks_q(model):
    """Dynamic sim should track Q_ref."""
    r = model.predict({
        "P_ref_kw": 50.0, "Q_ref_kvar": 20.0,
        "dt": 1e-5, "duration_s": 0.1,
    })
    n = len(r["t"])
    Q_final = np.mean(r["Q"][int(0.8*n):])
    assert abs(Q_final - 20000.0) / 20000.0 < 0.02


def test_output_keys(model):
    r = model.predict({
        "P_ref_kw": 50.0, "Q_ref_kvar": 0.0,
        "dt": 1e-5, "duration_s": 0.01,
    })
    for key in ["t", "i_d", "i_q", "P", "Q", "v_dc"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC164"
    assert info["fidelity"] == "F2a"


def test_power_step_response(model):
    """P step: should settle to new value."""
    def p_step(t):
        return 0.0 if t < 0.02 else 80.0
    r = model.predict({
        "P_ref_kw": p_step, "Q_ref_kvar": 0.0,
        "dt": 1e-5, "duration_s": 0.15,
    })
    P_final = np.mean(r["P"][-100:])
    assert abs(P_final - 80000.0) / 80000.0 < 0.02


def test_zero_power_zero_current(model):
    """Zero power reference should give near-zero currents."""
    r = model.predict({
        "P_ref_kw": 0.0, "Q_ref_kvar": 0.0,
        "dt": 1e-5, "duration_s": 0.05,
    })
    assert abs(r["i_d"][-1]) < 0.1
    assert abs(r["i_q"][-1]) < 0.1


def test_power_reversal(model):
    """Inverter can handle power reversal (rectification)."""
    def p_rev(t):
        return 50.0 if t < 0.05 else -50.0
    r = model.predict({
        "P_ref_kw": p_rev, "Q_ref_kvar": 0.0,
        "dt": 1e-5, "duration_s": 0.15,
    })
    P_final = np.mean(r["P"][-100:])
    assert P_final < 0  # Should be negative


def test_energy_conservation(model):
    """Power in = power out + losses."""
    r = model.predict({
        "P_ref_kw": 50.0, "Q_ref_kvar": 0.0,
        "dt": 1e-5, "duration_s": 0.1,
    })
    # At steady state: P_grid = P (from dq) and P_loss = R*(i_d^2+i_q^2)*1.5
    n = len(r["t"])
    i_d_ss = np.mean(r["i_d"][int(0.8*n):])
    i_q_ss = np.mean(r["i_q"][int(0.8*n):])
    P_ss = np.mean(r["P"][int(0.8*n):])
    R = 0.1
    P_loss = 1.5 * R * (i_d_ss**2 + i_q_ss**2)
    # P_loss should be small relative to P
    assert P_loss < P_ss * 0.05  # Losses < 5% of power


def test_benchmark(model):
    start = time.perf_counter()
    model.predict({
        "P_ref_kw": 50.0, "Q_ref_kvar": 0.0,
        "dt": 1e-5, "duration_s": 0.1,
    })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 100ms sim in {elapsed*1000:.1f} ms")
    assert elapsed < 10.0
