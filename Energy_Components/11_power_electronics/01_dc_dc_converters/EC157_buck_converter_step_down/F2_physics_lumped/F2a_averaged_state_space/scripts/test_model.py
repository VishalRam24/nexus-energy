"""EC157 -- Buck Converter -- F2a Averaged SSM -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import BuckConverterF2a


@pytest.fixture
def model():
    return ComponentModel()


@pytest.fixture
def raw_model(model):
    return model._model


# ---- Steady-state tests ----

def test_steady_state_voltage(model):
    """Steady-state V_out = D*V_in*R/(R+R_L) for buck converter."""
    ss = model.predict_steady_state({"v_in": 48.0, "duty_cycle": 0.25, "R_load": 1.2})
    expected = 0.25 * 48.0 * 1.2 / (1.2 + 0.05)
    assert abs(ss["v_out_ss"] - expected) < 1e-6, \
        f"V_out_ss={ss['v_out_ss']:.6f}, expected={expected:.6f}"


def test_steady_state_current(model):
    """Steady-state I_L = V_out / R_load."""
    ss = model.predict_steady_state({"v_in": 48.0, "duty_cycle": 0.25, "R_load": 1.2})
    assert abs(ss["i_L_ss"] - ss["v_out_ss"] / 1.2) < 1e-6


def test_steady_state_power_conservation(model):
    """P_out + P_loss_RL = D*V_in*I_L (input power)."""
    ss = model.predict_steady_state({"v_in": 48.0, "duty_cycle": 0.25, "R_load": 1.2})
    p_in = 0.25 * 48.0 * ss["i_L_ss"]
    p_out = ss["power_ss"]
    p_loss = ss["i_L_ss"] ** 2 * 0.05  # R_L losses
    assert abs(p_in - p_out - p_loss) < 1e-6


# ---- Dynamic simulation tests ----

def test_simulation_settles_to_steady_state(model):
    """Dynamic simulation must converge to analytic steady state."""
    ss = model.predict_steady_state({"v_in": 48.0, "duty_cycle": 0.25, "R_load": 1.2})
    r = model.predict({
        "v_in": 48.0, "duty_cycle": 0.25, "R_load": 1.2,
        "dt": 1e-6, "duration_s": 0.02,
    })
    # Final values should match steady state within 0.1%
    assert abs(r["v_out"][-1] - ss["v_out_ss"]) / ss["v_out_ss"] < 1e-3, \
        f"V_out final={r['v_out'][-1]:.4f}, SS={ss['v_out_ss']:.4f}"
    assert abs(r["i_L"][-1] - ss["i_L_ss"]) / ss["i_L_ss"] < 1e-3


def test_simulation_starts_from_zero(model):
    """Simulation starts from zero initial conditions."""
    r = model.predict({
        "v_in": 48.0, "duty_cycle": 0.25, "R_load": 1.2,
        "dt": 1e-6, "duration_s": 0.001,
    })
    assert r["v_out"][0] == 0.0
    assert r["i_L"][0] == 0.0


def test_output_keys(model):
    """Check all expected output keys."""
    r = model.predict({
        "v_in": 48.0, "duty_cycle": 0.25, "R_load": 1.2,
        "dt": 1e-6, "duration_s": 0.001,
    })
    for key in ["t", "v_out", "i_L", "i_out", "power"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC157"
    assert info["fidelity"] == "F2a"


def test_load_step_response(model):
    """Load step: output must settle to new steady state."""
    def r_load(t):
        return 1.2 if t < 0.005 else 2.4

    r = model.predict({
        "v_in": 48.0, "duty_cycle": 0.25, "R_load": r_load,
        "dt": 1e-6, "duration_s": 0.02,
    })
    ss_new = model.predict_steady_state({"v_in": 48.0, "duty_cycle": 0.25, "R_load": 2.4})
    # Should settle within 1% of new steady state
    assert abs(r["v_out"][-1] - ss_new["v_out_ss"]) / ss_new["v_out_ss"] < 0.01


def test_duty_cycle_step(model):
    """Duty cycle step: output voltage must increase."""
    def duty(t):
        return 0.25 if t < 0.005 else 0.40

    r = model.predict({
        "v_in": 48.0, "duty_cycle": duty, "R_load": 1.2,
        "dt": 1e-6, "duration_s": 0.02,
    })
    ss_new = model.predict_steady_state({"v_in": 48.0, "duty_cycle": 0.40, "R_load": 1.2})
    assert abs(r["v_out"][-1] - ss_new["v_out_ss"]) / ss_new["v_out_ss"] < 0.01


def test_energy_conservation(model):
    """Total energy in ~ total energy out + losses over simulation."""
    r = model.predict({
        "v_in": 48.0, "duty_cycle": 0.25, "R_load": 1.2,
        "dt": 1e-6, "duration_s": 0.02,
    })
    dt_arr = np.diff(r["t"])
    # Input power: P_in = D * V_in * i_L
    p_in = 0.25 * 48.0 * r["i_L"][1:]
    # Output power
    p_out = r["power"][1:]
    # Loss in R_L
    p_loss = r["i_L"][1:] ** 2 * 0.05

    E_in = np.sum(p_in * dt_arr)
    E_out = np.sum(p_out * dt_arr)
    E_loss = np.sum(p_loss * dt_arr)
    # Energy stored in L and C at end
    L = 100e-6
    C = 100e-6
    E_stored = 0.5 * L * r["i_L"][-1]**2 + 0.5 * C * r["v_out"][-1]**2

    balance = E_in - E_out - E_loss - E_stored
    # Should be near zero (relative to E_in)
    assert abs(balance) / E_in < 0.01, \
        f"Energy balance error: {balance:.6f} J, E_in={E_in:.6f} J"


def test_benchmark(model):
    """Benchmark: 10ms simulation should complete in < 5s."""
    start = time.perf_counter()
    model.predict({
        "v_in": 48.0, "duty_cycle": 0.25, "R_load": 1.2,
        "dt": 1e-6, "duration_s": 0.01,
    })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 10ms sim in {elapsed*1000:.1f} ms")
    assert elapsed < 5.0
