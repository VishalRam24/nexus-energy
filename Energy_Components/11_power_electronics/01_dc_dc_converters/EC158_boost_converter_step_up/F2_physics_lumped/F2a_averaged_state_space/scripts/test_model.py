"""EC158 -- Boost Converter -- F2a Averaged SSM -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import BoostConverterF2a


@pytest.fixture
def model():
    return ComponentModel()


# ---- Steady-state tests ----

def test_steady_state_voltage(model):
    """Steady-state V_out for boost with losses."""
    ss = model.predict_steady_state({"v_in": 12.0, "duty_cycle": 0.75, "R_load": 48.0})
    D_prime = 0.25
    expected = 12.0 * D_prime * 48.0 / (D_prime**2 * 48.0 + 0.1)
    assert abs(ss["v_out_ss"] - expected) < 1e-6


def test_steady_state_boost_ratio(model):
    """Ideal boost: V_out ~ V_in/(1-D) for low R_L."""
    ss = model.predict_steady_state({"v_in": 12.0, "duty_cycle": 0.50, "R_load": 100.0})
    ideal = 12.0 / 0.50
    # With small R_L=0.1, should be close to ideal
    assert abs(ss["v_out_ss"] - ideal) / ideal < 0.01  # within 1%


def test_steady_state_current_balance(model):
    """Steady state: (1-D)*i_L = v_C/R_load."""
    ss = model.predict_steady_state({"v_in": 12.0, "duty_cycle": 0.75, "R_load": 48.0})
    lhs = 0.25 * ss["i_L_ss"]
    rhs = ss["v_out_ss"] / 48.0
    assert abs(lhs - rhs) < 1e-6


def test_steady_state_power_conservation(model):
    """P_in = P_out + P_loss_RL."""
    ss = model.predict_steady_state({"v_in": 12.0, "duty_cycle": 0.75, "R_load": 48.0})
    p_in = 12.0 * ss["i_L_ss"]
    p_out = ss["power_ss"]
    p_loss = ss["i_L_ss"]**2 * 0.1
    assert abs(p_in - p_out - p_loss) < 1e-6


# ---- Dynamic simulation tests ----

def test_simulation_settles(model):
    """Dynamic sim must converge to steady state."""
    ss = model.predict_steady_state({"v_in": 12.0, "duty_cycle": 0.75, "R_load": 48.0})
    r = model.predict({
        "v_in": 12.0, "duty_cycle": 0.75, "R_load": 48.0,
        "dt": 5e-6, "duration_s": 0.1,
    })
    assert abs(r["v_out"][-1] - ss["v_out_ss"]) / ss["v_out_ss"] < 0.01


def test_output_keys(model):
    r = model.predict({
        "v_in": 12.0, "duty_cycle": 0.75, "R_load": 48.0,
        "dt": 5e-6, "duration_s": 0.005,
    })
    for key in ["t", "v_out", "i_L", "i_out", "power"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC158"
    assert info["fidelity"] == "F2a"


def test_load_step_settles(model):
    """Load step: must settle to new SS."""
    def r_load(t):
        return 48.0 if t < 0.025 else 24.0
    r = model.predict({
        "v_in": 12.0, "duty_cycle": 0.75, "R_load": r_load,
        "dt": 5e-6, "duration_s": 0.1,
    })
    ss_new = model.predict_steady_state({"v_in": 12.0, "duty_cycle": 0.75, "R_load": 24.0})
    assert abs(r["v_out"][-1] - ss_new["v_out_ss"]) / ss_new["v_out_ss"] < 0.02


def test_voltage_step_up(model):
    """Boost output must be higher than input in steady state."""
    ss = model.predict_steady_state({"v_in": 12.0, "duty_cycle": 0.75, "R_load": 48.0})
    assert ss["v_out_ss"] > 12.0


def test_energy_conservation(model):
    """Energy balance over simulation."""
    r = model.predict({
        "v_in": 12.0, "duty_cycle": 0.75, "R_load": 48.0,
        "dt": 5e-6, "duration_s": 0.1,
    })
    dt_arr = np.diff(r["t"])
    p_in = 12.0 * r["i_L"][1:]
    p_out = r["power"][1:]
    p_loss = r["i_L"][1:]**2 * 0.1

    E_in = np.sum(p_in * dt_arr)
    E_out = np.sum(p_out * dt_arr)
    E_loss = np.sum(p_loss * dt_arr)

    L = 200e-6
    C = 220e-6
    E_stored = 0.5 * L * r["i_L"][-1]**2 + 0.5 * C * r["v_out"][-1]**2
    E_stored_0 = 0.5 * C * r["v_out"][0]**2

    balance = E_in - E_out - E_loss - (E_stored - E_stored_0)
    assert abs(balance) / E_in < 0.01


def test_benchmark(model):
    start = time.perf_counter()
    model.predict({
        "v_in": 12.0, "duty_cycle": 0.75, "R_load": 48.0,
        "dt": 5e-6, "duration_s": 0.01,
    })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 10ms sim in {elapsed*1000:.1f} ms")
    assert elapsed < 5.0
