"""EC175 -- Induction Motor -- F2a dq-Frame -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_output_keys(model):
    r = model.predict({
        "v_supply_rms": 400.0, "frequency_hz": 50.0, "T_load_Nm": 10.0,
        "dt": 1e-4, "duration_s": 0.1,
    })
    for key in ["t", "speed_rpm", "torque", "current", "power", "slip"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC175"
    assert info["fidelity"] == "F2a"


def test_no_load_near_synchronous_speed(model):
    """No-load speed should be near synchronous speed (1500 rpm for 2 pole pairs, 50 Hz)."""
    r = model.predict({
        "v_supply_rms": 400.0, "frequency_hz": 50.0, "T_load_Nm": 0.0,
        "dt": 5e-4, "duration_s": 3.0,
    })
    sync_speed = 50.0 * 60.0 / 2.0  # = 1500 rpm
    final_speed = r["speed_rpm"][-1]
    # Should be within 2% of synchronous speed
    assert abs(final_speed - sync_speed) / sync_speed < 0.02, \
        f"No-load speed {final_speed:.1f}, expected ~{sync_speed:.0f} rpm"


def test_loaded_speed_below_synchronous(model):
    """Loaded motor runs below synchronous speed (positive slip)."""
    r = model.predict({
        "v_supply_rms": 400.0, "frequency_hz": 50.0, "T_load_Nm": 30.0,
        "dt": 5e-4, "duration_s": 3.0,
    })
    sync_speed = 1500.0
    assert r["speed_rpm"][-1] < sync_speed
    assert r["slip"][-1] > 0


def test_torque_matches_load_at_steady_state(model):
    """At steady state, electromagnetic torque ~ T_load + B*omega_r."""
    r = model.predict({
        "v_supply_rms": 400.0, "frequency_hz": 50.0, "T_load_Nm": 30.0,
        "dt": 5e-4, "duration_s": 3.0,
    })
    T_e_final = r["torque"][-1]
    omega_r_final = r["omega_r"][-1]
    B = 0.001
    expected = 30.0 + B * omega_r_final
    assert abs(T_e_final - expected) / expected < 0.05, \
        f"T_e={T_e_final:.2f}, expected={expected:.2f}"


def test_load_step_settles(model):
    """Motor settles to new speed after load step."""
    def t_load(t):
        return 10.0 if t < 1.0 else 40.0
    r = model.predict({
        "v_supply_rms": 400.0, "frequency_hz": 50.0, "T_load_Nm": t_load,
        "dt": 5e-4, "duration_s": 4.0,
    })
    # Speed at end should be lower than at t=0.9s
    idx_before = int(0.9 / 5e-4)
    speed_before = r["speed_rpm"][idx_before]
    speed_after = r["speed_rpm"][-1]
    assert speed_after < speed_before


def test_slip_positive_motoring(model):
    """Slip should be positive when motoring."""
    r = model.predict({
        "v_supply_rms": 400.0, "frequency_hz": 50.0, "T_load_Nm": 20.0,
        "dt": 5e-4, "duration_s": 3.0,
    })
    assert r["slip"][-1] > 0
    assert r["slip"][-1] < 0.1  # Reasonable slip for normal operation


def test_power_positive_motoring(model):
    """Mechanical output power should be positive when motoring."""
    r = model.predict({
        "v_supply_rms": 400.0, "frequency_hz": 50.0, "T_load_Nm": 20.0,
        "dt": 5e-4, "duration_s": 3.0,
    })
    assert r["power"][-1] > 0


def test_current_reasonable(model):
    """Stator current should be within reasonable range."""
    r = model.predict({
        "v_supply_rms": 400.0, "frequency_hz": 50.0, "T_load_Nm": 30.0,
        "dt": 5e-4, "duration_s": 3.0,
    })
    # Rated current for 10kW at 400V ~ 18A
    assert r["current"][-1] < 100.0  # Not excessively large
    assert r["current"][-1] > 0.1    # Not zero


def test_benchmark(model):
    start = time.perf_counter()
    model.predict({
        "v_supply_rms": 400.0, "frequency_hz": 50.0, "T_load_Nm": 30.0,
        "dt": 5e-4, "duration_s": 1.0,
    })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1s sim in {elapsed*1000:.1f} ms")
    assert elapsed < 10.0
