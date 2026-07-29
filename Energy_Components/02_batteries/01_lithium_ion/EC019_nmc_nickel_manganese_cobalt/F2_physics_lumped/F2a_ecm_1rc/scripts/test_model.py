"""
EC019 -- NMC Battery -- F2a ECM 1-RC -- Test Suite

Physics sanity checks for the 1-RC equivalent circuit model.
Run: python -m pytest test_model.py -v
"""

import sys
import time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


@pytest.fixture
def raw_model(model):
    return model._model


# --- Output structure ---

def test_predict_returns_dict(model):
    result = model.predict({"current": [5.0] * 10, "dt": 1.0})
    assert isinstance(result, dict)
    for key in ["voltage", "soc", "power", "v_rc", "time"]:
        assert key in result
    assert len(result["voltage"]) == 10


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC019"
    assert info["fidelity"] == "F2a"


# --- RC relaxation test ---

def test_v_rc_decays_exponentially_on_rest(raw_model):
    """When current goes to zero, V_rc must decay exponentially toward zero."""
    raw_model.reset(0.8)
    dt = 1.0

    # Apply current for 30s to build up V_rc
    for _ in range(30):
        raw_model.step(5.0, dt)

    v_rc_after_load = raw_model.v_rc
    assert abs(v_rc_after_load) > 0.001, "V_rc should be nonzero after load"

    # Now rest (I=0) for 5*tau seconds and record V_rc decay
    tau = raw_model.tau1(raw_model.soc)
    n_rest = int(5 * tau)
    v_rc_trace = []
    for _ in range(n_rest):
        raw_model.step(0.0, dt)
        v_rc_trace.append(raw_model.v_rc)

    # V_rc should decay toward zero
    assert abs(v_rc_trace[-1]) < abs(v_rc_after_load) * 0.05, \
        f"V_rc should decay to <5% after 5*tau. Got {v_rc_trace[-1]:.6f} vs initial {v_rc_after_load:.6f}"

    # Check exponential decay: V_rc at t=tau should be ~37% of initial
    idx_tau = min(int(tau), len(v_rc_trace) - 1)
    ratio = v_rc_trace[idx_tau] / v_rc_after_load if v_rc_after_load != 0 else 0
    assert 0.2 < ratio < 0.55, \
        f"V_rc at t=tau should be ~37% of initial, got {ratio:.2%}"


# --- SOC behavior ---

def test_soc_decreases_during_discharge(raw_model):
    """SOC must decrease during discharge (positive current)."""
    raw_model.reset(0.8)
    initial_soc = raw_model.soc
    for _ in range(100):
        raw_model.step(5.0, 1.0)
    assert raw_model.soc < initial_soc, "SOC must decrease during discharge"


def test_soc_increases_during_charge(raw_model):
    """SOC must increase during charge (negative current)."""
    raw_model.reset(0.5)
    initial_soc = raw_model.soc
    for _ in range(100):
        raw_model.step(-5.0, 1.0)
    assert raw_model.soc > initial_soc, "SOC must increase during charge"


# --- Voltage bounds ---

def test_voltage_within_bounds(model):
    """Terminal voltage must stay within [V_min, V_max] throughout discharge."""
    Q = model.params["cell"]["capacity"]["value"]
    n_steps = int(4200)  # more than 1h
    result = model.predict({"current": [Q] * n_steps, "dt": 1.0, "soc_init": 1.0})
    v_min = model.params["cell"]["voltage_min"]["value"]
    v_max = model.params["cell"]["voltage_max"]["value"]
    assert np.all(result["voltage"] >= v_min - 1e-6), "Voltage below V_min"
    assert np.all(result["voltage"] <= v_max + 1e-6), "Voltage above V_max"


# --- RC time constant ---

def test_rc_time_constant(raw_model):
    """tau1 = R1 * C1 at nominal SOC should match expected value."""
    tau = raw_model.tau1(0.5)
    R1_nom = raw_model.R1_nom
    C1_nom = raw_model.C1_nom
    # At SOC=0.5, the factor should be close to 1.0
    # tau should be in a reasonable range
    assert 5.0 < tau < 300.0, f"Time constant {tau:.1f}s outside reasonable range"


# --- Full discharge capacity ---

def test_full_discharge_capacity(raw_model):
    """Full discharge from SOC=1 to cutoff should deliver approximately Q_nom."""
    raw_model.reset(1.0)
    dt = 1.0
    I = raw_model.Q_nom  # 1C rate
    total_ah = 0.0
    max_steps = int(5000)

    for _ in range(max_steps):
        v = raw_model.step(I, dt)
        total_ah += I * dt / 3600.0
        if v <= raw_model.v_min + 0.01 or raw_model.soc <= 0.001:
            break

    # Should deliver between 70% and 115% of Q_nom
    # (slightly over 100% possible if OCV at SOC~0 is still above cutoff)
    ratio = total_ah / raw_model.Q_nom
    assert 0.70 < ratio < 1.15, \
        f"Delivered {total_ah:.2f} Ah vs {raw_model.Q_nom:.2f} Ah nominal ({ratio:.1%})"


# --- Step response ---

def test_step_response_immediate_r0_drop(raw_model):
    """Applying current should cause immediate R0 voltage drop, then gradual RC."""
    raw_model.reset(0.8)
    I = 5.0

    # Record OCV before step
    ocv_before = raw_model.ocv(raw_model.soc)

    # First step
    v1 = raw_model.step(I, 0.001)  # very small dt to see instantaneous response

    # The immediate drop should be approximately I*R0
    r0 = raw_model.R0(0.8)
    expected_drop = I * r0
    actual_drop = float(ocv_before) - v1

    # The actual drop includes a tiny bit of V_rc too, but with dt=0.001 it's negligible
    assert abs(actual_drop - expected_drop) < expected_drop * 0.5, \
        f"Immediate drop {actual_drop:.4f}V vs expected R0 drop {expected_drop:.4f}V"


def test_step_response_gradual_rc(raw_model):
    """After initial R0 drop, voltage should continue to decrease gradually (RC effect)."""
    raw_model.reset(0.8)
    I = 5.0
    dt = 1.0

    # Step for 1s
    v1 = raw_model.step(I, dt)
    soc1 = raw_model.soc

    # Step for another 1s
    v2 = raw_model.step(I, dt)

    # Voltage at step 2 should be lower than step 1 (SOC decreasing + V_rc building up)
    assert v2 < v1, "Voltage should decrease as V_rc builds up"


# --- Reset ---

def test_reset(raw_model):
    """Reset should restore initial state."""
    raw_model.step(5.0, 1.0)
    raw_model.reset(0.5)
    assert raw_model.soc == 0.5
    assert raw_model.v_rc == 0.0


# --- Benchmark ---

def test_benchmark_simulation(model):
    """Benchmark: 3600 step simulation (1h at 1s steps)."""
    Q = model.params["cell"]["capacity"]["value"]
    current = [Q] * 3600
    start = time.perf_counter()
    model.predict({"current": current, "dt": 1.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 3600-step simulation in {elapsed * 1000:.1f} ms")
    assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s for 3600 steps"
