"""
EC028 -- Lead-Acid Battery -- F2a ECM 1-RC -- Test Suite
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


def test_predict_returns_dict(model):
    result = model.predict({"current": [10.0] * 10, "dt": 1.0})
    assert isinstance(result, dict)
    for key in ["voltage", "soc", "power", "v_rc", "time"]:
        assert key in result
    assert len(result["voltage"]) == 10


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC028"
    assert info["fidelity"] == "F2a"


def test_v_rc_decays_exponentially_on_rest(raw_model):
    raw_model.reset(0.8)
    dt = 1.0
    for _ in range(60):
        raw_model.step(20.0, dt)

    v_rc_after_load = raw_model.v_rc
    assert abs(v_rc_after_load) > 0.001

    tau = raw_model.tau1(raw_model.soc)
    n_rest = int(5 * tau)
    v_rc_trace = []
    for _ in range(n_rest):
        raw_model.step(0.0, dt)
        v_rc_trace.append(raw_model.v_rc)

    assert abs(v_rc_trace[-1]) < abs(v_rc_after_load) * 0.05

    idx_tau = min(int(tau), len(v_rc_trace) - 1)
    ratio = v_rc_trace[idx_tau] / v_rc_after_load if v_rc_after_load != 0 else 0
    assert 0.2 < ratio < 0.55


def test_soc_decreases_during_discharge(raw_model):
    raw_model.reset(0.8)
    initial_soc = raw_model.soc
    for _ in range(100):
        raw_model.step(20.0, 1.0)
    assert raw_model.soc < initial_soc


def test_soc_increases_during_charge(raw_model):
    raw_model.reset(0.5)
    initial_soc = raw_model.soc
    for _ in range(100):
        raw_model.step(-20.0, 1.0)
    assert raw_model.soc > initial_soc


def test_voltage_within_bounds(model):
    Q = model.params["cell"]["capacity"]["value"]
    # Use C/5 rate for lead-acid
    I = Q / 5.0
    result = model.predict({"current": [I] * 20000, "dt": 1.0, "soc_init": 1.0})
    v_min = model.params["cell"]["voltage_min"]["value"]
    v_max = model.params["cell"]["voltage_max"]["value"]
    assert np.all(result["voltage"] >= v_min - 1e-6)
    assert np.all(result["voltage"] <= v_max + 1e-6)


def test_rc_time_constant(raw_model):
    tau = raw_model.tau1(0.5)
    assert 10.0 < tau < 500.0, f"Lead-acid tau={tau:.1f}s"


def test_full_discharge_capacity(raw_model):
    raw_model.reset(1.0)
    dt = 1.0
    # Use C/10 rate for lead-acid
    I = raw_model.Q_nom / 10.0
    total_ah = 0.0
    max_steps = int(50000)

    for _ in range(max_steps):
        v = raw_model.step(I, dt)
        total_ah += I * dt / 3600.0
        if v <= raw_model.v_min + 0.01 or raw_model.soc <= 0.001:
            break

    ratio = total_ah / raw_model.Q_nom
    assert 0.70 < ratio < 1.15, \
        f"Delivered {total_ah:.2f} Ah vs {raw_model.Q_nom:.2f} Ah nominal ({ratio:.1%})"


def test_step_response_immediate_r0_drop(raw_model):
    raw_model.reset(0.8)
    I = 20.0
    ocv_before = raw_model.ocv(raw_model.soc)
    v1 = raw_model.step(I, 0.001)
    r0 = raw_model.R0(0.8)
    expected_drop = I * r0
    actual_drop = float(ocv_before) - v1
    assert abs(actual_drop - expected_drop) < expected_drop * 0.5


def test_step_response_gradual_rc(raw_model):
    raw_model.reset(0.8)
    I = 20.0
    v1 = raw_model.step(I, 1.0)
    v2 = raw_model.step(I, 1.0)
    assert v2 < v1


def test_reset(raw_model):
    raw_model.step(20.0, 1.0)
    raw_model.reset(0.5)
    assert raw_model.soc == 0.5
    assert raw_model.v_rc == 0.0


def test_benchmark_simulation(model):
    Q = model.params["cell"]["capacity"]["value"]
    start = time.perf_counter()
    model.predict({"current": [Q / 10.0] * 3600, "dt": 1.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 3600-step simulation in {elapsed * 1000:.1f} ms")
    assert elapsed < 5.0
