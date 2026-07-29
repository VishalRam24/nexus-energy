"""EC041 — EDLC Supercapacitor — F1a RC — Test Suite"""

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


def test_predict_keys(model):
    r = model.predict({"v_cap": 2.5, "current": 50.0})
    for k in ["voltage", "soc", "charge", "stored_energy", "power", "dvcap_dt"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC041"
    assert info["fidelity"] == "F1a"


def test_charge_linear_in_voltage(model):
    """Q = C * V_cap is linear."""
    v_arr = np.linspace(0.0, 2.7, 20)
    r = model.predict({"v_cap": v_arr, "current": 0.0})
    C = model._model.C
    np.testing.assert_allclose(r["charge"], C * v_arr, rtol=1e-12)


def test_energy_quadratic_in_voltage(model):
    """E = 0.5*C*V_cap^2."""
    v_arr = np.linspace(0.0, 2.7, 20)
    r = model.predict({"v_cap": v_arr, "current": 0.0})
    C = model._model.C
    np.testing.assert_allclose(r["stored_energy"], 0.5 * C * v_arr ** 2, rtol=1e-12)


def test_voltage_drop_with_discharge(model):
    """V_term = V_cap - I*ESR drops with positive current."""
    r0 = model.predict({"v_cap": 2.5, "current":   0.0})
    r1 = model.predict({"v_cap": 2.5, "current": 100.0})
    assert float(r1["voltage"]) < float(r0["voltage"])


def test_voltage_rise_during_charge(model):
    """During charge (I<0), terminal voltage > V_cap (clipped at V_max)."""
    ESR = model._model.ESR
    v_cap = 2.0
    I = -100.0
    r = model.predict({"v_cap": v_cap, "current": I})
    expected = min(2.7, v_cap - I * ESR)
    assert float(r["voltage"]) == pytest.approx(expected, abs=1e-9)


def test_voltage_clipped_to_vmax(model):
    """V_term cannot exceed v_max."""
    r = model.predict({"v_cap": 2.7, "current": -300.0})
    assert float(r["voltage"]) <= 2.7 + 1e-12


def test_voltage_clipped_to_vmin(model):
    """V_term cannot drop below 0."""
    r = model.predict({"v_cap": 0.05, "current": 300.0})
    assert float(r["voltage"]) >= 0.0 - 1e-12


def test_soc_bounds(model):
    """SOC in [0, 1]."""
    v_arr = np.array([-0.5, 0.0, 1.35, 2.7, 3.0])
    r = model.predict({"v_cap": v_arr, "current": 0.0})
    assert np.all(r["soc"] >= 0.0)
    assert np.all(r["soc"] <= 1.0)


def test_dvcap_dt_sign(model):
    """Discharging (I>0) decreases V_cap, charging (I<0) increases it."""
    r_dis = model.predict({"v_cap": 2.0, "current":  100.0})
    r_chg = model.predict({"v_cap": 2.0, "current": -100.0})
    assert float(r_dis["dvcap_dt"]) < 0
    assert float(r_chg["dvcap_dt"]) > 0


def test_self_discharge_drains_cap(model):
    """At I=0, dV/dt < 0 due to leakage R_leak."""
    r = model.predict({"v_cap": 2.5, "current": 0.0})
    assert float(r["dvcap_dt"]) < 0


def test_zero_voltage_no_self_discharge(model):
    """At V_cap=0, leakage current is zero so dV/dt = 0 when I=0."""
    r = model.predict({"v_cap": 0.0, "current": 0.0})
    assert float(r["dvcap_dt"]) == pytest.approx(0.0, abs=1e-15)


def test_energy_conservation_constant_current(model):
    """
    Energy delivered to load over a short discharge equals decrease in stored
    energy minus losses in ESR (and tiny leakage). Check approximate balance.
    """
    v0 = 2.5
    I = 50.0
    dt = 0.01
    r = model.predict({"v_cap": v0, "current": I})
    dvdt = float(r["dvcap_dt"])
    v1 = v0 + dvdt * dt
    e0 = 0.5 * model._model.C * v0 ** 2
    e1 = 0.5 * model._model.C * v1 ** 2
    delivered = float(r["voltage"]) * I * dt        # external load
    esr_loss  = (I ** 2) * model._model.ESR * dt    # I^2 R loss
    leak_loss = (v0 ** 2 / model._model.R_leak) * dt  # leakage power
    balance = (e0 - e1) - (delivered + esr_loss + leak_loss)
    assert abs(balance) < 1e-3 * e0


def test_array_inputs(model):
    v = np.array([0.5, 1.5, 2.5])
    I = np.array([10.0, 20.0, 30.0])
    r = model.predict({"v_cap": v, "current": I})
    assert r["voltage"].shape == (3,)


def test_benchmark(model):
    v = np.random.uniform(0.0, 2.7, 1000)
    I = np.random.uniform(-200.0, 200.0, 1000)
    start = time.perf_counter()
    model.predict({"v_cap": v, "current": I})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
