"""EC082 — Ice Thermal Storage — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"soc": 0.5})
    for k in ["dSOC_dt", "energy_stored_kwh", "max_charge_kw", "max_discharge_kw",
              "q_charge_effective_kw", "q_discharge_effective_kw", "heat_loss_w"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC082"
    assert info["fidelity"] == "F1a"


def test_energy_endpoints(model):
    """E=0 at SOC=0; E=capacity at SOC=1."""
    r0 = model.predict({"soc": 0.0})
    r1 = model.predict({"soc": 1.0})
    assert float(r0["energy_stored_kwh"]) == 0.0
    assert abs(float(r1["energy_stored_kwh"]) - model._model.cap_kwh) < 1e-9


def test_max_charge_zero_at_full(model):
    """Cannot charge a full ice tank."""
    r = model.predict({"soc": 1.0})
    assert float(r["max_charge_kw"]) == 0.0


def test_max_discharge_zero_at_empty(model):
    """Cannot discharge an empty ice tank."""
    r = model.predict({"soc": 0.0})
    assert float(r["max_discharge_kw"]) == 0.0


def test_max_charge_at_empty(model):
    """Maximum charge when SOC=0."""
    r = model.predict({"soc": 0.0})
    assert abs(float(r["max_charge_kw"]) - model._model.Q_chg_max / 1000.0) < 1e-9


def test_dSOC_increases_with_charging(model):
    r0 = model.predict({"soc": 0.5, "q_charge": 0.0,    "q_discharge": 0.0, "t_ambient": 0.0})
    r1 = model.predict({"soc": 0.5, "q_charge": 80000.0,"q_discharge": 0.0, "t_ambient": 0.0})
    assert float(r1["dSOC_dt"]) > float(r0["dSOC_dt"])


def test_heat_loss_positive_when_warm(model):
    r = model.predict({"soc": 0.5, "t_ambient": 25.0})
    assert float(r["heat_loss_w"]) > 0


def test_self_discharge(model):
    """With no charging/discharging and warm ambient, ice melts (dSOC < 0)."""
    r = model.predict({"soc": 0.8, "q_charge": 0.0, "q_discharge": 0.0, "t_ambient": 25.0})
    assert float(r["dSOC_dt"]) < 0


def test_charge_request_clipped(model):
    """If user requests more than max_charge, effective charge is clipped."""
    r = model.predict({"soc": 0.95, "q_charge": 1e9, "q_discharge": 0.0, "t_ambient": 0.0})
    assert float(r["q_charge_effective_kw"]) <= float(r["max_charge_kw"]) + 1e-6


def test_discharge_request_clipped(model):
    r = model.predict({"soc": 0.05, "q_charge": 0.0, "q_discharge": 1e9, "t_ambient": 0.0})
    assert float(r["q_discharge_effective_kw"]) <= float(r["max_discharge_kw"]) + 1e-6


def test_full_charge_simulation(model):
    """Time-step a full charging cycle from 0 -> 1 SOC, check it converges."""
    soc = 0.0
    dt = 60.0  # s
    for _ in range(int(40 * 3600 / dt)):  # 40 hours max
        r = model.predict({"soc": soc, "q_charge": 1e9, "q_discharge": 0.0, "t_ambient": 0.0})
        soc = min(1.0, soc + float(r["dSOC_dt"]) * dt)
        if soc >= 0.999:
            break
    assert soc >= 0.95


def test_benchmark(model):
    n = 1000
    soc = np.random.uniform(0, 1, n)
    qc  = np.random.uniform(0, 100000, n)
    qd  = np.random.uniform(0, 100000, n)
    Ta  = np.random.uniform(15, 30, n)
    start = time.perf_counter()
    model.predict({"soc": soc, "q_charge": qc, "q_discharge": qd, "t_ambient": Ta})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
