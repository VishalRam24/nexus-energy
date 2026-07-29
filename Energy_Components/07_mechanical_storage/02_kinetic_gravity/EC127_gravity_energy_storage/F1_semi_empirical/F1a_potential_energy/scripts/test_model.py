"""EC127 — Gravity Energy Storage — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"mode": "charge", "velocity_mps": 0.02, "soc": 0.5})
    for k in ["power_kw", "height_m", "potential_energy_kwh",
              "energy_capacity_kwh", "round_trip_eta",
              "charge_eta", "discharge_eta"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC127"
    assert info["fidelity"] == "F1a"


def test_round_trip_eta_range(model):
    """RTE should be in 0.80-0.90."""
    rte = model.predict({"mode": "idle"})["round_trip_eta"]
    assert 0.80 <= rte <= 0.90, f"RTE = {rte:.3f} outside expected 0.80-0.90"


def test_round_trip_eta_less_than_one(model):
    rte = model.predict({"mode": "idle"})["round_trip_eta"]
    assert rte < 1.0


def test_charge_power_positive(model):
    p = float(model.predict({"mode": "charge", "velocity_mps": 0.01})["power_kw"])
    assert p > 0.0


def test_discharge_power_negative(model):
    p = float(model.predict({"mode": "discharge", "velocity_mps": 0.01})["power_kw"])
    assert p < 0.0


def test_charge_power_clamped_to_rated(model):
    """Excessive lift velocity must clamp at P_rated."""
    p = float(model.predict({"mode": "charge", "velocity_mps": 5.0})["power_kw"])
    assert abs(p - model._model.P_rated) < 1e-6


def test_discharge_power_clamped_to_rated(model):
    p = float(model.predict({"mode": "discharge", "velocity_mps": 5.0})["power_kw"])
    assert abs(-p - model._model.P_rated) < 1e-6


def test_soc_clamped(model):
    r_lo = model.predict({"mode": "idle", "soc": -0.5})
    r_hi = model.predict({"mode": "idle", "soc": 1.5})
    assert float(r_lo["height_m"]) == model._model.h_min
    assert abs(float(r_hi["height_m"]) - model._model.h_max) < 1e-9


def test_potential_energy_linear_in_height(model):
    """E = m*g*h: doubling SOC (h) doubles E."""
    e1 = float(model.predict({"mode": "idle", "soc": 0.25})["potential_energy_kwh"])
    e2 = float(model.predict({"mode": "idle", "soc": 0.50})["potential_energy_kwh"])
    assert abs(e2 / e1 - 2.0) < 1e-9


def test_potential_energy_monotonic(model):
    soc = np.linspace(0, 1, 50)
    r = model.predict({"mode": "idle", "soc": soc})
    assert np.all(np.diff(r["potential_energy_kwh"]) > 0)


def test_energy_capacity_matches_mgh(model):
    m = model._model
    expected = m.m * m.g * (m.h_max - m.h_min) / 3.6e6
    cap = model.predict({"mode": "idle"})["energy_capacity_kwh"]
    assert abs(cap - expected) < 1e-9


def test_soc_update_charge_increases(model):
    s_new = model._model.soc_update(0.3, 5000.0, 0.5, "charge")
    assert s_new > 0.3


def test_soc_update_discharge_decreases(model):
    s_new = model._model.soc_update(0.7, 5000.0, 0.5, "discharge")
    assert s_new < 0.7


def test_soc_update_clamps_to_one(model):
    s_new = model._model.soc_update(0.95, 5000.0, 24.0, "charge")
    assert s_new <= 1.0 + 1e-9


def test_soc_update_clamps_to_zero(model):
    s_new = model._model.soc_update(0.05, 5000.0, 24.0, "discharge")
    assert s_new >= 0.0 - 1e-9


def test_soc_update_idle_unchanged(model):
    s_new = model._model.soc_update(0.5, 1000.0, 1.0, "idle")
    assert s_new == 0.5


def test_round_trip_energy_conservation(model):
    """Charging E_in then discharging same mass yields E_in * RTE."""
    m = model._model
    s0 = 0.0
    # Charge to SOC=1
    s_after_charge = 1.0
    E_in_kwh = m.energy_capacity_kwh() / m.charge_efficiency()
    # Discharge fully back
    E_out_kwh = m.energy_capacity_kwh() * m.discharge_efficiency()
    rte = E_out_kwh / E_in_kwh
    assert abs(rte - m.round_trip_efficiency()) < 1e-9


def test_invalid_mode(model):
    with pytest.raises(ValueError):
        model.predict({"mode": "lift", "velocity_mps": 0.01})


def test_vectorized_input(model):
    v = np.linspace(0.001, 0.05, 25)
    r = model.predict({"mode": "charge", "velocity_mps": v})
    assert len(r["power_kw"]) == 25


def test_benchmark(model):
    v = np.random.uniform(0.001, 0.1, 1000)
    s = np.random.uniform(0, 1, 1000)
    start = time.perf_counter()
    model.predict({"mode": "charge", "velocity_mps": v, "soc": s})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
