"""EC124 — Liquid Air Energy Storage (LAES) — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"mode": "charge", "m_dot_liquid_kgs": 30.0})
    for k in ["power_kw", "liquid_mass_kg", "tank_volume_m3",
              "soc_after_standby", "energy_capacity_kwh", "round_trip_eta"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC124"
    assert info["fidelity"] == "F1a"


def test_round_trip_eta_range(model):
    """LAES RTE should be in 0.50-0.70."""
    rte = model.predict({"mode": "idle"})["round_trip_eta"]
    assert 0.50 <= rte <= 0.70, f"RTE = {rte:.3f} outside expected 0.50-0.70"


def test_round_trip_eta_less_than_one(model):
    rte = model.predict({"mode": "idle"})["round_trip_eta"]
    assert rte < 1.0


def test_charge_power_positive(model):
    p = float(model.predict({"mode": "charge", "m_dot_liquid_kgs": 30.0})["power_kw"])
    assert p > 0.0


def test_discharge_power_negative(model):
    p = float(model.predict({"mode": "discharge", "m_dot_liquid_kgs": 60.0})["power_kw"])
    assert p < 0.0


def test_charge_power_linear_in_mdot(model):
    p1 = float(model.predict({"mode": "charge", "m_dot_liquid_kgs": 20.0})["power_kw"])
    p2 = float(model.predict({"mode": "charge", "m_dot_liquid_kgs": 60.0})["power_kw"])
    assert abs(p2 / p1 - 3.0) < 1e-6


def test_soc_clamped(model):
    r_lo = model.predict({"mode": "idle", "soc": -0.5})
    r_hi = model.predict({"mode": "idle", "soc": 1.5})
    assert float(r_lo["liquid_mass_kg"]) == 0.0
    assert abs(float(r_hi["liquid_mass_kg"]) - model._model.m_tank_max) < 1e-6


def test_liquid_mass_proportional_to_soc(model):
    soc = np.linspace(0, 1, 50)
    r = model.predict({"mode": "idle", "soc": soc})
    assert np.all(np.diff(r["liquid_mass_kg"]) >= -1e-9)


def test_boil_off_decreases_soc(model):
    r0 = model.predict({"mode": "idle", "soc": 1.0, "time_hours": 0.0})
    r1 = model.predict({"mode": "idle", "soc": 1.0, "time_hours": 24.0})
    assert float(r1["soc_after_standby"]) < float(r0["soc_after_standby"])


def test_boil_off_rate_about_half_percent_per_day(model):
    """At ~0.5%/day, after 24 h SOC should drop ~0.5% from 1.0."""
    r = model.predict({"mode": "idle", "soc": 1.0, "time_hours": 24.0})
    soc_after = float(r["soc_after_standby"])
    drop = 1.0 - soc_after
    assert 0.003 < drop < 0.008, f"24h boil-off drop = {drop:.4f}"


def test_energy_capacity_positive(model):
    cap = model.predict({"mode": "idle"})["energy_capacity_kwh"]
    assert cap > 0.0


def test_soc_update_charge_increases(model):
    s_new = model._model.soc_update(0.3, 50000.0, 1.0, "charge")
    assert s_new > 0.3


def test_soc_update_discharge_decreases(model):
    s_new = model._model.soc_update(0.7, 50000.0, 1.0, "discharge")
    assert s_new < 0.7


def test_soc_update_clamps_to_one(model):
    s_new = model._model.soc_update(0.95, 1.0e7, 24.0, "charge")
    assert s_new <= 1.0 + 1e-9


def test_soc_update_clamps_to_zero(model):
    s_new = model._model.soc_update(0.05, 1.0e7, 24.0, "discharge")
    assert s_new >= 0.0 - 1e-9


def test_soc_update_idle_decays(model):
    s_new = model._model.soc_update(0.8, 0.0, 48.0, "idle")
    assert s_new < 0.8


def test_invalid_mode(model):
    with pytest.raises(ValueError):
        model.predict({"mode": "drain", "m_dot_liquid_kgs": 50.0})


def test_vectorized_input(model):
    m = np.linspace(5, 100, 25)
    r = model.predict({"mode": "discharge", "m_dot_liquid_kgs": m})
    assert len(r["power_kw"]) == 25


def test_energy_conservation(model):
    """Charging full tank then discharging full tank: E_out / E_in = RTE."""
    m = model._model
    E_in = m.m_tank_max * m.w_liq / m.eta_liq
    E_out = m.m_tank_max * m.w_disch * m.eta_pump * m.eta_exp * m.eta_gen
    assert abs(E_out / E_in - m.round_trip_efficiency()) < 1e-9


def test_benchmark(model):
    m = np.random.uniform(5, 100, 1000)
    s = np.random.uniform(0, 1, 1000)
    start = time.perf_counter()
    model.predict({"mode": "charge", "m_dot_liquid_kgs": m, "soc": s})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
