"""EC123 — Compressed Air Energy Storage (CAES) — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys_charge(model):
    r = model.predict({"mode": "charge", "m_dot_air": 100.0, "soc": 0.5})
    for k in ["power_kw", "fuel_power_kw", "fuel_mass_flow_kgs",
              "cavern_pressure_pa", "cavern_air_mass_kg",
              "energy_capacity_kwh", "round_trip_eta", "electric_rt_ratio"]:
        assert k in r


def test_predict_keys_discharge(model):
    r = model.predict({"mode": "discharge", "m_dot_air": 400.0, "soc": 0.5})
    for k in ["power_kw", "fuel_power_kw", "fuel_mass_flow_kgs",
              "cavern_pressure_pa", "cavern_air_mass_kg",
              "energy_capacity_kwh", "round_trip_eta", "electric_rt_ratio"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC123"
    assert info["fidelity"] == "F1a"


def test_round_trip_eta_range(model):
    """Diabatic CAES RTE should be ~0.42-0.55."""
    rte = model.predict({"mode": "idle"})["round_trip_eta"]
    assert 0.40 <= rte <= 0.60, f"RTE = {rte:.3f} outside expected 0.40-0.60"


def test_round_trip_eta_less_than_one(model):
    rte = model.predict({"mode": "idle"})["round_trip_eta"]
    assert rte < 1.0


def test_charge_power_positive(model):
    """Charge mode → positive electrical power into compressor."""
    p = float(model.predict({"mode": "charge", "m_dot_air": 100.0})["power_kw"])
    assert p > 0.0


def test_discharge_power_negative(model):
    """Discharge mode → negative electrical power (delivered to grid)."""
    p = float(model.predict({"mode": "discharge", "m_dot_air": 400.0})["power_kw"])
    assert p < 0.0


def test_fuel_only_in_discharge(model):
    """Fuel input only during discharge."""
    r_c = model.predict({"mode": "charge", "m_dot_air": 100.0})
    r_d = model.predict({"mode": "discharge", "m_dot_air": 400.0})
    assert float(r_c["fuel_power_kw"]) == 0.0
    assert float(r_d["fuel_power_kw"]) > 0.0


def test_soc_clamped(model):
    """SOC outside [0,1] should clamp."""
    r_lo = model.predict({"mode": "idle", "soc": -0.5})
    r_hi = model.predict({"mode": "idle", "soc": 1.5})
    p_lo = float(r_lo["cavern_pressure_pa"])
    p_hi = float(r_hi["cavern_pressure_pa"])
    assert abs(p_lo - model._model.p_min) < 1.0
    assert abs(p_hi - model._model.p_max) < 1.0


def test_cavern_pressure_monotonic_with_soc(model):
    soc = np.linspace(0, 1, 50)
    r = model.predict({"mode": "idle", "soc": soc})
    p = r["cavern_pressure_pa"]
    assert np.all(np.diff(p) > 0)


def test_energy_capacity_positive(model):
    cap = model.predict({"mode": "idle"})["energy_capacity_kwh"]
    assert cap > 0.0


def test_charge_power_linear_in_mdot(model):
    p1 = float(model.predict({"mode": "charge", "m_dot_air": 50.0})["power_kw"])
    p2 = float(model.predict({"mode": "charge", "m_dot_air": 100.0})["power_kw"])
    assert abs(p2 / p1 - 2.0) < 1e-6


def test_soc_update_charge_increases(model):
    s_new = model._model.soc_update(0.3, 50000.0, 1.0, "charge")
    assert s_new > 0.3


def test_soc_update_discharge_decreases(model):
    s_new = model._model.soc_update(0.7, 50000.0, 1.0, "discharge")
    assert s_new < 0.7


def test_soc_update_clamps_to_one(model):
    """Charging beyond capacity must clamp at SOC=1."""
    s_new = model._model.soc_update(0.95, 1.0e7, 24.0, "charge")
    assert s_new <= 1.0 + 1e-9


def test_soc_update_clamps_to_zero(model):
    """Discharging beyond empty must clamp at SOC=0."""
    s_new = model._model.soc_update(0.05, 1.0e7, 24.0, "discharge")
    assert s_new >= 0.0 - 1e-9


def test_soc_update_idle_unchanged(model):
    s_new = model._model.soc_update(0.5, 50000.0, 1.0, "idle")
    assert s_new == 0.5


def test_invalid_mode(model):
    with pytest.raises(ValueError):
        model.predict({"mode": "boost", "m_dot_air": 50.0})


def test_vectorized_input(model):
    m = np.linspace(10, 200, 25)
    r = model.predict({"mode": "discharge", "m_dot_air": m})
    assert len(r["power_kw"]) == 25


def test_energy_conservation_round_trip(model):
    """E_out_elec / (E_in_elec + E_in_fuel) must equal RTE definition (all in kWh/kg)."""
    m = model._model
    E_out = m.w_exp * m.eta_exp * m.eta_gen / 3600.0          # kWh/kg
    E_in_elec = m.w_comp / (m.eta_comp * m.eta_motor) / 3600.0  # kWh/kg
    E_in_fuel = E_out * m.heat_rate / 3600.0                  # kWh/kg
    rte = E_out / (E_in_elec + E_in_fuel)
    assert abs(rte - m.round_trip_efficiency()) < 1e-9


def test_benchmark(model):
    m = np.random.uniform(10, 400, 1000)
    s = np.random.uniform(0, 1, 1000)
    start = time.perf_counter()
    model.predict({"mode": "discharge", "m_dot_air": m, "soc": s})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
