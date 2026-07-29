"""EC125 — Adiabatic CAES (A-CAES) — F1a — Test Suite

Physics rules enforced:
  - eta_RT (A-CAES) in [0.62, 0.78] — literature range 65-72%; bounded below by diabatic
  - eta_RT (A-CAES) > eta_RT (diabatic) = ~0.42-0.55 for same pressure ratio
    (key physical requirement: TES recovery improves RTE)
  - eta_RT < 1.0 (thermodynamic limit)
  - No fuel input at any time (distinguishes A-CAES from EC123)
  - Charge power positive, discharge power negative (sign convention)
  - SOC monotonically increases with cavern pressure
  - SOC clamped [0, 1]
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

# Diabatic CAES RTE upper bound for comparison (EC123 Huntorf-class)
DIABATIC_RTE_UPPER = 0.55


@pytest.fixture
def model():
    return ComponentModel()


# ---------- interface tests ----------

def test_predict_keys_charge(model):
    r = model.predict({"mode": "charge", "m_dot_air": 100.0, "soc": 0.5})
    for k in ["power_kw", "fuel_power_kw", "fuel_mass_flow_kgs", "tes_heat_kw",
              "cavern_pressure_pa", "cavern_air_mass_kg",
              "energy_capacity_kwh", "round_trip_eta", "electric_rt_ratio"]:
        assert k in r


def test_predict_keys_discharge(model):
    r = model.predict({"mode": "discharge", "m_dot_air": 400.0, "soc": 0.5})
    for k in ["power_kw", "fuel_power_kw", "fuel_mass_flow_kgs", "tes_heat_kw",
              "cavern_pressure_pa", "cavern_air_mass_kg",
              "energy_capacity_kwh", "round_trip_eta", "electric_rt_ratio"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC125"
    assert info["fidelity"] == "F1a"


# ---------- A-CAES vs diabatic physics ----------

def test_round_trip_eta_range(model):
    """A-CAES RTE must be in 62-78% range (literature: 65-72%)."""
    rte = model.predict({"mode": "idle"})["round_trip_eta"]
    assert 0.62 <= rte <= 0.78, f"A-CAES RTE = {rte:.3f} outside expected 0.62-0.78"


def test_acaes_rte_exceeds_diabatic(model):
    """
    A-CAES RTE must be greater than diabatic CAES upper bound (0.55).
    Physical requirement: TES recovery of compression heat improves round-trip efficiency.
    """
    rte = model.predict({"mode": "idle"})["round_trip_eta"]
    assert rte > DIABATIC_RTE_UPPER, (
        f"A-CAES RTE {rte:.3f} must exceed diabatic CAES upper bound "
        f"{DIABATIC_RTE_UPPER:.2f} (TES improves efficiency)"
    )


def test_round_trip_eta_less_than_one(model):
    """RTE must be < 1 (cannot create energy)."""
    rte = model.predict({"mode": "idle"})["round_trip_eta"]
    assert rte < 1.0, f"RTE {rte:.3f} >= 1 — physically impossible"


# ---------- no-fuel requirement ----------

def test_no_fuel_in_discharge(model):
    """A-CAES discharge must have ZERO fuel input — distinguishes it from diabatic EC123."""
    r = model.predict({"mode": "discharge", "m_dot_air": 400.0})
    assert float(r["fuel_power_kw"])      == 0.0, "A-CAES must have zero fuel power input"
    assert float(r["fuel_mass_flow_kgs"]) == 0.0, "A-CAES must have zero fuel mass flow"


def test_no_fuel_in_charge(model):
    """Charging phase also must have zero fuel."""
    r = model.predict({"mode": "charge", "m_dot_air": 100.0})
    assert float(r["fuel_power_kw"])      == 0.0
    assert float(r["fuel_mass_flow_kgs"]) == 0.0


# ---------- TES heat ----------

def test_tes_heat_positive_during_charge(model):
    """Heat must be stored in TES during charging."""
    r = model.predict({"mode": "charge", "m_dot_air": 100.0})
    assert float(r["tes_heat_kw"]) > 0.0, "TES heat stored must be positive during charge"


def test_tes_heat_zero_during_discharge(model):
    """TES heat output tracked separately; discharge power_kw already accounts for it."""
    r = model.predict({"mode": "discharge", "m_dot_air": 400.0})
    assert float(r["tes_heat_kw"]) == 0.0


# ---------- power sign convention ----------

def test_charge_power_positive(model):
    """Charge mode → positive electrical power into compressor."""
    p = float(model.predict({"mode": "charge", "m_dot_air": 100.0})["power_kw"])
    assert p > 0.0


def test_discharge_power_negative(model):
    """Discharge mode → negative electrical power (delivered to grid)."""
    p = float(model.predict({"mode": "discharge", "m_dot_air": 400.0})["power_kw"])
    assert p < 0.0


# ---------- cavern state ----------

def test_soc_clamped(model):
    """SOC outside [0,1] should clamp to boundaries."""
    r_lo = model.predict({"mode": "idle", "soc": -0.5})
    r_hi = model.predict({"mode": "idle", "soc": 1.5})
    p_lo = float(r_lo["cavern_pressure_pa"])
    p_hi = float(r_hi["cavern_pressure_pa"])
    assert abs(p_lo - model._model.p_min) < 1.0
    assert abs(p_hi - model._model.p_max) < 1.0


def test_cavern_pressure_monotonic_with_soc(model):
    """Cavern pressure must increase monotonically with SOC."""
    soc = np.linspace(0, 1, 50)
    r   = model.predict({"mode": "idle", "soc": soc})
    assert np.all(np.diff(r["cavern_pressure_pa"]) > 0)


def test_energy_capacity_positive(model):
    cap = model.predict({"mode": "idle"})["energy_capacity_kwh"]
    assert cap > 0.0


def test_charge_power_linear_in_mdot(model):
    """Charge power must be linear in mass flow."""
    p1 = float(model.predict({"mode": "charge", "m_dot_air": 50.0})["power_kw"])
    p2 = float(model.predict({"mode": "charge", "m_dot_air": 100.0})["power_kw"])
    assert abs(p2 / p1 - 2.0) < 1e-6


# ---------- SOC update ----------

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
    """RTE definition consistency: E_out / E_in must match round_trip_efficiency()."""
    mdl = model._model
    E_out     = mdl.w_exp * mdl.eta_exp * mdl.eta_gen / 3600.0
    E_in_elec = mdl.w_comp / (mdl.eta_comp * mdl.eta_motor) / 3600.0
    rte = E_out / E_in_elec
    assert abs(rte - mdl.round_trip_efficiency()) < 1e-9


# ---------- benchmark ----------

def test_benchmark(model):
    m_dot = np.random.uniform(10, 400, 1000)
    soc   = np.random.uniform(0, 1, 1000)
    start = time.perf_counter()
    model.predict({"mode": "discharge", "m_dot_air": m_dot, "soc": soc})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
