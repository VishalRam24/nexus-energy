"""EC125 — A-CAES — F1b Thermal — Test Suite

Physics rules enforced:
  - eta_RT (A-CAES) < 0.75 at ALL conditions (hard physical limit — Budt 2016)
  - eta_RT at design conditions in [0.60, 0.75]
  - eta_RT > diabatic CAES (0.55) at design conditions
  - No fuel input ever
  - TES heat stored during charge, zero during discharge
  - RTE increases as T_amb decreases (less compression work needed)
  - TES heat fraction decreases as TES cools
  - Expansion work reduces when TES is cold
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

DIABATIC_RTE_UPPER = 0.55


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys_charge(model):
    r = model.predict({"mode": "charge", "m_dot_air": 100.0, "soc": 0.5})
    for k in ["power_kw", "fuel_power_kw", "tes_heat_kw", "round_trip_eta",
              "cavern_pressure_pa", "cavern_air_mass_kg", "energy_capacity_kwh"]:
        assert k in r


def test_predict_keys_discharge(model):
    r = model.predict({"mode": "discharge", "m_dot_air": 400.0, "soc": 0.5})
    for k in ["power_kw", "fuel_power_kw", "round_trip_eta"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC125"
    assert info["fidelity"] == "F1b"


# --- RTE physical limits ---

def test_rte_design_in_range(model):
    """A-CAES design RTE must be in [0.60, 0.75]."""
    rte = float(model.predict({"mode": "idle"})["round_trip_eta"])
    assert 0.60 <= rte <= 0.75, f"A-CAES design RTE={rte:.4f} outside [0.60, 0.75]"


def test_rte_less_than_physical_limit(model):
    """A-CAES RTE must be < 0.75 at ALL temperatures (hard physical limit)."""
    for T in [253.15, 273.15, 288.15, 303.15, 313.15]:
        rte = float(model.predict({"mode": "idle", "T_amb_K": T})["round_trip_eta"])
        assert rte < 0.75, f"A-CAES RTE={rte:.4f} >= 0.75 at T_amb={T} K — violates Budt (2016)"


def test_rte_exceeds_diabatic(model):
    """A-CAES RTE must exceed diabatic CAES (0.55) at design conditions."""
    rte = float(model.predict({"mode": "idle"})["round_trip_eta"])
    assert rte > DIABATIC_RTE_UPPER, f"A-CAES RTE={rte:.4f} must exceed diabatic upper {DIABATIC_RTE_UPPER}"


def test_rte_less_than_one(model):
    """RTE must be < 1.0."""
    rte = float(model.predict({"mode": "idle"})["round_trip_eta"])
    assert rte < 1.0


# --- T_amb effect ---

def test_rte_increases_at_cold_ambient(model):
    """Colder ambient → less compression work → higher RTE."""
    rte_cold = float(model.predict({"mode": "idle", "T_amb_K": 263.15})["round_trip_eta"])
    rte_hot  = float(model.predict({"mode": "idle", "T_amb_K": 308.15})["round_trip_eta"])
    assert rte_cold > rte_hot, f"Cold RTE={rte_cold:.4f} should exceed hot RTE={rte_hot:.4f}"


def test_comp_work_increases_with_tamb(model):
    """Specific compression work must increase with ambient temperature."""
    w_cold = float(model.predict({"mode": "charge", "m_dot_air": 100.0, "T_amb_K": 263.15})
                   ["specific_comp_work_kj_kg"])
    w_hot  = float(model.predict({"mode": "charge", "m_dot_air": 100.0, "T_amb_K": 308.15})
                   ["specific_comp_work_kj_kg"])
    assert w_hot > w_cold


# --- TES temperature effect ---

def test_expansion_work_lower_with_cold_tes(model):
    """Cold TES → less heat returned → lower expansion work."""
    w_hot  = float(model.predict({"mode": "discharge", "m_dot_air": 100.0, "T_tes_K": 550.0})
                   ["expansion_work_eff_kj_kg"])
    w_cold = float(model.predict({"mode": "discharge", "m_dot_air": 100.0, "T_tes_K": 380.0})
                   ["expansion_work_eff_kj_kg"])
    assert w_hot > w_cold, f"Hot TES w_exp={w_hot:.1f} should exceed cold TES w_exp={w_cold:.1f}"


def test_rte_lower_with_cold_tes(model):
    """Cold TES reduces both expansion work and RTE."""
    rte_hot  = float(model.predict({"mode": "idle", "T_tes_K": 550.0})["round_trip_eta"])
    rte_cold = float(model.predict({"mode": "idle", "T_tes_K": 380.0})["round_trip_eta"])
    assert rte_hot > rte_cold


def test_tes_heat_fraction_at_design(model):
    """TES heat fraction must be 1.0 at design temperature."""
    f = model._model.tes_heat_available_fraction(model._model.T_tes_design)
    assert abs(float(f) - 1.0) < 1e-9


def test_tes_heat_fraction_decreases_with_cooling(model):
    """TES heat fraction must decrease as TES temperature falls below design."""
    m = model._model
    f_hot  = float(m.tes_heat_available_fraction(m.T_tes_design))
    f_cold = float(m.tes_heat_available_fraction(m.T_tes_ambient + 100.0))
    assert f_hot > f_cold


# --- No fuel ---

def test_no_fuel_any_mode(model):
    """A-CAES never consumes fuel."""
    for mode in ["charge", "discharge", "idle"]:
        r = model.predict({"mode": mode, "m_dot_air": 200.0, "soc": 0.5})
        assert float(r["fuel_power_kw"]) == 0.0
        assert float(r["fuel_mass_flow_kgs"]) == 0.0


# --- TES heat bookkeeping ---

def test_tes_heat_positive_during_charge(model):
    r = model.predict({"mode": "charge", "m_dot_air": 100.0})
    assert float(r["tes_heat_kw"]) > 0.0


def test_tes_heat_zero_during_discharge(model):
    r = model.predict({"mode": "discharge", "m_dot_air": 400.0})
    assert float(r["tes_heat_kw"]) == 0.0


# --- Power signs ---

def test_charge_power_positive(model):
    P = float(model.predict({"mode": "charge", "m_dot_air": 100.0})["power_kw"])
    assert P > 0.0


def test_discharge_power_positive(model):
    """Discharge power is positive (output convention in F1b)."""
    P = float(model.predict({"mode": "discharge", "m_dot_air": 400.0})["power_kw"])
    assert P > 0.0


# --- TES thermal decay ---

def test_tes_temperature_decays_over_time(model):
    m = model._model
    T1 = float(m.tes_temperature_after_idle(550.0, 3600.0))
    T2 = float(m.tes_temperature_after_idle(550.0, 86400.0))
    assert T1 > T2 > m.T_tes_ambient


def test_tes_converges_to_ambient(model):
    m = model._model
    T_inf = float(m.tes_temperature_after_idle(550.0, 10 * m.tau_tes))
    assert abs(T_inf - m.T_tes_ambient) < 1.0


# --- Cavern thermal drift ---

def test_cavern_drift_toward_rock(model):
    m = model._model
    T1 = float(m.cavern_temperature_drift(340.0, 3600.0))
    assert m.T_rock <= T1 <= 340.0


# --- SOC update ---

def test_soc_update_charge(model):
    s = model._model.soc_update(0.3, 50000.0, 1.0, "charge")
    assert s > 0.3


def test_soc_update_discharge(model):
    s = model._model.soc_update(0.7, 50000.0, 1.0, "discharge")
    assert s < 0.7


def test_soc_update_clamps(model):
    assert model._model.soc_update(0.99, 1e8, 24.0, "charge") <= 1.0 + 1e-9
    assert model._model.soc_update(0.01, 1e8, 24.0, "discharge") >= -1e-9


def test_invalid_mode(model):
    with pytest.raises(ValueError):
        model._model.soc_update(0.5, 1000.0, 1.0, "boost")


# --- Vectorized ---

def test_vectorized(model):
    m_dot = np.linspace(50, 500, 25)
    r = model.predict({"mode": "discharge", "m_dot_air": m_dot})
    assert len(r["power_kw"]) == 25


# --- Benchmark ---

def test_benchmark(model):
    m_dot = np.random.uniform(10, 500, 1000)
    soc   = np.random.uniform(0, 1, 1000)
    start = time.perf_counter()
    model.predict({"mode": "discharge", "m_dot_air": m_dot, "soc": soc})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
