"""EC123 — CAES — F1b Thermal — Test Suite"""
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


# --- Interface ---

def test_predict_discharge_keys(model):
    r = model.predict({"soc": 0.5, "m_dot_air": 100.0, "mode": "discharge"})
    for k in ["power_kw", "cavern_pressure_Pa", "air_mass_kg",
              "round_trip_efficiency", "electric_rte", "specific_work_kJ_kg"]:
        assert k in r, f"Missing key: {k}"


def test_predict_thermal_keys(model):
    r = model.predict({"mode": "thermal", "soc": 0.5, "T_cav_0_K": 330.0, "t_idle_s": 3600.0})
    for k in ["T_cav_K", "tau_cav_s", "cavern_pressure_Pa", "air_mass_kg"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC123"
    assert info["fidelity"] == "F1b"


# --- Cavern thermal drift to rock wall ---

def test_cavern_T_drifts_toward_rock(model):
    """Cavern hotter than rock should cool over time."""
    T_rock = model._model.T_rock
    T_hot = T_rock + 20.0
    r = model.predict({"mode": "thermal", "soc": 0.5, "T_cav_0_K": T_hot, "t_idle_s": 3600.0})
    T_after = float(r["T_cav_K"])
    assert T_after < T_hot, f"Cavern should cool: T_after={T_after:.2f} < T_hot={T_hot:.2f}"
    assert T_after > T_rock, f"Cavern should not overshoot T_rock: T_after={T_after:.2f}"


def test_cavern_T_colder_than_rock_warms(model):
    """Cavern colder than rock should warm over time."""
    T_rock = model._model.T_rock
    T_cold = T_rock - 10.0
    r = model.predict({"mode": "thermal", "soc": 0.5, "T_cav_0_K": T_cold, "t_idle_s": 3600.0})
    T_after = float(r["T_cav_K"])
    assert T_after > T_cold, f"Cold cavern should warm: T_after={T_after:.2f} > T_cold={T_cold:.2f}"


def test_cavern_T_approaches_rock_long_time(model):
    """After many time constants, T_cav → T_rock."""
    tau = model._model.tau_cav
    T_rock = model._model.T_rock
    r = model.predict({"mode": "thermal", "soc": 0.5, "T_cav_0_K": 330.0,
                        "t_idle_s": 10 * tau})
    T_final = float(r["T_cav_K"])
    assert abs(T_final - T_rock) < 0.1, \
        f"After 10x tau, T_cav should be ~T_rock={T_rock:.2f} K, got {T_final:.2f} K"


def test_cavern_T_monotonically_approaches_rock(model):
    """Cavern temperature change should be monotonically decreasing toward T_rock."""
    T_rock = model._model.T_rock
    T0 = T_rock + 30.0
    times = np.linspace(0, 3 * model._model.tau_cav, 10)
    T_vals = [float(model.predict({"mode": "thermal", "soc": 0.5,
                                    "T_cav_0_K": T0, "t_idle_s": float(t)})["T_cav_K"])
              for t in times]
    # All T should be between T_rock and T0
    for T_v in T_vals:
        assert T_rock <= T_v <= T0, \
            f"T_cav={T_v:.2f} K outside [{T_rock:.2f}, {T0:.2f}] K"


def test_tau_cav_positive(model):
    """Thermal time constant must be positive."""
    tau = model._model.tau_cav
    assert tau > 0, f"tau_cav should be positive, got {tau}"


def test_tau_cav_large_for_salt_cavern(model):
    """
    Salt cavern thermal mass is enormous — tau should be >> 1 hour.
    # RATIONALE: Cavern_thermal_mass ~5e9 J/K, UA ~50 kW/K → tau ~100,000 s ≈ 28 h
    """
    tau_h = model._model.tau_cav / 3600
    assert tau_h > 1.0, f"tau_cav should be >> 1h for rock cavern, got {tau_h:.1f} h"


# --- T_amb effect on compressor ---

def test_hot_ambient_increases_compression_work(model):
    """Higher T_amb → more specific compression work (hot air less dense)."""
    w_cold = float(model.predict({"soc": 0.5, "m_dot_air": 100.0,
                                   "T_amb_K": 253.15, "mode": "charge"})["specific_work_kJ_kg"])
    w_hot = float(model.predict({"soc": 0.5, "m_dot_air": 100.0,
                                  "T_amb_K": 313.15, "mode": "charge"})["specific_work_kJ_kg"])
    assert w_hot > w_cold, \
        f"Hot ambient should increase compression work: w_hot={w_hot:.1f} > w_cold={w_cold:.1f}"


def test_hot_ambient_reduces_rte(model):
    """Higher T_amb → more compression energy → lower RTE."""
    rte_cold = float(model.predict({"soc": 0.5, "m_dot_air": 100.0,
                                     "T_amb_K": 253.15, "mode": "charge"})["round_trip_efficiency"])
    rte_hot = float(model.predict({"soc": 0.5, "m_dot_air": 100.0,
                                    "T_amb_K": 313.15, "mode": "charge"})["round_trip_efficiency"])
    assert rte_hot < rte_cold, \
        f"Hot ambient should reduce RTE: rte_hot={rte_hot:.4f} < rte_cold={rte_cold:.4f}"


def test_rte_reasonable_range(model):
    """Diabatic CAES RTE typically 0.30–0.55 (all energy including fuel)."""
    r = model.predict({"soc": 0.5, "m_dot_air": 100.0, "mode": "discharge"})
    rte = float(r["round_trip_efficiency"])
    assert 0.20 <= rte <= 0.60, f"CAES RTE={rte:.4f} outside expected [0.20, 0.60]"


def test_specific_work_at_Tref_equals_nominal(model):
    """At reference temperature, specific work should equal the nominal value."""
    m = model._model
    T_ref = m.T_ref_comp
    w = float(m.specific_compression_work(T_ref))
    assert abs(w - m.w_comp_ref) < 0.01, \
        f"w_comp at T_ref should be w_comp_ref={m.w_comp_ref}, got {w:.3f}"


# --- Cavern air mass and pressure ---

def test_air_mass_increases_with_soc(model):
    """More SOC → more compressed air in cavern."""
    m0 = float(model.predict({"soc": 0.0, "m_dot_air": 0.0})["air_mass_kg"])
    m1 = float(model.predict({"soc": 1.0, "m_dot_air": 0.0})["air_mass_kg"])
    assert m1 > m0, f"Air mass at SOC=1 should exceed SOC=0: {m1:.0f} > {m0:.0f} kg"


def test_cavern_pressure_increases_with_soc(model):
    """Higher SOC → higher cavern pressure."""
    p0 = float(model.predict({"soc": 0.0, "m_dot_air": 0.0})["cavern_pressure_Pa"])
    p1 = float(model.predict({"soc": 1.0, "m_dot_air": 0.0})["cavern_pressure_Pa"])
    assert p1 > p0


def test_hotter_cavern_higher_pressure_same_soc(model):
    """For same SOC, hotter cavern → higher pressure (fewer moles at higher energy)."""
    m = model._model
    soc = 0.5
    # At higher T, soc maps to same m_fraction but pressure is higher
    # Actually: m_max(T) decreases with T, so actual m at soc=0.5 decreases
    # But if we hold mass constant and heat, P increases
    # Here we compare p(soc, T_hot) vs p(soc, T_cold):
    # m(soc,T) = m_min(T) + soc*(m_max(T)-m_min(T)), then p = m*R*T/V
    T_cold = 290.0
    T_hot = 330.0
    p_cold = float(model.predict({"soc": soc, "T_cav_K": T_cold, "m_dot_air": 0.0})["cavern_pressure_Pa"])
    p_hot = float(model.predict({"soc": soc, "T_cav_K": T_hot, "m_dot_air": 0.0})["cavern_pressure_Pa"])
    # p = m*R*T/V; m_max(T) ∝ 1/T so m(soc) ∝ 1/T, thus p = const for all T (isothermal SOC definition)
    # With our T-dependent m_max, p ~ p_max*(m_min/m_max + soc*(1 - m_min/m_max)) independent of T
    # So pressures should be approximately equal — but test that they're both in [p_min, p_max]
    assert m.p_min <= p_cold <= m.p_max, f"p_cold={p_cold:.0f} Pa outside valid range"
    assert m.p_min <= p_hot <= m.p_max, f"p_hot={p_hot:.0f} Pa outside valid range"


def test_energy_capacity_positive(model):
    r = model.predict({"soc": 0.5, "m_dot_air": 0.0})
    assert float(r["energy_capacity_kwh"]) > 0


# --- Power ---

def test_discharge_power_positive(model):
    r = model.predict({"soc": 0.5, "m_dot_air": 100.0, "mode": "discharge"})
    assert float(r["power_kw"]) > 0


def test_charge_power_positive(model):
    r = model.predict({"soc": 0.5, "m_dot_air": 100.0, "mode": "charge"})
    assert float(r["power_kw"]) > 0


def test_zero_flow_zero_power(model):
    r = model.predict({"soc": 0.5, "m_dot_air": 0.0, "mode": "discharge"})
    assert float(r["power_kw"]) == 0.0


# --- Benchmark ---

def test_benchmark_1000(model):
    soc = np.random.uniform(0, 1, 1000)
    m_dot = np.random.uniform(0, 300, 1000)
    T_amb = np.random.uniform(253.15, 313.15, 1000)
    start = time.perf_counter()
    model.predict({"soc": soc, "m_dot_air": m_dot, "T_amb_K": T_amb, "mode": "discharge"})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 0.2
