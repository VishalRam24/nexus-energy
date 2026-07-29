"""EC124 — LAES — F1b Thermal — Test Suite"""
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
    r = model.predict({"soc": 0.5, "m_dot_liquid": 50.0, "mode": "discharge"})
    for k in ["power_kw", "round_trip_efficiency", "boil_off_rate_per_day",
              "cold_recycle_effectiveness", "effective_discharge_work_kwh_kg",
              "effective_liquefaction_work_kwh_kg", "energy_capacity_kwh"]:
        assert k in r, f"Missing key: {k}"


def test_predict_idle_keys(model):
    r = model.predict({"soc": 0.8, "time_hours": 24.0, "mode": "idle"})
    for k in ["soc_after", "liquid_mass_kg", "boil_off_fraction", "boil_off_rate_per_day"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC124"
    assert info["fidelity"] == "F1b"


# --- BOR monotonically increases with T_amb ---

def test_bor_increases_with_T_amb(model):
    """Hotter ambient → faster boil-off (more heat ingress into cryogenic tank)."""
    T_arr = [253.15, 273.15, 293.15, 303.15, 313.15, 323.15]
    bors = [float(model.predict({"soc": 0.8, "m_dot_liquid": 0.0, "T_amb_K": T})
                  ["boil_off_rate_per_day"]) for T in T_arr]
    assert all(bors[i] < bors[i + 1] for i in range(len(bors) - 1)), \
        f"BOR should increase monotonically with T_amb: {bors}"


def test_bor_at_ref_equals_nominal(model):
    """BOR at T_amb_ref should equal the reference value."""
    m = model._model
    bor = float(m.boil_off_rate_per_day(m.T_amb_ref))
    assert abs(bor - m.bor_ref) < 1e-9, \
        f"BOR at T_ref should be {m.bor_ref}, got {bor}"


def test_bor_positive(model):
    """BOR must always be positive."""
    for T in [253.15, 273.15, 298.15, 323.15]:
        bor = float(model._model.boil_off_rate_per_day(T))
        assert bor > 0, f"BOR at T={T} K should be positive, got {bor}"


def test_bor_at_ref_is_0p5_percent_per_day(model):
    """Reference BOR is ~0.5%/day — typical for large cryogenic tank."""
    bor = model._model.bor_ref
    assert 0.003 <= bor <= 0.010, \
        f"Reference BOR should be 0.3-1.0 %/day, got {bor * 100:.3f} %/day"


# --- Boil-off self-discharge during standby ---

def test_soc_decreases_during_standby(model):
    """SOC should decrease monotonically during idle (boil-off)."""
    times = [0, 24, 48, 72, 168]
    socs = [float(model.predict({"soc": 0.9, "time_hours": t, "mode": "idle",
                                  "T_amb_K": 298.15})["soc_after"]) for t in times]
    assert all(socs[i] >= socs[i + 1] for i in range(len(socs) - 1)), \
        f"SOC should monotonically decrease: {socs}"


def test_soc_after_zero_time_unchanged(model):
    """Zero standby time → SOC unchanged."""
    soc0 = 0.7
    r = model.predict({"soc": soc0, "time_hours": 0.0, "mode": "idle", "T_amb_K": 298.15})
    assert abs(float(r["soc_after"]) - soc0) < 1e-9


def test_hot_ambient_faster_boiloff(model):
    """Hotter ambient → more SOC lost after same standby duration."""
    soc0 = 0.9
    t_h = 168.0   # 1 week
    soc_cold = float(model.predict({"soc": soc0, "time_hours": t_h,
                                     "T_amb_K": 253.15, "mode": "idle"})["soc_after"])
    soc_hot = float(model.predict({"soc": soc0, "time_hours": t_h,
                                    "T_amb_K": 323.15, "mode": "idle"})["soc_after"])
    assert soc_hot < soc_cold, \
        f"Hotter ambient should cause more boil-off: soc_hot={soc_hot:.4f} < soc_cold={soc_cold:.4f}"


def test_boiloff_fraction_bounded_0_1(model):
    """Boil-off fraction should be in [0, 1]."""
    for t_h in [0, 24, 168, 8760]:
        r = model.predict({"soc": 1.0, "time_hours": t_h, "T_amb_K": 298.15, "mode": "idle"})
        bf = float(r["boil_off_fraction"])
        assert 0.0 <= bf <= 1.0, f"Boil-off fraction {bf} out of [0,1] at t={t_h}h"


def test_boiloff_fraction_increases_with_time(model):
    """More time in storage → more cumulative boil-off loss."""
    times = [24, 48, 168, 336, 720]
    fracs = [float(model.predict({"soc": 1.0, "time_hours": t,
                                   "T_amb_K": 298.15, "mode": "idle"})["boil_off_fraction"])
             for t in times]
    assert all(fracs[i] < fracs[i + 1] for i in range(len(fracs) - 1)), \
        f"Boil-off fraction should increase with time: {fracs}"


# --- Cold recycle effectiveness ---

def test_cold_recycle_decreases_with_T_amb(model):
    """Warmer ambient → worse cold recovery from re-gasification."""
    eps_cold = float(model._model.cold_recycle_effectiveness(253.15))
    eps_warm = float(model._model.cold_recycle_effectiveness(313.15))
    assert eps_cold > eps_warm, \
        f"Cold recycle should be better at lower T_amb: eps(cold)={eps_cold:.4f} > eps(warm)={eps_warm:.4f}"


def test_cold_recycle_at_ref_equals_nominal(model):
    """At T_amb_ref, cold recycle effectiveness = eps_ref."""
    m = model._model
    eps = float(m.cold_recycle_effectiveness(m.T_amb_ref))
    assert abs(eps - m.eps_ref) < 1e-9


def test_effective_liq_work_increases_with_T_amb(model):
    """Warmer ambient → worse cold recycle → more effective liquefaction work needed."""
    m = model._model
    w_cold = float(m.effective_liquefaction_work(253.15))
    w_warm = float(m.effective_liquefaction_work(313.15))
    assert w_warm > w_cold, \
        f"Effective liq work should increase with T_amb: w_warm={w_warm:.4f} > w_cold={w_cold:.4f}"


# --- Discharge specific work ---

def test_discharge_work_decreases_with_T_amb(model):
    """Warmer ambient reduces available expansion work."""
    m = model._model
    w_cold = float(m.effective_discharge_work(253.15))
    w_warm = float(m.effective_discharge_work(313.15))
    assert w_warm < w_cold, \
        f"Discharge work should decrease with T_amb: w_warm={w_warm:.4f} < w_cold={w_cold:.4f}"


def test_discharge_work_at_ref_equals_nominal(model):
    m = model._model
    w = float(m.effective_discharge_work(m.T_amb_ref))
    assert abs(w - m.w_disch_ref) < 1e-9


# --- RTE ---

def test_rte_decreases_with_T_amb(model):
    """Warmer ambient reduces LAES RTE (worse cold recycle, less discharge work)."""
    rte_cold = float(model.predict({"soc": 0.5, "m_dot_liquid": 50.0,
                                     "T_amb_K": 253.15})["round_trip_efficiency"])
    rte_warm = float(model.predict({"soc": 0.5, "m_dot_liquid": 50.0,
                                     "T_amb_K": 313.15})["round_trip_efficiency"])
    assert rte_warm < rte_cold, \
        f"RTE should decrease with T_amb: rte_warm={rte_warm:.4f} < rte_cold={rte_cold:.4f}"


def test_rte_typical_range(model):
    """LAES RTE typically 0.45–0.70."""
    r = model.predict({"soc": 0.5, "m_dot_liquid": 50.0, "T_amb_K": 298.15})
    rte = float(r["round_trip_efficiency"])
    assert 0.30 <= rte <= 0.80, f"LAES RTE={rte:.4f} outside [0.30, 0.80]"


# --- Power ---

def test_charge_power_positive(model):
    r = model.predict({"soc": 0.5, "m_dot_liquid": 50.0, "mode": "charge"})
    assert float(r["power_kw"]) > 0


def test_discharge_power_positive(model):
    r = model.predict({"soc": 0.5, "m_dot_liquid": 50.0, "mode": "discharge"})
    assert float(r["power_kw"]) > 0


def test_zero_flow_zero_power(model):
    r = model.predict({"soc": 0.5, "m_dot_liquid": 0.0, "mode": "discharge"})
    assert float(r["power_kw"]) == 0.0


def test_charge_power_higher_at_warm_ambient(model):
    """Warmer ambient → more effective liq work → more charge power per kg/s."""
    r_cold = model.predict({"soc": 0.5, "m_dot_liquid": 50.0,
                             "T_amb_K": 253.15, "mode": "charge"})
    r_warm = model.predict({"soc": 0.5, "m_dot_liquid": 50.0,
                             "T_amb_K": 313.15, "mode": "charge"})
    assert float(r_warm["power_kw"]) > float(r_cold["power_kw"]), \
        "Warm ambient should require more charge power (worse cold recycle)"


# --- Benchmark ---

def test_benchmark_1000(model):
    soc = np.random.uniform(0, 1, 1000)
    m_dot = np.random.uniform(0, 150, 1000)
    T_amb = np.random.uniform(253.15, 323.15, 1000)
    start = time.perf_counter()
    model.predict({"soc": soc, "m_dot_liquid": m_dot, "T_amb_K": T_amb, "mode": "discharge"})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 0.2
