"""EC114 — Supercritical / Ultra-Supercritical Coal Plant — F1a — Test Suite

Physics rules enforced:
  - eta_net in [0.42, 0.50] (SC/USC range; above subcritical, below Carnot)
  - eta_net (SC/USC) > eta_net (subcritical ~37%) — confirmed by parameter bound
  - eta < eta_Carnot(T_steam=700C, T_cond=40C)
  - CO2 intensity in [700, 900] g/kWh (lower than subcritical due to higher efficiency)
  - CO2 intensity > CCGT (~400-500 g/kWh) — still coal
  - Part-load reduces efficiency
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# ---------- interface tests ----------

def test_predict_keys(model):
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    for k in ["power_mw", "efficiency", "coal_rate_kgs", "co2_rate_kgs", "co2_intensity"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC114"
    assert info["fidelity"] == "F1a"


# ---------- efficiency physics ----------

def test_rated_iso_efficiency_range(model):
    """At PLR=1, T_amb=15C, net efficiency must be in SC/USC range 42-47%."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    eta = float(r["efficiency"])
    assert 0.42 <= eta <= 0.47, f"Expected 42-47% SC/USC, got {eta*100:.2f}%"


def test_efficiency_above_subcritical(model):
    """SC/USC efficiency at rated must exceed subcritical upper bound (38%)."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    eta = float(r["efficiency"])
    assert eta > 0.38, f"SC/USC efficiency {eta*100:.1f}% must exceed subcritical max (38%)"


def test_efficiency_below_carnot(model):
    """Net efficiency must be below Carnot limit for steam at 700C condenser at 40C."""
    eta_carnot = model._model.carnot_efficiency(T_cond_c=40.0)
    plr   = np.linspace(0.30, 1.0, 30)
    T_amb = np.full(30, 15.0)
    r = model.predict({"part_load_ratio": plr, "ambient_temp": T_amb})
    assert np.all(r["efficiency"] < eta_carnot), \
        f"Efficiency exceeds Carnot limit {eta_carnot:.3f}"


def test_efficiency_below_50pct(model):
    """Practical upper bound: no coal plant reaches 50% LHV efficiency."""
    plr   = np.linspace(0.30, 1.0, 30)
    T_amb = np.full(30, 15.0)
    r = model.predict({"part_load_ratio": plr, "ambient_temp": T_amb})
    assert np.all(r["efficiency"] < 0.50), "Efficiency exceeded 50% — physically unreachable for coal"


def test_efficiency_drops_low_plr(model):
    """Efficiency at PLR=0.30 must be lower than at PLR=1.0 (ISO conditions)."""
    r_full = model.predict({"part_load_ratio": 1.0,  "ambient_temp": 15.0})
    r_part = model.predict({"part_load_ratio": 0.30, "ambient_temp": 15.0})
    assert float(r_part["efficiency"]) < float(r_full["efficiency"])


def test_efficiency_drops_high_tamb(model):
    """Higher ambient temperature must degrade efficiency."""
    r_cool = model.predict({"part_load_ratio": 1.0, "ambient_temp": 5.0})
    r_hot  = model.predict({"part_load_ratio": 1.0, "ambient_temp": 40.0})
    assert float(r_hot["efficiency"]) < float(r_cool["efficiency"])


# ---------- CO2 physics ----------

def test_co2_intensity_range(model):
    """
    CO2 intensity at full load (PLR=1.0) must be within 700-900 g/kWh for SC/USC coal.
    # RATIONALE: Published benchmark 750-850 g/kWh applies at rated operation. At partial
    # load the part-load efficiency penalty raises CO2/kWh significantly above 900 g/kWh
    # (correct physics). We validate the full-load value only to avoid penalising correct
    # part-load behaviour; same approach as EC113.
    """
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    intensity = float(r["co2_intensity"])
    assert intensity >= 700, f"CO2 intensity too low at rated load: {intensity:.0f} g/kWh"
    assert intensity <= 900, f"CO2 intensity too high at rated load: {intensity:.0f} g/kWh"


def test_co2_intensity_lower_than_subcritical(model):
    """
    At same PLR, SC/USC CO2 intensity must be lower than subcritical (~1000 g/kWh at full load).
    SC/USC at rated: ~800 g/kWh; subcritical at rated: ~950 g/kWh.
    """
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    assert float(r["co2_intensity"]) < 950.0, \
        "SC/USC CO2 intensity must be < subcritical reference (950 g/kWh)"


def test_co2_intensity_above_gas(model):
    """Coal CO2 intensity must exceed CCGT reference (< 600 g/kWh)."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    assert float(r["co2_intensity"]) > 600.0, \
        "Coal CO2 intensity must be > 600 g/kWh (greater than gas turbine)"


# ---------- energy conservation ----------

def test_fuel_energy_exceeds_electrical(model):
    """Coal thermal input must exceed electrical output (eta < 1)."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    P_out_mw = float(r["power_mw"])
    m_coal   = float(r["coal_rate_kgs"])
    fuel_mw  = m_coal * model._model.LHV_coal
    assert fuel_mw > P_out_mw, f"Fuel {fuel_mw:.1f} MW_th <= Power {P_out_mw:.1f} MW_e"


def test_power_scales_with_plr(model):
    """Electrical output must scale linearly with PLR."""
    plr = np.array([0.30, 0.50, 0.75, 1.0])
    r = model.predict({"part_load_ratio": plr, "ambient_temp": 15.0})
    P = r["power_mw"]
    np.testing.assert_allclose(P / P[-1], plr / 1.0, rtol=1e-6)


def test_coal_rate_positive(model):
    plr   = np.linspace(0.30, 1.0, 10)
    T_amb = np.full(10, 15.0)
    r = model.predict({"part_load_ratio": plr, "ambient_temp": T_amb})
    assert np.all(r["coal_rate_kgs"] > 0)


# ---------- benchmark ----------

def test_benchmark(model):
    plr   = np.random.uniform(0.30, 1.0, 1000)
    T_amb = np.random.uniform(-10, 45, 1000)
    start = time.perf_counter()
    model.predict({"part_load_ratio": plr, "ambient_temp": T_amb})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
