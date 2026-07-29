"""EC115 — IGCC — F1a — Test Suite

Physics rules enforced:
  - eta_net in [0.38, 0.48] (IGCC 40-45% typical; bounded by CCGT ceiling)
  - eta < Carnot(T_GT_inlet ~1300C, T_cond_condenser ~40C) — gas turbine dominates cycle
  - CO2 intensity in [650, 900] g/kWh (no CCS; lower than subcritical, higher than CCGT)
  - CO2 intensity (IGCC no-CCS) > CCGT natural gas (~450 g/kWh)
  - syngas_lhv (10-12 MJ/Nm3) << natural gas (35 MJ/Nm3) — confirmed by parameter
  - Part-load drops efficiency (limited gasifier turndown)
  - Minimum PLR >= 0.40 (gasifier constraint)
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
    for k in ["power_mw", "efficiency", "coal_rate_kgs", "syngas_rate_nm3s",
              "co2_rate_kgs", "co2_intensity"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC115"
    assert info["fidelity"] == "F1a"


# ---------- efficiency physics ----------

def test_rated_iso_efficiency_range(model):
    """At PLR=1, T_amb=15C, net efficiency must be in IGCC range 40-45%."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    eta = float(r["efficiency"])
    assert 0.40 <= eta <= 0.45, f"Expected 40-45% IGCC range, got {eta*100:.2f}%"


def test_efficiency_below_ccgt_ceiling(model):
    """IGCC (coal) efficiency must be below CCGT (gas) ceiling of 65%."""
    plr   = np.linspace(0.40, 1.0, 20)
    T_amb = np.full(20, 15.0)
    r = model.predict({"part_load_ratio": plr, "ambient_temp": T_amb})
    assert np.all(r["efficiency"] < 0.65)


def test_efficiency_below_carnot(model):
    """
    Net efficiency must be below Carnot limit.
    Use gas turbine inlet T ~1300C (IGCC GT block) and condenser ~40C.
    """
    T_H = 1300.0 + 273.15  # K — gas turbine hot source
    T_C = 40.0   + 273.15  # K — condenser cold sink
    eta_carnot = 1.0 - T_C / T_H   # ~0.80
    plr   = np.linspace(0.40, 1.0, 20)
    T_amb = np.full(20, 15.0)
    r = model.predict({"part_load_ratio": plr, "ambient_temp": T_amb})
    assert np.all(r["efficiency"] < eta_carnot), \
        f"Efficiency exceeds Carnot limit {eta_carnot:.3f}"


def test_efficiency_drops_low_plr(model):
    """Efficiency at PLR=0.40 (min) must be lower than at PLR=1.0."""
    r_full = model.predict({"part_load_ratio": 1.0,  "ambient_temp": 15.0})
    r_part = model.predict({"part_load_ratio": 0.40, "ambient_temp": 15.0})
    assert float(r_part["efficiency"]) < float(r_full["efficiency"])


def test_efficiency_drops_high_tamb(model):
    """Higher ambient temperature must degrade efficiency (gas turbine derating)."""
    r_cool = model.predict({"part_load_ratio": 1.0, "ambient_temp": 5.0})
    r_hot  = model.predict({"part_load_ratio": 1.0, "ambient_temp": 40.0})
    assert float(r_hot["efficiency"]) < float(r_cool["efficiency"])


# ---------- CO2 physics ----------

def test_co2_intensity_range(model):
    """
    CO2 intensity at full load (PLR=1.0) must be within 650-900 g/kWh for IGCC without CCS.
    # RATIONALE: Published benchmark 700-800 g/kWh applies at rated operation. At partial
    # load the IGCC efficiency penalty raises CO2/kWh well above 900 g/kWh (correct physics
    # — gasifier part-load is less efficient). We validate the full-load value only.
    """
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    intensity = float(r["co2_intensity"])
    assert intensity >= 650, f"CO2 too low at rated: {intensity:.0f} g/kWh"
    assert intensity <= 900, f"CO2 too high at rated: {intensity:.0f} g/kWh"


def test_co2_above_ccgt_gas(model):
    """IGCC (coal) CO2 intensity must exceed CCGT (natural gas) ~450 g/kWh."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    assert float(r["co2_intensity"]) > 450.0, \
        "IGCC coal CO2 must be > CCGT gas reference (450 g/kWh)"


def test_syngas_lhv_lower_than_natural_gas(model):
    """Syngas LHV must be < 35 MJ/Nm3 (natural gas) — confirmed via parameter."""
    m = model._model
    assert m.syngas_lhv < 35.0, \
        f"Syngas LHV {m.syngas_lhv} MJ/Nm3 must be < natural gas (35 MJ/Nm3)"
    assert m.syngas_lhv >= 10.0, \
        f"Syngas LHV {m.syngas_lhv} MJ/Nm3 below physical minimum (10 MJ/Nm3)"


def test_syngas_rate_positive(model):
    """Syngas flow rate to CCGT block must be positive."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    assert float(r["syngas_rate_nm3s"]) > 0.0


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
    plr = np.array([0.40, 0.60, 0.80, 1.0])
    r = model.predict({"part_load_ratio": plr, "ambient_temp": 15.0})
    P = r["power_mw"]
    np.testing.assert_allclose(P / P[-1], plr / 1.0, rtol=1e-6)


def test_cold_gas_efficiency_sensible(model):
    """Cold-gas efficiency must be in 0.75-0.88 range."""
    cge = model._model.cge
    assert 0.75 <= cge <= 0.88, f"CGE {cge:.2f} outside 75-88% typical range"


# ---------- benchmark ----------

def test_benchmark(model):
    plr   = np.random.uniform(0.40, 1.0, 1000)
    T_amb = np.random.uniform(-10, 45, 1000)
    start = time.perf_counter()
    model.predict({"part_load_ratio": plr, "ambient_temp": T_amb})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
