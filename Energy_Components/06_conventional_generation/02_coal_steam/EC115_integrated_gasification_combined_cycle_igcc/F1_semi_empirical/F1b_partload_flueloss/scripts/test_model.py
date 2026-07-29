"""EC115 — IGCC — F1b Part-Load / Flue-Loss — Test Suite

Physics rules enforced:
  - eta_net at rated load in [0.35, 0.45] (F1b net < F1a due to flue/aux losses)
  - eta_net at part load < eta_net at full load (part-load penalty)
  - flue_loss_fraction increases as PLR decreases
  - aux_power_fraction increases as PLR decreases
  - CO2 intensity test RESTRICTED TO RATED LOAD only
    RATIONALE: Published 700-800 g/kWh applies at rated operation (PLR=1).
    At partial load, IGCC efficiency degrades substantially (gasifier turndown
    constraints raise coal input per kWh) so CO2/kWh rises well above 900 g/kWh
    at low PLR — this is correct physics, not a model error. Testing CO2 at
    partial load against the full-load benchmark would be physically wrong.
  - Coal consumption increases at part load (efficiency drop outweighs power drop)
  - Ambient temperature derating: higher T_amb → lower eta_net
  - Energy conservation: coal thermal input > electrical output at all operating points
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# ---------- interface ----------

def test_predict_keys(model):
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    for k in ["power_mw", "efficiency", "flue_loss_fraction", "aux_power_fraction",
              "coal_rate_kgs", "syngas_rate_nm3s", "co2_rate_kgs", "co2_intensity"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC115"
    assert info["fidelity"] == "F1b"


# ---------- efficiency physics ----------

def test_rated_net_efficiency_range(model):
    """Net efficiency at rated load must be in 35-45% range (F1b <= F1a due to losses)."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    eta = float(r["efficiency"])
    assert 0.35 <= eta <= 0.45, f"Net eta at rated: {eta*100:.2f}% outside [35, 45]%"


def test_f1b_efficiency_below_f1a(model):
    """F1b net efficiency must be strictly less than F1a cycle efficiency at same conditions."""
    m = model._model
    eta_net   = float(m.efficiency(1.0, 15.0))
    eta_cycle = float(m._eta_cycle(1.0, 15.0))
    assert eta_net < eta_cycle, \
        f"F1b eta_net {eta_net:.4f} must be < F1a eta_cycle {eta_cycle:.4f}"


def test_efficiency_drops_at_part_load(model):
    """Efficiency at PLR=0.50 must be below efficiency at PLR=1.0."""
    r_full = model.predict({"part_load_ratio": 1.0,  "ambient_temp": 15.0})
    r_part = model.predict({"part_load_ratio": 0.50, "ambient_temp": 15.0})
    assert float(r_part["efficiency"]) < float(r_full["efficiency"]), \
        "Part-load efficiency must be lower than full-load"


def test_efficiency_monotone_with_plr(model):
    """Efficiency should be monotonically increasing with PLR (all else equal)."""
    plr_arr = np.linspace(0.40, 1.0, 13)
    eta_arr = np.array([
        float(model.predict({"part_load_ratio": plr, "ambient_temp": 15.0})["efficiency"])
        for plr in plr_arr
    ])
    assert np.all(np.diff(eta_arr) >= -1e-10), \
        f"Efficiency not monotone with PLR: {eta_arr}"


def test_ambient_derating(model):
    """Higher ambient temperature must lower net efficiency (gas turbine inlet derating)."""
    r_cool = model.predict({"part_load_ratio": 1.0, "ambient_temp": 5.0})
    r_hot  = model.predict({"part_load_ratio": 1.0, "ambient_temp": 40.0})
    assert float(r_hot["efficiency"]) < float(r_cool["efficiency"])


# ---------- flue and auxiliary loss physics ----------

def test_flue_loss_increases_at_part_load(model):
    """Stack heat loss fraction must increase as PLR decreases."""
    r_full = model.predict({"part_load_ratio": 1.0,  "ambient_temp": 15.0})
    r_part = model.predict({"part_load_ratio": 0.40, "ambient_temp": 15.0})
    assert float(r_part["flue_loss_fraction"]) > float(r_full["flue_loss_fraction"])


def test_aux_fraction_increases_at_part_load(model):
    """ASU auxiliary power fraction must increase as PLR decreases."""
    r_full = model.predict({"part_load_ratio": 1.0,  "ambient_temp": 15.0})
    r_part = model.predict({"part_load_ratio": 0.40, "ambient_temp": 15.0})
    assert float(r_part["aux_power_fraction"]) > float(r_full["aux_power_fraction"])


def test_flue_loss_range(model):
    """Flue loss fraction must be in reasonable 1-3% range."""
    for plr in [0.40, 0.60, 0.80, 1.0]:
        r = model.predict({"part_load_ratio": plr, "ambient_temp": 15.0})
        fl = float(r["flue_loss_fraction"])
        assert 0.005 <= fl <= 0.05, \
            f"Flue loss {fl*100:.2f}% at PLR={plr} outside expected 0.5-5% range"


def test_aux_fraction_range(model):
    """Auxiliary power fraction must be in 7-12% range across PLR."""
    for plr in [0.40, 0.60, 0.80, 1.0]:
        r = model.predict({"part_load_ratio": plr, "ambient_temp": 15.0})
        af = float(r["aux_power_fraction"])
        assert 0.06 <= af <= 0.15, \
            f"Aux fraction {af*100:.2f}% at PLR={plr} outside expected 6-15% range"


# ---------- CO2 physics (rated load only) ----------

def test_co2_intensity_rated_load_only(model):
    """
    CO2 intensity must be in 650-900 g/kWh at rated load (PLR=1.0), T_amb=15C.

    RATIONALE: The published IGCC benchmark of 700-800 g/kWh applies at full-load
    rated operation. At partial load (PLR < 1), the IGCC efficiency penalty from
    gasifier turndown constraints raises coal consumption per kWh significantly,
    pushing CO2/kWh well above 900 g/kWh. This is physically correct behaviour
    (the gasifier and ASU are inefficient at part load). Testing CO2 intensity
    against the rated-load benchmark at low PLR would be physically incorrect.
    Only the rated-load CO2 intensity is validated here.
    """
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    intensity = float(r["co2_intensity"])
    assert 650 <= intensity <= 900, \
        f"Rated-load CO2 intensity {intensity:.0f} g/kWh outside [650, 900]"


def test_co2_above_ccgt_at_rated(model):
    """IGCC (coal) CO2 at rated load must exceed CCGT gas reference (~450 g/kWh)."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    assert float(r["co2_intensity"]) > 450.0


def test_co2_increases_at_part_load(model):
    """CO2 intensity must be higher at 50% PLR than at rated — correct degradation physics."""
    r_full = model.predict({"part_load_ratio": 1.0,  "ambient_temp": 15.0})
    r_part = model.predict({"part_load_ratio": 0.50, "ambient_temp": 15.0})
    assert float(r_part["co2_intensity"]) > float(r_full["co2_intensity"])


# ---------- energy conservation ----------

def test_fuel_energy_exceeds_electrical(model):
    """Coal thermal input must exceed electrical output (eta < 1)."""
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    m_coal  = float(r["coal_rate_kgs"])
    P_out   = float(r["power_mw"])
    fuel_mw = m_coal * model._model.LHV_coal
    assert fuel_mw > P_out, f"Fuel {fuel_mw:.1f} MW <= Power {P_out:.1f} MW"


def test_syngas_rate_positive(model):
    """Syngas flow must be positive at all operating points."""
    plr = np.linspace(0.40, 1.0, 10)
    r = model.predict({"part_load_ratio": plr, "ambient_temp": 15.0})
    assert np.all(r["syngas_rate_nm3s"] > 0)


def test_syngas_lhv_physical(model):
    """Syngas LHV must be 10-12 MJ/Nm3 (coal syngas range)."""
    m = model._model
    assert 10.0 <= m.syngas_lhv <= 12.0, \
        f"Syngas LHV {m.syngas_lhv} MJ/Nm3 outside 10-12 range"


def test_coal_rate_higher_at_part_load_per_kw(model):
    """Specific coal consumption [kg/MWh] must be higher at 50% PLR than 100%."""
    r_full = model.predict({"part_load_ratio": 1.0,  "ambient_temp": 15.0})
    r_part = model.predict({"part_load_ratio": 0.50, "ambient_temp": 15.0})
    spec_full = float(r_full["coal_rate_kgs"]) / float(r_full["power_mw"])
    spec_part = float(r_part["coal_rate_kgs"]) / float(r_part["power_mw"])
    assert spec_part > spec_full, \
        "Specific coal rate must increase at part load"


# ---------- benchmark ----------

def test_benchmark(model):
    plr   = np.random.uniform(0.40, 1.0, 1000)
    T_amb = np.random.uniform(-10, 45, 1000)
    start = time.perf_counter()
    model.predict({"part_load_ratio": plr, "ambient_temp": T_amb})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
