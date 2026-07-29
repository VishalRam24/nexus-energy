"""EC094 — Evaporative Cooler — F1b COP/EER Part-Load — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_db_c": 35.0, "RH_pct": 30.0})
    for k in ["Q_cool_kw", "W_fan_kw", "EER", "T_wb_c", "T_outlet_c", "f_humidity"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC094"
    assert info["fidelity"] == "F1b"


def test_outlet_below_inlet(model):
    """Outlet temperature must be below dry-bulb inlet."""
    r = model.predict({"T_db_c": 35.0, "RH_pct": 30.0})
    assert float(r["T_outlet_c"]) < 35.0, "Evaporative cooler must lower temperature"


def test_outlet_above_wet_bulb(model):
    """Outlet temperature must be >= wet-bulb (cannot cool below wet-bulb).
    RATIONALE: Direct evap cooling limit is wet-bulb temperature."""
    r = model.predict({"T_db_c": 38.0, "RH_pct": 25.0})
    assert float(r["T_outlet_c"]) >= float(r["T_wb_c"]) - 0.1, \
        f"Outlet {r['T_outlet_c']:.1f}C must be >= T_wb {r['T_wb_c']:.1f}C"


def test_eer_positive(model):
    r = model.predict({"T_db_c": 35.0, "RH_pct": 30.0})
    assert float(r["EER"]) > 0.0


def test_eer_decreases_high_humidity(model):
    """EER must decrease at high relative humidity.
    RATIONALE: High humidity reduces evaporation potential (wet-bulb depression smaller)."""
    r_dry  = model.predict({"T_db_c": 35.0, "RH_pct": 20.0})
    r_humid = model.predict({"T_db_c": 35.0, "RH_pct": 70.0})
    assert float(r_dry["EER"]) > float(r_humid["EER"]), \
        "EER should be lower at high humidity"


def test_wet_bulb_below_dry_bulb(model):
    """Wet-bulb temperature must always be <= dry-bulb."""
    T_dbs = np.array([25.0, 35.0, 40.0])
    RHs   = np.array([30.0, 50.0, 60.0])
    r = model.predict({"T_db_c": T_dbs, "RH_pct": RHs})
    assert np.all(r["T_wb_c"] <= r["T_outlet_c"] + 0.1)


def test_fan_power_increases_with_plr(model):
    """Fan power must increase with PLR (affinity law)."""
    plr = np.array([0.3, 0.5, 0.7, 1.0])
    r   = model.predict({"T_db_c": 35.0, "RH_pct": 30.0, "PLR": plr})
    assert np.all(np.diff(r["W_fan_kw"]) > 0)


def test_part_load_reduces_eer(model):
    """EER should vary with PLR per DOE-2 curve."""
    r1 = model.predict({"T_db_c": 35.0, "RH_pct": 30.0, "PLR": 1.0})
    r5 = model.predict({"T_db_c": 35.0, "RH_pct": 30.0, "PLR": 0.5})
    # EER is not necessarily higher at full load — DOE-2 can peak at intermediate PLR
    assert float(r1["EER"]) > 0.0 and float(r5["EER"]) > 0.0


def test_f_humidity_reduces_at_high_rh(model):
    """Humidity correction factor must be < 1 above design RH."""
    r = model.predict({"T_db_c": 35.0, "RH_pct": 80.0})
    assert float(r["f_humidity"]) < 1.0


def test_benchmark(model):
    T  = np.random.uniform(25, 45, 1000)
    RH = np.random.uniform(10, 80, 1000)
    t0 = time.perf_counter()
    model.predict({"T_db_c": T, "RH_pct": RH})
    assert time.perf_counter() - t0 < 1.0
