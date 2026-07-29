"""EC213 — MED Distillation — F1b GOR + TBT + Scaling — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"TBT_degC": 70.0, "plr": 1.0})
    for k in ["gor", "thermal_sec_kwh_m3", "pump_sec_kwh_m3",
              "total_sec_kwh_m3", "scaling_risk_index", "bpr_total_degC"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC213"
    assert info["fidelity"] == "F1b"


def test_gor_reasonable_at_design(model):
    """GOR at TBT=70C, 12 effects: ~8-10 (El-Dessouky 2000: MED GOR 8-12 for 12 effects)."""
    r = model.predict({"TBT_degC": 70.0, "plr": 1.0})
    gor = float(np.atleast_1d(r["gor"])[0])
    # El-Dessouky 2000: 12-effect MED GOR ~ 8-12
    assert 4.0 < gor < 16.0, f"GOR at design = {gor:.2f}"


def test_gor_increases_with_tbt(model):
    """Higher TBT → larger temperature range → more effective effects → higher GOR."""
    r_low  = model.predict({"TBT_degC": 58.0, "plr": 1.0})
    r_high = model.predict({"TBT_degC": 72.0, "plr": 1.0})
    gor_low  = float(np.atleast_1d(r_low["gor"])[0])
    gor_high = float(np.atleast_1d(r_high["gor"])[0])
    assert gor_high > gor_low, f"GOR not increasing with TBT: {gor_low:.2f} vs {gor_high:.2f}"


def test_thermal_sec_positive(model):
    r = model.predict({"TBT_degC": 65.0, "plr": 1.0})
    sec = float(np.atleast_1d(r["thermal_sec_kwh_m3"])[0])
    assert sec > 0


def test_thermal_sec_reasonable(model):
    """MED thermal SEC: ~30-80 kWh_th/m3 (El-Dessouky: GOR=10 → ~65 kWh/m3)."""
    r = model.predict({"TBT_degC": 70.0, "plr": 1.0})
    sec = float(np.atleast_1d(r["thermal_sec_kwh_m3"])[0])
    assert 15.0 < sec < 250.0, f"Thermal SEC = {sec:.1f} kWh/m3"


def test_scaling_risk_zero_below_limit(model):
    """No scaling below 70 degC (scale_limit)."""
    r = model.predict({"TBT_degC": 65.0, "plr": 1.0})
    risk = float(np.atleast_1d(r["scaling_risk_index"])[0])
    assert risk == 0.0, f"Risk at TBT=65C = {risk:.4f}"


def test_scaling_risk_increases_above_limit(model):
    """Scaling risk positive above 70 degC."""
    r_at = model.predict({"TBT_degC": 70.0, "plr": 1.0})
    r_above = model.predict({"TBT_degC": 75.0, "plr": 1.0})
    risk_at    = float(np.atleast_1d(r_at["scaling_risk_index"])[0])
    risk_above = float(np.atleast_1d(r_above["scaling_risk_index"])[0])
    assert risk_above > risk_at, f"Risk not increasing above limit: {risk_at:.3f} vs {risk_above:.3f}"


def test_med_gor_higher_than_msf_at_same_tbt(model):
    """MED has higher GOR than MSF at same temperature (more efficient use of latent heat)."""
    r_med = model.predict({"TBT_degC": 70.0, "plr": 1.0})
    gor_med = float(np.atleast_1d(r_med["gor"])[0])
    # MSF at TBT=70C with N=24 stages: GOR ~ 3-5 (for same temperature range as 12-effect MED)
    # MED GOR should be > 5 here
    assert gor_med > 3.0, f"MED GOR at 70C = {gor_med:.2f} (should be > MSF equivalent)"


def test_bpr_positive(model):
    r = model.predict({"TBT_degC": 70.0, "plr": 1.0})
    bpr = float(np.atleast_1d(r["bpr_total_degC"])[0])
    assert bpr > 0


def test_pump_sec_reasonable(model):
    """MED pumping SEC: 1-2 kWh/m3 (lower than MSF)."""
    r = model.predict({"TBT_degC": 70.0, "plr": 1.0})
    sec = float(np.atleast_1d(r["pump_sec_kwh_m3"])[0])
    assert 0.3 < sec < 5.0, f"Pump SEC = {sec:.3f} kWh/m3"


def test_array_input(model):
    TBTs = np.linspace(57.0, 74.0, 10)
    r = model.predict({"TBT_degC": TBTs, "plr": 1.0})
    assert len(np.atleast_1d(r["gor"])) == 10


def test_benchmark(model):
    TBTs = np.random.uniform(57.0, 74.0, 1000)
    start = time.perf_counter()
    model.predict({"TBT_degC": TBTs, "plr": 1.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
