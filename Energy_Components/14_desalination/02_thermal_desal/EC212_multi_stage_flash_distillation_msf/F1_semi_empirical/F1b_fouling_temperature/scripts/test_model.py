"""EC212 — MSF Distillation — F1b GOR + TBT + Scaling — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"TBT_degC": 110.0, "plr": 1.0})
    for k in ["gor", "thermal_sec_kwh_m3", "pump_sec_kwh_m3",
              "total_sec_kwh_m3", "scaling_risk_index"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC212"
    assert info["fidelity"] == "F1b"


def test_gor_reasonable_range(model):
    """GOR at design TBT=110C should be 6-12 (El-Dessouky 2002: MSF GOR 6-10)."""
    r = model.predict({"TBT_degC": 110.0, "plr": 1.0})
    gor = float(np.atleast_1d(r["gor"])[0])
    assert 4.0 < gor < 14.0, f"GOR at TBT=110C = {gor:.2f} (expected 6-12)"


def test_gor_increases_with_tbt(model):
    """Higher TBT → higher GOR (larger flash range)."""
    r_low  = model.predict({"TBT_degC": 80.0, "plr": 1.0})
    r_high = model.predict({"TBT_degC": 115.0, "plr": 1.0})
    gor_low  = float(np.atleast_1d(r_low["gor"])[0])
    gor_high = float(np.atleast_1d(r_high["gor"])[0])
    assert gor_high > gor_low, f"GOR not increasing with TBT: {gor_low:.2f} vs {gor_high:.2f}"


def test_thermal_sec_positive(model):
    r = model.predict({"TBT_degC": 100.0, "plr": 1.0})
    sec = float(np.atleast_1d(r["thermal_sec_kwh_m3"])[0])
    assert sec > 0


def test_thermal_sec_reasonable(model):
    """Thermal SEC at TBT=110C: ~50-120 kWh_th/m3 (El-Dessouky: GOR=8 → ~70 kWh/m3)."""
    r = model.predict({"TBT_degC": 110.0, "plr": 1.0})
    sec = float(np.atleast_1d(r["thermal_sec_kwh_m3"])[0])
    # El-Dessouky: MSF SEC 50-150 kWh_th/m3; PR=8 → ~70 kWh/m3
    assert 30.0 < sec < 300.0, f"Thermal SEC = {sec:.1f} kWh/m3"


def test_scaling_risk_zero_below_limit(model):
    """No scaling risk below T_scale_limit (90 degC)."""
    r = model.predict({"TBT_degC": 85.0, "plr": 1.0})
    risk = float(np.atleast_1d(r["scaling_risk_index"])[0])
    assert risk == 0.0, f"Scaling risk at TBT=85C = {risk:.3f} (should be 0)"


def test_scaling_risk_positive_above_limit(model):
    """Scaling risk should be > 0 above 90 degC."""
    r = model.predict({"TBT_degC": 100.0, "plr": 1.0})
    risk = float(np.atleast_1d(r["scaling_risk_index"])[0])
    assert risk > 0, f"Scaling risk at TBT=100C = {risk:.3f} (should be > 0)"


def test_scaling_risk_increases_with_tbt(model):
    """Scaling risk increases with TBT above limit."""
    r_low  = model.predict({"TBT_degC": 95.0, "plr": 1.0})
    r_high = model.predict({"TBT_degC": 115.0, "plr": 1.0})
    risk_low  = float(np.atleast_1d(r_low["scaling_risk_index"])[0])
    risk_high = float(np.atleast_1d(r_high["scaling_risk_index"])[0])
    assert risk_high > risk_low, f"Risk not increasing: {risk_low:.3f} vs {risk_high:.3f}"


def test_pump_sec_increases_at_part_load(model):
    """Pumping SEC should be higher at lower PLR (efficiency drop)."""
    r_full = model.predict({"TBT_degC": 110.0, "plr": 1.0})
    r_part = model.predict({"TBT_degC": 110.0, "plr": 0.5})
    sec_full = float(np.atleast_1d(r_full["pump_sec_kwh_m3"])[0])
    sec_part = float(np.atleast_1d(r_part["pump_sec_kwh_m3"])[0])
    assert sec_part >= sec_full, f"Pump SEC not higher at part-load: {sec_full:.3f} vs {sec_part:.3f}"


def test_total_sec_greater_than_thermal(model):
    r = model.predict({"TBT_degC": 110.0, "plr": 1.0})
    total = float(np.atleast_1d(r["total_sec_kwh_m3"])[0])
    th    = float(np.atleast_1d(r["thermal_sec_kwh_m3"])[0])
    assert total > th


def test_array_input(model):
    TBTs = np.linspace(80.0, 115.0, 10)
    r = model.predict({"TBT_degC": TBTs, "plr": 1.0})
    assert len(np.atleast_1d(r["gor"])) == 10


def test_benchmark(model):
    TBTs = np.random.uniform(75.0, 115.0, 1000)
    start = time.perf_counter()
    model.predict({"TBT_degC": TBTs, "plr": 1.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
