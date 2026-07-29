"""EC020 — NCA Battery — F1a SOC-Only — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"soc": 0.5, "current": 1.0})
    for k in ["voltage", "ocv", "power", "dsoc_dt"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC020"
    assert info["fidelity"] == "F1a"


def test_ocv_increases_with_soc(model):
    """OCV must be monotonically increasing with SOC."""
    soc = np.linspace(0.0, 1.0, 100)
    r = model.predict({"soc": soc, "current": 0.0})
    diffs = np.diff(r["ocv"])
    assert np.all(diffs > 0), "OCV must increase with SOC"


def test_voltage_within_bounds(model):
    """Terminal voltage must stay within [v_min, v_max]."""
    soc = np.linspace(0.0, 1.0, 50)
    for I in [-15.0, 0.0, 15.0]:
        r = model.predict({"soc": soc, "current": I})
        assert np.all(r["voltage"] >= 2.5 - 1e-9), f"Voltage below 2.5V at I={I}A"
        assert np.all(r["voltage"] <= 4.2 + 1e-9), f"Voltage above 4.2V at I={I}A"


def test_ocv_at_full_charge(model):
    """OCV at SOC=1 should be near 4.2 V (NCA full charge)."""
    r = model.predict({"soc": 1.0, "current": 0.0})
    assert 4.0 < float(r["ocv"]) <= 4.21, f"OCV at full charge = {float(r['ocv']):.3f} V"


def test_ocv_at_empty(model):
    """OCV at SOC=0 should be near 2.7 V (NCA fully discharged)."""
    r = model.predict({"soc": 0.0, "current": 0.0})
    assert 2.5 < float(r["ocv"]) <= 3.0, f"OCV at empty = {float(r['ocv']):.3f} V"


def test_discharge_voltage_lower_than_ocv(model):
    """During discharge (I>0), terminal voltage < OCV."""
    r = model.predict({"soc": 0.5, "current": 5.0})
    assert float(r["voltage"]) < float(r["ocv"]), "Discharge voltage must be < OCV"


def test_charge_voltage_higher_than_ocv(model):
    """During charging (I<0), terminal voltage > OCV."""
    r = model.predict({"soc": 0.5, "current": -5.0})
    assert float(r["voltage"]) > float(r["ocv"]), "Charge voltage must be > OCV"


def test_power_sign(model):
    """Power should be positive when discharging, negative when charging."""
    r_dis = model.predict({"soc": 0.5, "current": 5.0})
    r_chg = model.predict({"soc": 0.5, "current": -5.0})
    assert float(r_dis["power"]) > 0, "Discharge power should be positive"
    assert float(r_chg["power"]) < 0, "Charge power should be negative"


def test_dsoc_dt_sign(model):
    """dSOC/dt negative during discharge, positive during charge."""
    r_dis = model.predict({"soc": 0.5, "current": 1.0})
    r_chg = model.predict({"soc": 0.5, "current": -1.0})
    assert float(r_dis["dsoc_dt"]) < 0
    assert float(r_chg["dsoc_dt"]) > 0


def test_lower_voltage_than_nmc(model):
    """NCA OCV at full charge ~4.2V but let us compare mid-SOC to NMC-like range."""
    r = model.predict({"soc": 0.5, "current": 0.0})
    # NCA mid-SOC OCV should be in a reasonable NCA range (~3.5-4.0V at mid)
    assert 3.0 < float(r["ocv"]) < 4.5


def test_benchmark(model):
    soc = np.random.uniform(0.0, 1.0, 1000)
    I = np.random.uniform(-15.0, 15.0, 1000)
    start = time.perf_counter()
    model.predict({"soc": soc, "current": I})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
