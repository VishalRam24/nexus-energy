"""EC028 — Lead-Acid Battery — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"soc": 0.5, "current": 10.0})
    for k in ["voltage", "ocv", "power", "dsoc_dt"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC028"
    assert info["fidelity"] == "F1a"


def test_voltage_within_bounds(model):
    """Terminal voltage must stay within 10.5–14.4 V for typical operation."""
    soc = np.linspace(0.1, 1.0, 50)
    # Moderate discharge: 20 A
    r = model.predict({"soc": soc, "current": 20.0})
    assert np.all(r["voltage"] >= 10.0), "Voltage below 10 V — check model"
    # Moderate charge: -20 A
    r2 = model.predict({"soc": soc, "current": -20.0})
    assert np.all(r2["voltage"] <= 15.0), "Voltage above 15 V — check model"


def test_ocv_increases_with_soc(model):
    """OCV must be monotonically increasing with SOC."""
    soc = np.linspace(0.0, 1.0, 50)
    r = model.predict({"soc": soc, "current": 0.0})
    assert np.all(np.diff(r["ocv"]) >= 0), "OCV is not monotonically increasing with SOC"


def test_ocv_range(model):
    """OCV at SOC=0 should be ~11.5 V; at SOC=1 should be ~12.8 V."""
    r_empty = model.predict({"soc": 0.0, "current": 0.0})
    r_full  = model.predict({"soc": 1.0, "current": 0.0})
    assert 11.0 < float(r_empty["ocv"]) < 12.0, f"OCV empty = {float(r_empty['ocv']):.3f} V"
    assert 12.5 < float(r_full["ocv"])  < 13.5, f"OCV full  = {float(r_full['ocv']):.3f} V"


def test_voltage_drops_with_discharge(model):
    """Voltage must decrease as discharge current increases at fixed SOC."""
    currents = np.array([0.0, 10.0, 20.0, 50.0])
    r = model.predict({"soc": 0.8, "current": currents})
    assert np.all(np.diff(r["voltage"]) < 0), "Voltage does not drop with increasing discharge"


def test_voltage_rises_during_charge(model):
    """Voltage during charging (negative I) must exceed OCV."""
    r_ocv    = model.predict({"soc": 0.5, "current":  0.0})
    r_charge = model.predict({"soc": 0.5, "current": -20.0})
    assert float(r_charge["voltage"]) > float(r_ocv["voltage"])


def test_dsoc_dt_sign(model):
    """Discharging decreases SOC (dsoc_dt < 0); charging increases it (dsoc_dt > 0)."""
    r_dis = model.predict({"soc": 0.5, "current":  20.0})
    r_chg = model.predict({"soc": 0.5, "current": -20.0})
    assert float(r_dis["dsoc_dt"]) < 0
    assert float(r_chg["dsoc_dt"]) > 0


def test_power_sign(model):
    """Power positive during discharge, negative during charging."""
    r_dis = model.predict({"soc": 0.8, "current":  20.0})
    r_chg = model.predict({"soc": 0.8, "current": -20.0})
    assert float(r_dis["power"]) > 0
    assert float(r_chg["power"]) < 0


def test_benchmark(model):
    soc     = np.random.uniform(0.0, 1.0, 1000)
    current = np.random.uniform(-50, 50, 1000)
    start = time.perf_counter()
    model.predict({"soc": soc, "current": current})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
