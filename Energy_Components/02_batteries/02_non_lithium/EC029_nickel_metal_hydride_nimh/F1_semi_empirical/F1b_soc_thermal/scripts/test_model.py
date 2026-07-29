"""
EC029 -- NiMH Battery -- F1b SOC-Thermal -- Test Suite

Physics sanity checks. Key difference from Li-ion: dOCV/dT > 0 for NiMH.
Run: python -m pytest test_model.py -v
"""

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


def test_predict_returns_all_keys(model):
    result = model.predict({"soc": 0.5, "current": 1.0, "temperature": 298.15})
    for key in ["terminal_voltage", "power", "heat_generation",
                "effective_capacity", "internal_resistance"]:
        assert key in result


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC029"
    assert info["fidelity"] == "F1b"


def test_resistance_increases_at_low_temperature(model):
    r_cold = model.predict({"soc": 0.5, "current": 0, "temperature": 253.15})["internal_resistance"]
    r_ref = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance"]
    r_hot = model.predict({"soc": 0.5, "current": 0, "temperature": 333.15})["internal_resistance"]
    assert float(r_cold) > float(r_ref), "R at -20C must be > R at 25C"
    assert float(r_ref) > float(r_hot), "R at 25C must be > R at 60C"


def test_resistance_at_reference_equals_R_ref(model):
    r = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["internal_resistance"]
    assert abs(float(r) - 0.030) < 1e-6


def test_voltage_lower_at_cold_high_current(model):
    v_cold = model.predict({"soc": 0.5, "current": 5.0, "temperature": 253.15})["terminal_voltage"]
    v_hot = model.predict({"soc": 0.5, "current": 5.0, "temperature": 333.15})["terminal_voltage"]
    assert float(v_cold) < float(v_hot), "Voltage at -20C must be lower than at 60C (high current)"


def test_heat_generation_positive_discharge(model):
    """NiMH discharge: both Joule and reversible terms are positive (dOCV/dT>0, I>0)."""
    q = model.predict({"soc": 0.5, "current": 5.0, "temperature": 298.15})["heat_generation"]
    assert float(q) > 0, "Heat generation must be positive during discharge"


def test_heat_generation_increases_with_current(model):
    q1 = model.predict({"soc": 0.5, "current": 1.0, "temperature": 298.15})["heat_generation"]
    q5 = model.predict({"soc": 0.5, "current": 5.0, "temperature": 298.15})["heat_generation"]
    q10 = model.predict({"soc": 0.5, "current": 10.0, "temperature": 298.15})["heat_generation"]
    assert float(q5) > float(q1)
    assert float(q10) > float(q5)


def test_nimh_positive_docv_dt(model):
    """NiMH has positive dOCV/dT, unlike Li-ion. Verify by checking that at low current,
    OCV increases with temperature (sign of dOCV/dT)."""
    # dOCV/dT > 0 means OCV rises slightly with temperature
    # We can verify this: at zero current V = OCV, and OCV uses fixed polynomial (no T-dep in OCV itself),
    # but the reversible heat term I*T*dOCV/dT with I>0 and dOCV/dT>0 should be positive.
    q_small_discharge = float(model.predict({"soc": 0.5, "current": 0.1, "temperature": 298.15})["heat_generation"])
    # For NiMH: I^2*R ~ 0.0003 W, I*T*dOCV/dT ~ 0.1*298.15*0.0005 ~ 0.0149 W -> total > 0
    assert q_small_discharge > 0, "NiMH heat generation must be positive even at small discharge"


def test_capacity_increases_with_temperature(model):
    """NiMH capacity is notably temperature-sensitive (alpha_c=0.006 > NMC 0.005)."""
    c_cold = model.predict({"soc": 0.5, "current": 0, "temperature": 253.15})["effective_capacity"]
    c_ref = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["effective_capacity"]
    c_hot = model.predict({"soc": 0.5, "current": 0, "temperature": 333.15})["effective_capacity"]
    assert float(c_cold) < float(c_ref) < float(c_hot)


def test_capacity_at_reference(model):
    c = model.predict({"soc": 0.5, "current": 0, "temperature": 298.15})["effective_capacity"]
    assert abs(float(c) - 2.0) < 1e-6


def test_zero_current_voltage_equals_ocv(model):
    result = model.predict({"soc": 0.5, "current": 0.0, "temperature": 298.15})
    assert abs(float(result["terminal_voltage"]) - float(result["ocv"])) < 1e-10


def test_soc_bounds(model):
    for soc in [0.0, 1.0]:
        result = model.predict({"soc": soc, "current": 1.0, "temperature": 298.15})
        assert np.isfinite(float(result["terminal_voltage"]))


def test_temperature_extremes(model):
    for T in [253.15, 333.15]:
        result = model.predict({"soc": 0.5, "current": 1.0, "temperature": T})
        assert np.isfinite(float(result["terminal_voltage"]))
        assert float(result["internal_resistance"]) > 0


def test_array_inputs(model):
    socs = np.array([0.2, 0.5, 0.8])
    currents = np.array([1.0, 2.0, 3.0])
    temps = np.array([263.15, 298.15, 313.15])
    result = model.predict({"soc": socs, "current": currents, "temperature": temps})
    assert result["terminal_voltage"].shape == (3,)


def test_benchmark_1000_predictions(model):
    socs = np.random.uniform(0.0, 1.0, 1000)
    currents = np.random.uniform(-5.0, 5.0, 1000)
    temps = np.random.uniform(253.15, 333.15, 1000)
    start = time.perf_counter()
    model.predict({"soc": socs, "current": currents, "temperature": temps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
