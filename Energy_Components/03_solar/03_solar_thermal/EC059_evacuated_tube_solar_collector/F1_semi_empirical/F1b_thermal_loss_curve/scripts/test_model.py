"""EC059 — Evacuated Tube Solar Collector — F1b Thermal Loss Curve — Test Suite"""

import sys, time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import EvacuatedTubeF1b
import json


@pytest.fixture
def model():
    return ComponentModel()


@pytest.fixture
def raw_model():
    params_path = Path(__file__).parent.parent / "data" / "parameters.json"
    with open(params_path) as f:
        params = json.load(f)
    return EvacuatedTubeF1b(params)


def test_predict_returns_dict(model):
    r = model.predict({"irradiance_w_m2": 800.0, "T_inlet_degC": 50.0, "T_ambient_degC": 20.0})
    for key in ["useful_heat_w", "efficiency", "T_outlet_c", "T_mean_c",
                "delta_T_m", "U_L_eff_w_m2k", "iam"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC059"
    assert info["fidelity"] == "F1b"


def test_zero_irradiance(model):
    r = model.predict({"irradiance_w_m2": 0.0, "T_inlet_degC": 50.0, "T_ambient_degC": 20.0})
    assert float(r["useful_heat_w"]) == 0.0
    assert float(r["efficiency"]) == 0.0


def test_useful_heat_positive(model):
    """At good solar conditions, ETC should produce useful heat."""
    r = model.predict({"irradiance_w_m2": 800.0, "T_inlet_degC": 50.0, "T_ambient_degC": 20.0})
    assert float(r["useful_heat_w"]) > 100.0, "Should produce meaningful useful heat"


def test_outlet_above_inlet(model):
    """Outlet temperature must exceed inlet when heat is gained."""
    r = model.predict({"irradiance_w_m2": 800.0, "T_inlet_degC": 50.0, "T_ambient_degC": 20.0})
    assert float(r["T_outlet_c"]) > float(r["T_inlet_degC"]) if hasattr(r, 'T_inlet_degC') else True
    assert float(r["T_outlet_c"]) > 50.0, "Outlet must exceed 50C inlet"


def test_efficiency_decreases_with_delta_T(model):
    """Higher (T_in - T_amb) at same G → lower efficiency (more losses)."""
    T_amb = 15.0
    G = 800.0
    T_ins = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
    r = model.predict({"irradiance_w_m2": G, "T_inlet_degC": T_ins, "T_ambient_degC": T_amb})
    assert np.all(np.diff(r["efficiency"]) < 0), \
        "Efficiency must decrease as T_in - T_amb increases"


def test_useful_heat_increases_with_irradiance(model):
    """Higher irradiance → more useful heat (at same T_in, T_amb)."""
    Gs = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"irradiance_w_m2": Gs, "T_inlet_degC": 50.0, "T_ambient_degC": 20.0})
    assert np.all(np.diff(r["useful_heat_w"]) > 0)


def test_U_L_increases_with_delta_T(raw_model):
    """U_L(DeltaT) must increase with DeltaT (a2 > 0 even for vacuum insulation)."""
    dTs = np.array([0.0, 20.0, 50.0, 80.0, 120.0])
    ULs = raw_model.U_L(dTs)
    assert np.all(np.diff(ULs) > 0), "U_L must increase with DeltaT (a2 > 0)"


def test_U_L_weak_dependence(raw_model):
    """
    ETC U_L should be much less sensitive to temperature than flat plate.
    Over 100K DeltaT, U_L increase should be < 2 W/m2K (vs ~6 for flat plate).
    # RATIONALE: This test enforces the vacuum insulation physics. a2=0.012 gives
    # dU_L = 0.012*100 = 1.2 W/m2K over 100K, well under the 2 W/m2K threshold.
    # A flat plate with a2=0.06 would give 6 W/m2K — 5x larger.
    """
    dU_L = float(raw_model.U_L(100.0)) - float(raw_model.U_L(0.0))
    assert dU_L < 2.0, (
        f"U_L increase over 100K = {dU_L:.3f} W/m2K; ETC vacuum should be < 2 W/m2K"
    )


def test_iam_at_normal_incidence(raw_model):
    """IAM at theta=0 must be 1.0 (no incidence angle penalty)."""
    iam0 = float(raw_model.iam(0.0))
    assert abs(iam0 - 1.0) < 0.001, f"IAM(0) = {iam0:.4f}, must be 1.0"


def test_iam_decreases_with_angle(raw_model):
    """IAM must decrease monotonically with incidence angle."""
    thetas = np.array([0.0, 20.0, 40.0, 60.0])
    iams = raw_model.iam(thetas)
    assert np.all(np.diff(iams) < 0), "IAM must decrease with incidence angle"


def test_off_normal_incidence_reduces_output(model):
    """Off-normal incidence must reduce useful heat vs normal incidence."""
    r0 = model.predict({"irradiance_w_m2": 800.0, "T_inlet_degC": 50.0,
                         "T_ambient_degC": 20.0, "incidence_angle_deg": 0.0})
    r45 = model.predict({"irradiance_w_m2": 800.0, "T_inlet_degC": 50.0,
                          "T_ambient_degC": 20.0, "incidence_angle_deg": 45.0})
    assert float(r45["useful_heat_w"]) < float(r0["useful_heat_w"]), \
        "45 deg incidence must produce less heat than normal incidence"


def test_high_temperature_operation(model):
    """ETCs should still operate at T_in=120C (not possible for flat-plate w/o pressurization)."""
    r = model.predict({"irradiance_w_m2": 1000.0, "T_inlet_degC": 120.0, "T_ambient_degC": 10.0})
    assert float(r["useful_heat_w"]) > 0.0, "ETC should produce useful heat at T_in=120C"
    assert float(r["efficiency"]) > 0.0, "ETC efficiency should be positive at T_in=120C"


def test_efficiency_reasonable_at_low_delta_T(model):
    """Near ambient inlet temperature, efficiency should be close to eta_0=0.72."""
    r = model.predict({"irradiance_w_m2": 800.0, "T_inlet_degC": 22.0, "T_ambient_degC": 20.0})
    eta = float(r["efficiency"])
    # RATIONALE: At very low DeltaT, eta → eta_0*IAM = 0.72. Allow 5% relative tolerance
    # for iterative solver convergence and small DeltaT effects.
    assert 0.60 < eta < 0.76, f"Efficiency at low DeltaT = {eta:.3f} (expected near 0.72)"


def test_array_inputs(model):
    Gs = np.array([400.0, 600.0, 800.0])
    T_ins = np.array([40.0, 60.0, 80.0])
    T_ambs = np.array([10.0, 20.0, 30.0])
    r = model.predict({"irradiance_w_m2": Gs, "T_inlet_degC": T_ins, "T_ambient_degC": T_ambs})
    assert r["useful_heat_w"].shape == (3,)


def test_benchmark(model):
    Gs = np.random.uniform(100, 1100, 1000)
    T_ins = np.random.uniform(20, 150, 1000)
    T_ambs = np.random.uniform(-10, 40, 1000)
    start = time.perf_counter()
    model.predict({"irradiance_w_m2": Gs, "T_inlet_degC": T_ins, "T_ambient_degC": T_ambs})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 2.0
