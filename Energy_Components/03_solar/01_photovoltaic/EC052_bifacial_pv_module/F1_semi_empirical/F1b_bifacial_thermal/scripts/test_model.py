"""EC052 — Bifacial PV Module — F1b Bifacial + Thermal — Test Suite"""

import sys, time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_returns_dict(model):
    r = model.predict({"irradiance_front_w_m2": 1000.0, "T_ambient_degC": 25.0})
    for key in ["i_mp", "v_mp", "p_mp", "i_sc", "v_oc", "fill_factor", "efficiency",
                "G_eff_w_m2", "G_rear_w_m2", "T_cell_front_c", "T_cell_rear_c", "T_cell_eff_c"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC052"
    assert info["fidelity"] == "F1b"


def test_stc_front_only(model):
    """At STC front-only (albedo=0), bifacial module should produce ~nameplate 400W."""
    r = model.predict({"irradiance_front_w_m2": 1000.0, "T_ambient_degC": 25.0, "albedo": 0.0})
    p = float(r["p_mp"])
    assert 300.0 < p < 500.0, f"Pmp at STC front-only = {p:.1f}W"


def test_bifacial_gain_positive(model):
    """Adding rear irradiance (albedo > 0) must increase power output."""
    r_no_rear = model.predict({"irradiance_front_w_m2": 1000.0, "T_ambient_degC": 25.0, "albedo": 0.0})
    r_with_rear = model.predict({"irradiance_front_w_m2": 1000.0, "T_ambient_degC": 25.0, "albedo": 0.3})
    assert float(r_with_rear["p_mp"]) > float(r_no_rear["p_mp"]), \
        "Bifacial with albedo=0.3 must produce more power than front-only"


def test_rear_irradiance_nonzero(model):
    """Rear irradiance must be nonzero when albedo > 0."""
    r = model.predict({"irradiance_front_w_m2": 1000.0, "T_ambient_degC": 25.0, "albedo": 0.2})
    assert float(r["G_rear_w_m2"]) > 0.0


def test_rear_cell_cooler_than_front(model):
    """Rear cell temperature must be lower than front cell temperature under illumination."""
    r = model.predict({"irradiance_front_w_m2": 800.0, "T_ambient_degC": 25.0, "albedo": 0.2})
    T_front = float(r["T_cell_front_c"])
    T_rear = float(r["T_cell_rear_c"])
    assert T_rear < T_front, (
        f"T_rear={T_rear:.2f}C must be < T_front={T_front:.2f}C (rear receives less irradiance)"
    )


def test_eff_cell_temp_between_front_and_rear(model):
    """Effective cell temperature must be between front and rear temperatures."""
    r = model.predict({"irradiance_front_w_m2": 800.0, "T_ambient_degC": 25.0, "albedo": 0.2})
    T_f = float(r["T_cell_front_c"])
    T_r = float(r["T_cell_rear_c"])
    T_e = float(r["T_cell_eff_c"])
    assert T_r <= T_e <= T_f, (
        f"T_eff={T_e:.2f}C must be in [{T_r:.2f}, {T_f:.2f}]"
    )


def test_power_decreases_with_ambient_temperature(model):
    T_ambs = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
    r = model.predict({"irradiance_front_w_m2": 1000.0, "T_ambient_degC": T_ambs, "albedo": 0.2})
    assert np.all(np.diff(r["p_mp"]) < 0), "Pmp must decrease with T_amb"


def test_power_scales_with_irradiance(model):
    irr = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"irradiance_front_w_m2": irr, "T_ambient_degC": 25.0, "albedo": 0.2})
    assert np.all(np.diff(r["p_mp"]) > 0)


def test_zero_irradiance(model):
    r = model.predict({"irradiance_front_w_m2": 0.0, "T_ambient_degC": 25.0})
    assert float(r["p_mp"]) < 1.0


def test_bifacial_gain_increases_with_albedo(model):
    """Higher albedo should give higher bifacial gain."""
    p_vals = [float(model.predict({"irradiance_front_w_m2": 1000.0, "T_ambient_degC": 25.0,
                                    "albedo": a})["p_mp"]) for a in [0.1, 0.2, 0.4, 0.6]]
    assert all(p_vals[i] < p_vals[i + 1] for i in range(len(p_vals) - 1)), \
        "Power must increase with albedo"


def test_direct_rear_override(model):
    """Direct G_rear input must match equivalent albedo result."""
    r_alb = model.predict({"irradiance_front_w_m2": 1000.0, "T_ambient_degC": 25.0, "albedo": 0.2})
    G_rear_expected = float(r_alb["G_rear_w_m2"])
    r_direct = model.predict({"irradiance_front_w_m2": 1000.0, "T_ambient_degC": 25.0,
                               "irradiance_rear_w_m2": G_rear_expected})
    assert abs(float(r_direct["p_mp"]) - float(r_alb["p_mp"])) < 1.0, \
        "Direct G_rear and albedo-computed G_rear must give same power"


def test_efficiency_reasonable(model):
    r = model.predict({"irradiance_front_w_m2": 1000.0, "T_ambient_degC": 25.0, "albedo": 0.0})
    eff = float(r["efficiency"])
    assert 0.15 < eff < 0.25, f"Efficiency = {eff:.3f}"


def test_benchmark(model):
    irr = np.random.uniform(100, 1100, 500)
    T_ambs = np.random.uniform(5, 45, 500)
    start = time.perf_counter()
    model.predict({"irradiance_front_w_m2": irr, "T_ambient_degC": T_ambs, "albedo": 0.2})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 500 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 10.0
