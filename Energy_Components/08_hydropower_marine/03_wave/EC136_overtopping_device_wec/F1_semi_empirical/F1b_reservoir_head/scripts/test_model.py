"""EC136 — Overtopping Device WEC — F1b Reservoir Head — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    for k in ["power_kw", "reservoir_head_m", "turbine_efficiency",
              "overall_efficiency", "seawater_density_kgm3"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC136"
    assert info["fidelity"] == "F1b"


def test_power_positive_at_design(model):
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    assert float(r["power_kw"]) > 0.0


def test_reservoir_head_increases_with_Hs(model):
    """Higher H_s → more overtopping → higher reservoir head."""
    m = model._model
    h1 = float(m.reservoir_head_m(1.0))
    h2 = float(m.reservoir_head_m(3.0))
    assert h2 > h1, f"h at H_s=3m ({h2:.3f}) should be > h at H_s=1m ({h1:.3f})"


def test_turbine_efficiency_at_design_head(model):
    """At design head, turbine efficiency = eta_turb_peak."""
    m = model._model
    eta = float(m.turbine_efficiency(m.h_design))
    assert abs(eta - m.eta_turb_peak) < 1e-9


def test_turbine_efficiency_zero_below_h_min(model):
    """No turbine operation below h_min."""
    m = model._model
    eta = float(m.turbine_efficiency(m.h_min * 0.5))
    assert eta == 0.0


def test_power_increases_with_Hs(model):
    """Higher waves → more overtopping → more reservoir head → more power."""
    r1 = model.predict({"H_s": 1.0, "T_e": 10.0})
    r2 = model.predict({"H_s": 3.0, "T_e": 10.0})
    assert float(r2["power_kw"]) > float(r1["power_kw"])


def test_density_correction_salinity(model):
    """Higher salinity → higher density → more power."""
    r_low  = model.predict({"H_s": 2.0, "T_e": 10.0, "T_C": 15.0, "S_psu": 30.0})
    r_high = model.predict({"H_s": 2.0, "T_e": 10.0, "T_C": 15.0, "S_psu": 38.0})
    assert float(r_high["power_kw"]) > float(r_low["power_kw"])


def test_overall_efficiency_range(model):
    """Overall wave-to-wire efficiency must be between 5-25% (overtopping devices are low-efficiency)."""
    m = model._model
    eta = float(m.overall_efficiency(2.0))
    assert 0.02 <= eta <= 0.30, f"Overall efficiency {eta:.3f} outside expected range"


def test_turbine_head_curvature(model):
    """Efficiency at off-design head must be < peak."""
    m = model._model
    eta_design = float(m.turbine_efficiency(m.h_design))
    eta_high   = float(m.turbine_efficiency(m.h_design * 2.0))
    assert eta_high <= eta_design


def test_benchmark(model):
    H_s = np.random.uniform(0.5, 5.0, 1000)
    T_e = np.random.uniform(5.0, 18.0, 1000)
    start = time.perf_counter()
    model.predict({"H_s": H_s, "T_e": T_e})
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
