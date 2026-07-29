"""EC134 — OWC — F1b PTO Losses — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    for k in ["power_kw", "turbine_efficiency", "overall_efficiency", "seawater_density_kgm3"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC134"
    assert info["fidelity"] == "F1b"


def test_power_positive(model):
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    assert float(r["power_kw"]) > 0.0


def test_turbine_efficiency_peak_at_design(model):
    """At design T_e, turbine efficiency should equal eta_turb_peak."""
    m = model._model
    eta = float(m.turbine_efficiency(m.T_e_design))
    assert abs(eta - m.eta_turb_peak) < 1e-9


def test_turbine_efficiency_degrades_off_design(model):
    """Efficiency at T_e far from design must be < eta_peak."""
    m = model._model
    eta_off = float(m.turbine_efficiency(m.T_e_design + m.bandwidth * 2))
    assert eta_off < m.eta_turb_peak


def test_turbine_efficiency_in_bounds(model):
    for T_e in [5.0, 8.0, 10.0, 15.0, 20.0]:
        m = model._model
        eta = float(m.turbine_efficiency(T_e))
        assert m.eta_turb_min <= eta <= m.eta_turb_peak


def test_power_scales_with_Hs_squared(model):
    """Wave power scales as H_s^2 at constant T_e."""
    r1 = model.predict({"H_s": 2.0, "T_e": 10.0})
    r2 = model.predict({"H_s": 4.0, "T_e": 10.0})
    ratio = float(r2["power_kw"]) / float(r1["power_kw"])
    assert abs(ratio - 4.0) < 0.3, f"Power ratio at 2x H_s: {ratio:.2f}, expected ~4"


def test_directionality_factor_reduces_power(model):
    """Power with directionality must be < without."""
    m = model._model
    P_with    = float(m.power_kw(2.0, 10.0, apply_directionality=True))
    P_without = float(m.power_kw(2.0, 10.0, apply_directionality=False))
    assert P_with < P_without


def test_salinity_increases_power(model):
    r_low  = model.predict({"H_s": 2.0, "T_e": 10.0, "T_C": 15.0, "S_psu": 30.0})
    r_high = model.predict({"H_s": 2.0, "T_e": 10.0, "T_C": 15.0, "S_psu": 38.0})
    assert float(r_high["power_kw"]) > float(r_low["power_kw"])


def test_temperature_decreases_density(model):
    m = model._model
    rho_cold = float(m.seawater_density(5.0, 35.0))
    rho_warm = float(m.seawater_density(25.0, 35.0))
    assert rho_warm < rho_cold


def test_f1b_lower_than_f1a_at_design(model):
    """F1b output (with directionality) must be <= F1a at same conditions."""
    m = model._model
    P_f1b = float(m.power_kw(3.0, m.T_e_design, apply_directionality=True))
    P_f1a = float(m.power_kw(3.0, m.T_e_design, apply_directionality=False))
    assert P_f1b <= P_f1a


def test_benchmark(model):
    H_s = np.random.uniform(0.5, 6.0, 1000)
    T_e = np.random.uniform(5.0, 18.0, 1000)
    start = time.perf_counter()
    model.predict({"H_s": H_s, "T_e": T_e})
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
