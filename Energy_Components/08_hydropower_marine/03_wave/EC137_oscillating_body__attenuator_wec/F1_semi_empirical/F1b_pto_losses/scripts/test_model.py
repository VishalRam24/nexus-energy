"""EC137 — Attenuator WEC — F1b PTO Losses — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"H_s": 3.0, "T_e": 10.0})
    for k in ["power_kw", "cwr", "pto_efficiency", "overall_efficiency", "seawater_density_kgm3"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC137"
    assert info["fidelity"] == "F1b"


def test_power_positive_at_design(model):
    r = model.predict({"H_s": 3.0, "T_e": 10.0})
    assert float(r["power_kw"]) > 0.0


def test_cwr_peak_at_design(model):
    """CWR at design (H_s_design, T_e_design) must equal cwr_design."""
    m = model._model
    cwr = float(m.capture_width_ratio(m.H_s_design, m.T_e_design))
    assert abs(cwr - m.cwr_design) < 0.01


def test_cwr_decreases_off_period(model):
    """CWR drops away from design T_e."""
    m = model._model
    cwr_on  = float(m.capture_width_ratio(m.H_s_design, m.T_e_design))
    cwr_off = float(m.capture_width_ratio(m.H_s_design, m.T_e_design + m.sigma_T * 2))
    assert cwr_off < cwr_on


def test_pto_efficiency_at_design(model):
    """PTO efficiency at design H_s must equal eta_pto_design."""
    m = model._model
    eta = float(m.pto_efficiency(m.H_s_design))
    assert abs(eta - m.eta_pto_design) < 1e-9


def test_directionality_reduces_power(model):
    m = model._model
    P_with    = float(m.power_kw(3.0, 10.0, apply_directionality=True))
    P_without = float(m.power_kw(3.0, 10.0, apply_directionality=False))
    assert P_with < P_without, "Directionality must reduce output"


def test_salinity_increases_power(model):
    r_low  = model.predict({"H_s": 3.0, "T_e": 10.0, "T_C": 15.0, "S_psu": 30.0})
    r_high = model.predict({"H_s": 3.0, "T_e": 10.0, "T_C": 15.0, "S_psu": 38.0})
    assert float(r_high["power_kw"]) > float(r_low["power_kw"])


def test_power_scales_with_Hs(model):
    """Power increases monotonically with H_s at constant T_e (within efficiency bounds)."""
    r1 = model.predict({"H_s": 1.0, "T_e": 10.0})
    r2 = model.predict({"H_s": 4.0, "T_e": 10.0})
    assert float(r2["power_kw"]) > float(r1["power_kw"])


def test_per_joint_power(model):
    """Power per joint = total / n_joints."""
    m = model._model
    P_total = float(m.power_kw(3.0, 10.0))
    P_joint = float(m.power_per_joint_kw(3.0, 10.0))
    assert abs(P_joint - P_total / m.n_joints) < 1e-6


def test_pto_efficiency_in_bounds(model):
    """PTO efficiency must stay within physical range [0.50, 0.92]."""
    m = model._model
    for H_s in [0.5, 1.0, 3.0, 5.0, 8.0]:
        eta = float(m.pto_efficiency(H_s))
        assert 0.45 <= eta <= 0.95, f"eta_pto={eta:.3f} at H_s={H_s}m out of range"


def test_benchmark(model):
    H_s = np.random.uniform(0.5, 7.0, 1000)
    T_e = np.random.uniform(5.0, 18.0, 1000)
    start = time.perf_counter()
    model.predict({"H_s": H_s, "T_e": T_e})
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
