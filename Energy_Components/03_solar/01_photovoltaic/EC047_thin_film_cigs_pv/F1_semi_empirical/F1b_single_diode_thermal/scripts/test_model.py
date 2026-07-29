"""EC047 — Thin-Film CIGS PV — F1b Single-Diode + Thermal — Test Suite"""

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
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    for key in ["i_mp", "v_mp", "p_mp", "i_sc", "v_oc", "fill_factor", "efficiency", "T_cell_c"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC047"
    assert info["fidelity"] == "F1b"


def test_stc_power(model):
    """
    CIGS SF170-S ~170W at STC. At T_amb=25C NOCT gives T_cell~56C.
    # RATIONALE: NOCT=45C gives T_cell=56.25C at G=1000, T_amb=25C.
    # At -0.31%/K over 31.25K, Pmp drops ~9.7% from 170W → ~154W.
    # Lower bound 120W accounts for model parameter fit variance.
    """
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    p = float(r["p_mp"])
    assert 120.0 < p < 210.0, f"Pmp at T_amb=25C = {p:.1f}W (expected ~154W for SF170-S)"


def test_cell_temp_above_ambient(model):
    r = model.predict({"irradiance_w_m2": 800.0, "T_ambient_degC": 25.0})
    assert float(r["T_cell_c"]) > 25.0


def test_power_decreases_with_ambient_temperature(model):
    T_ambs = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": T_ambs})
    assert np.all(np.diff(r["p_mp"]) < 0)


def test_power_scales_with_irradiance(model):
    irr = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"irradiance_w_m2": irr, "T_ambient_degC": 25.0})
    assert np.all(np.diff(r["p_mp"]) > 0)


def test_zero_irradiance(model):
    r = model.predict({"irradiance_w_m2": 0.0, "T_ambient_degC": 25.0})
    assert float(r["p_mp"]) < 1.0


def test_cigs_tempco_magnitude(model):
    """
    CIGS tempco ~-0.31%/K.
    25→55C (30K): expected drop ~9.3%; accept 6-14%.
    # RATIONALE: De Soto model captures CIGS thermal behaviour adequately for this
    # bandgap (~1.15 eV). The 6-14% tolerance (±50%) accounts for model fit
    # uncertainty without over-constraining the test.
    """
    r_25 = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0})
    r_55 = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 55.0})
    drop = (float(r_25["p_mp"]) - float(r_55["p_mp"])) / float(r_25["p_mp"])
    # RATIONALE: ±50% on 9.3% nominal gives 4.7-14%, rounded to 6-14% for safety.
    assert 0.06 < drop < 0.14, f"CIGS tempco drop 25→55C = {drop*100:.2f}% (expected ~9.3%)"


def test_cigs_tempco_between_cdte_and_polysi(model):
    """CIGS tempco (-0.31%/K) should be between CdTe (-0.28%/K) and poly-Si (-0.39%/K)."""
    r_25 = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0})
    r_55 = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 55.0})
    drop_cigs = (float(r_25["p_mp"]) - float(r_55["p_mp"])) / float(r_25["p_mp"])
    drop_cdte_ref = 0.0028 * 30   # 8.4%
    drop_polysi_ref = 0.0039 * 30  # 11.7%
    assert drop_cdte_ref < drop_cigs < drop_polysi_ref, (
        f"CIGS drop={drop_cigs*100:.2f}% should be between CdTe={drop_cdte_ref*100:.1f}% "
        f"and poly-Si={drop_polysi_ref*100:.1f}%"
    )


def test_voc_decreases_with_temperature(model):
    T_cells = np.array([10.0, 25.0, 50.0, 70.0])
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": T_cells})
    assert np.all(np.diff(r["v_oc"]) < 0)


def test_efficiency_reasonable(model):
    """
    CIGS efficiency at T_amb=25C, G=1000 W/m2.
    # RATIONALE: NOCT=45C gives T_cell=56.25C at these conditions. CIGS -0.31%/K
    # over 31.25K → ~9.7% power drop from STC. SF170-S nameplate 13.8% (170W/1.232m2/1000)
    # drops to ~12.4%. Lower bound 0.10 catches clearly wrong models while reflecting
    # the thermal operating point (not STC cell temperature).
    """
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    eff = float(r["efficiency"])
    assert 0.10 < eff < 0.18, f"Efficiency = {eff:.3f}"


def test_array_inputs(model):
    irr = np.array([200.0, 500.0, 800.0])
    T_ambs = np.array([10.0, 25.0, 40.0])
    r = model.predict({"irradiance_w_m2": irr, "T_ambient_degC": T_ambs})
    assert r["p_mp"].shape == (3,)


def test_benchmark(model):
    irr = np.random.uniform(100, 1100, 1000)
    T_ambs = np.random.uniform(5, 45, 1000)
    start = time.perf_counter()
    model.predict({"irradiance_w_m2": irr, "T_ambient_degC": T_ambs})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 5.0
