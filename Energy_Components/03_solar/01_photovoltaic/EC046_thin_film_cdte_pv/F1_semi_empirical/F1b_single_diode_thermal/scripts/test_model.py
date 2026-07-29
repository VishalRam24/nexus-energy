"""EC046 — Thin-Film CdTe PV — F1b Single-Diode + Thermal — Test Suite"""

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
    assert info["ec_id"] == "EC046"
    assert info["fidelity"] == "F1b"


def test_stc_power(model):
    """
    CdTe First Solar FS6 nameplate ~445W; at T_amb=25C NOCT gives T_cell~56C.
    # RATIONALE: NOCT=45C gives T_cell = 25 + 1000*(45-20)/800 = 56.25C at G=1000.
    # At -0.28%/K over 31.25K, Pmp drops ~8.75% from 445W → ~406W.
    # Lower bound 340W allows for model parameter fit variance.
    """
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    p = float(r["p_mp"])
    assert 340.0 < p < 520.0, f"Pmp at T_amb=25C = {p:.1f}W (expected ~406W for FS6)"


def test_cell_temp_above_ambient(model):
    r = model.predict({"irradiance_w_m2": 800.0, "T_ambient_degC": 25.0})
    assert float(r["T_cell_c"]) > 25.0


def test_power_decreases_with_ambient_temperature(model):
    T_ambs = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": T_ambs})
    assert np.all(np.diff(r["p_mp"]) < 0), "Pmp must decrease with T_amb"


def test_power_scales_with_irradiance(model):
    irr = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"irradiance_w_m2": irr, "T_ambient_degC": 25.0})
    assert np.all(np.diff(r["p_mp"]) > 0)


def test_zero_irradiance(model):
    r = model.predict({"irradiance_w_m2": 0.0, "T_ambient_degC": 25.0})
    assert float(r["p_mp"]) < 1.0


def test_cdte_tempco_lower_than_polysi(model):
    """
    CdTe tempco is -0.28%/K; poly-Si is -0.39%/K.
    CdTe should lose less power per degree than poly-Si.
    Quantitative: 25→55C drop must be less than poly-Si reference (11.7%).
    """
    r_25 = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0})
    r_55 = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 55.0})
    drop_frac_cdte = (float(r_25["p_mp"]) - float(r_55["p_mp"])) / float(r_25["p_mp"])
    polysi_ref_drop = 0.0039 * 30  # 30K * 0.39%/K = 11.7%
    assert drop_frac_cdte < polysi_ref_drop, (
        f"CdTe power drop 25→55C = {drop_frac_cdte*100:.2f}% should be < poly-Si {polysi_ref_drop*100:.1f}%  "
        f"(CdTe tempco -0.28%/K is lower than poly-Si -0.39%/K)"
    )


def test_cdte_tempco_magnitude(model):
    """
    Between 25 and 55 degC (30 K), CdTe Pmp drop should be ~8.4% (30*0.28%).
    # RATIONALE: Tolerance ±40% of nominal = 5-12%, broader than Si because
    # the empirical correction operates on an estimated De Soto gamma_desoto;
    # a ±10% uncertainty in the correction factor maps to ±0.04% range on the drop.
    """
    r_25 = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0})
    r_55 = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 55.0})
    drop_frac = (float(r_25["p_mp"]) - float(r_55["p_mp"])) / float(r_25["p_mp"])
    # RATIONALE: 5-12% tolerance around 8.4% nominal accounts for uncertainty
    # in gamma_desoto_approx used in the empirical correction derivation.
    assert 0.05 < drop_frac < 0.12, (
        f"CdTe power drop 25→55C = {drop_frac*100:.2f}% (expected ~8.4%)"
    )


def test_voc_decreases_with_temperature(model):
    T_cells = np.array([10.0, 25.0, 50.0, 70.0])
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": T_cells})
    assert np.all(np.diff(r["v_oc"]) < 0)


def test_efficiency_reasonable(model):
    """
    CdTe efficiency at T_amb=25C, G=1000: T_cell~56C, so efficiency is below nameplate.
    # RATIONALE: At T_cell=56.25C and -0.28%/K, FS6 nameplate 18% drops ~8.75% → ~16.4%.
    # Lower bound 0.13 catches clearly wrong models, upper 0.22 catches over-prediction.
    """
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    eff = float(r["efficiency"])
    assert 0.13 < eff < 0.22, f"Efficiency = {eff:.3f}"


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
