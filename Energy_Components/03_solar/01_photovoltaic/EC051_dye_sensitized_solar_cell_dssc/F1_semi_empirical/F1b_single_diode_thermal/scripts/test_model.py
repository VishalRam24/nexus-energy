"""EC051 — DSSC — F1b Single-Diode + Thermal — Test Suite"""

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
    assert info["ec_id"] == "EC051"
    assert info["fidelity"] == "F1b"


def test_stc_power_range(model):
    """
    DSSC 100cm2 module, PCE ~10% at STC conditions.
    P_stc = 0.10 * 0.01 * 1000 = 1 W.
    # RATIONALE: At T_amb=25C, T_cell = 25 + 1000*(38-20)/800 = 47.5C.
    # At -0.25%/K for 22.5K delta: ~5.6% derating → ~0.94W.
    # Bounds 0.3-2.5W allow for PCE 3-25% range across DSSC variants.
    """
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    p = float(r["p_mp"])
    assert 0.3 < p < 2.5, f"DSSC Pmp = {p:.3f}W (expected ~0.94W for 10% PCE)"


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
    assert float(r["p_mp"]) < 0.01


def test_dssc_voc_range(model):
    """
    DSSC Voc per cell ~0.65-0.75V. Module with 4 cells in series → 2.6-3.0V.
    """
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    voc = float(r["v_oc"])
    assert 1.5 < voc < 4.0, f"DSSC 4-cell Voc = {voc:.3f}V (expected 2.6-3.0V)"


def test_dssc_tempco_lower_than_polysi(model):
    """
    DSSC tempco (-0.25%/K) < poly-Si (-0.39%/K).
    DSSC should lose less power per degree than Si.
    """
    r_25 = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    r_55 = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 55.0})
    drop_frac = (float(r_25["p_mp"]) - float(r_55["p_mp"])) / float(r_25["p_mp"])
    polysi_drop = 0.0039 * 30
    assert drop_frac < polysi_drop, (
        f"DSSC drop = {drop_frac*100:.2f}% should be < poly-Si {polysi_drop*100:.1f}%"
    )


def test_dssc_tempco_magnitude(model):
    """
    DSSC tempco ~-0.25%/K; 30K drop = ~7.5%.
    # RATIONALE: Tolerance ±50% around nominal (3.7-11.2%) because the De Soto
    # gamma_desoto_approx is estimated from parameter fitting, not directly measured.
    # Wider bounds than Si needed due to DSSC parameter uncertainty.
    """
    r_25 = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    r_55 = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 55.0})
    drop = (float(r_25["p_mp"]) - float(r_55["p_mp"])) / float(r_25["p_mp"])
    assert 0.03 < drop < 0.12, f"DSSC drop 25→55C (ambient) = {drop*100:.2f}% (expect ~7.5%)"


def test_efficiency_reasonable(model):
    """DSSC module efficiency: 5-13% at operating conditions."""
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    eff = float(r["efficiency"])
    assert 0.04 < eff < 0.15, f"DSSC efficiency = {eff:.3f}"


def test_fill_factor_range(model):
    """DSSC fill factor 0.50-0.72 (lower than Si due to high Rs from electrolyte)."""
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    ff = float(r["fill_factor"])
    assert 0.35 < ff < 0.80, f"DSSC fill factor = {ff:.3f}"


def test_isc_positive_tempco(model):
    """
    DSSC Isc has positive temperature coefficient (unlike Si where alpha_sc is small positive).
    Isc at 55C should be higher than at 25C.
    """
    r_25 = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    r_55 = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 55.0})
    # Cell temp at T_amb=55 > cell temp at T_amb=25 → I_sc should be higher
    assert float(r_55["i_sc"]) > float(r_25["i_sc"]), \
        "DSSC Isc must increase with temperature due to improved electron injection"


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
