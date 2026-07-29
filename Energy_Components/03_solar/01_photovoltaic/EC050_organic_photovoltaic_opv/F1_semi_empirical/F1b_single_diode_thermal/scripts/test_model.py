"""EC050 — Organic PV (OPV) — F1b Single-Diode + Thermal — Test Suite"""

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
    assert info["ec_id"] == "EC050"
    assert info["fidelity"] == "F1b"


def test_stc_power_range(model):
    """
    OPV module 800cm2, PCE ~12% → P_stc ~96W. Thermally derated at T_cell > 25C.
    # RATIONALE: At T_amb=25C, T_cell = 25 + 1000*(40-20)/800 = 50C, derate ~5.25%.
    # Expected Pmp ~91W. Bounds 40-150W accommodate parameter variance in NFA OPV.
    """
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    p = float(r["p_mp"])
    assert 40.0 < p < 150.0, f"OPV Pmp = {p:.1f}W (expected ~91W)"


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


def test_opv_tempco_lower_than_polysi(model):
    """
    OPV tempco (-0.21%/K) < poly-Si (-0.39%/K).
    OPV should lose less power per degree C than poly-Si reference.
    """
    r_25 = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    r_55 = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 55.0})
    drop_frac = (float(r_25["p_mp"]) - float(r_55["p_mp"])) / float(r_25["p_mp"])
    polysi_drop = 0.0039 * 30 * (1 + 25 * (40 - 20) / 800 / 30)  # approximate cell T diff
    # Simplified: OPV drop (30K * 0.21%) = 6.3% < poly-Si (30K * 0.39%) = 11.7%
    # Due to NOCT difference, cell T difference is actually (T_cell_55 - T_cell_25) ≈ 30K
    assert drop_frac < 0.117, (
        f"OPV power drop = {drop_frac*100:.2f}% should be < poly-Si 11.7% (30K * 0.39%/K)"
    )


def test_opv_tempco_magnitude(model):
    """
    OPV tempco ~-0.21%/K. Over 30K (T_amb from 25 to 55C), drop ~6.3%.
    # RATIONALE: Tolerance ±50% around nominal because the De Soto gamma_desoto_approx
    # is estimated, not directly measured. Bounds 3-12% allow for this uncertainty
    # while ensuring the model does not produce Si-like or zero tempco behavior.
    """
    r_25 = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    r_55 = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 55.0})
    drop = (float(r_25["p_mp"]) - float(r_55["p_mp"])) / float(r_25["p_mp"])
    assert 0.03 < drop < 0.12, f"OPV drop 25→55C (ambient) = {drop*100:.2f}% (expect ~6.3%)"


def test_efficiency_reasonable(model):
    """
    OPV module efficiency at STC-like conditions: 8-16% (NFA devices up to ~18%).
    # RATIONALE: T_cell ~50C at T_amb=25C; effective PCE slightly reduced from nameplate.
    """
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    eff = float(r["efficiency"])
    assert 0.06 < eff < 0.20, f"OPV efficiency = {eff:.3f}"


def test_fill_factor_range(model):
    """OPV fill factor 0.55-0.80 (lower than Si due to high Rs and recombination)."""
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    ff = float(r["fill_factor"])
    assert 0.40 < ff < 0.85, f"OPV fill factor = {ff:.3f}"


def test_voc_decreases_with_temperature(model):
    T_cells = np.array([10.0, 25.0, 50.0, 70.0])
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": T_cells})
    assert np.all(np.diff(r["v_oc"]) < 0)


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
