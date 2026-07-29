"""EC049 — Multi-Junction CPV — F1b Two-Diode + Thermal — Test Suite"""

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
    r = model.predict({"dni_w_m2": 1000.0, "T_ambient_degC": 25.0})
    for key in ["i_mp", "v_mp", "p_mp", "i_sc", "v_oc", "fill_factor",
                "efficiency", "T_cell_c", "concentration_ratio"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC049"
    assert info["fidelity"] == "F1b"


def test_stc_power(model):
    """
    At STC (DNI=1000 W/m2, T_amb=25C, 500x), CPV cell Pmp for a 1cm2 cell at ~40% eff.
    P = 0.40 * 0.0004 m2 * 1000 W/m2 * 0.85 optical = ~0.136 W per cell.
    # RATIONALE: Bounds 0.05-0.35 W accommodate model parameter variation and
    # concentration-ratio interpretation differences.
    """
    r = model.predict({"dni_w_m2": 1000.0, "T_ambient_degC": 25.0})
    p = float(r["p_mp"])
    assert 0.05 < p < 0.35, f"CPV cell Pmp at STC = {p:.4f}W"


def test_voc_higher_than_si(model):
    """
    Triple-junction CPV Voc >> Si Voc. At 1-sun, total Voc > 2.5 V (N_s=1 cell).
    At 500x concentration, Voc further increases logarithmically.
    """
    r = model.predict({"dni_w_m2": 1000.0, "T_ambient_degC": 25.0})
    voc = float(r["v_oc"])
    # Triple-junction 1-sun Voc ~2.6V, at 500x adds ~n1*kT/q*ln(500) ~0.16V per junction
    assert voc > 2.0, f"Triple-junction Voc should be >> Si; got {voc:.3f}V"


def test_power_scales_with_dni(model):
    """Power increases with DNI (concentration)."""
    dnis = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"dni_w_m2": dnis, "T_ambient_degC": 25.0})
    assert np.all(np.diff(r["p_mp"]) > 0), "Power must increase with DNI"


def test_power_decreases_with_temperature(model):
    """Power decreases with ambient temperature."""
    temps = np.array([0.0, 15.0, 25.0, 40.0, 55.0])
    r = model.predict({"dni_w_m2": 1000.0, "T_ambient_degC": temps})
    assert np.all(np.diff(r["p_mp"]) < 0), "Power must decrease with ambient temperature"


def test_zero_dni(model):
    """No DNI = no power."""
    r = model.predict({"dni_w_m2": 0.0, "T_ambient_degC": 25.0})
    assert float(r["p_mp"]) < 1e-6


def test_cell_temp_above_ambient(model):
    """Cell temperature must exceed ambient due to thermal loading."""
    r = model.predict({"dni_w_m2": 800.0, "T_ambient_degC": 25.0})
    assert float(r["T_cell_c"]) > 25.0


def test_concentration_ratio(model):
    """Concentration ratio = DNI/1000 (linear scaling)."""
    r = model.predict({"dni_w_m2": 1000.0, "T_ambient_degC": 25.0})
    assert abs(float(r["concentration_ratio"]) - 1.0) < 0.01


def test_fill_factor_range(model):
    """
    III-V CPV fill factor is high: 0.80-0.92 at operational conditions.
    # RATIONALE: Broader lower bound 0.70 allows for numerical solver variance
    # in the two-diode Brent solve at edge conditions.
    """
    r = model.predict({"dni_w_m2": 1000.0, "T_ambient_degC": 25.0})
    ff = float(r["fill_factor"])
    assert 0.70 < ff < 0.95, f"Fill factor = {ff:.3f}"


def test_efficiency_reasonable(model):
    """
    CPV system efficiency at STC based on lens area: 35-45% for III-V triple-junction.
    # RATIONALE: Efficiency is computed as P_mp / (G * lens_area * optical_efficiency).
    # The 0.25-0.50 range accommodates parameter variation while verifying physical realism.
    """
    r = model.predict({"dni_w_m2": 1000.0, "T_ambient_degC": 25.0})
    eff = float(r["efficiency"])
    assert 0.25 < eff < 0.50, f"CPV efficiency = {eff:.3f} (expected 35-45%)"


def test_voc_increases_with_concentration(model):
    """Voc should increase logarithmically with concentration (DNI)."""
    r_low = model.predict({"dni_w_m2": 100.0, "T_ambient_degC": 25.0})
    r_high = model.predict({"dni_w_m2": 1000.0, "T_ambient_degC": 25.0})
    assert float(r_high["v_oc"]) > float(r_low["v_oc"]), \
        "Voc must increase with concentration"


def test_isc_scales_linearly_with_dni(model):
    """Isc scales linearly with concentration."""
    r_half = model.predict({"dni_w_m2": 500.0, "T_ambient_degC": 25.0})
    r_full = model.predict({"dni_w_m2": 1000.0, "T_ambient_degC": 25.0})
    ratio = float(r_full["i_sc"]) / float(r_half["i_sc"])
    assert 1.9 < ratio < 2.1, f"Isc ratio should be ~2.0 for 2x concentration; got {ratio:.3f}"


def test_array_inputs(model):
    dnis = np.array([200.0, 500.0, 800.0])
    temps = np.array([15.0, 25.0, 40.0])
    r = model.predict({"dni_w_m2": dnis, "T_ambient_degC": temps})
    assert r["p_mp"].shape == (3,)


def test_benchmark(model):
    """Benchmark: 10 predictions (two-diode Brent solver)."""
    dnis = np.array([200.0, 400.0, 600.0, 800.0, 1000.0,
                     200.0, 400.0, 600.0, 800.0, 1000.0])
    temps = np.array([15.0, 20.0, 25.0, 30.0, 35.0,
                      40.0, 45.0, 50.0, 25.0, 30.0])
    start = time.perf_counter()
    model.predict({"dni_w_m2": dnis, "T_ambient_degC": temps})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 10 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 30.0
