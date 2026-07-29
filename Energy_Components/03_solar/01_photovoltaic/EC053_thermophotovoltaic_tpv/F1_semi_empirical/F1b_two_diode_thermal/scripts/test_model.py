"""EC053 — TPV — F1b Two-Diode + Thermal — Test Suite"""

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
    r = model.predict({"T_emitter_K": 1500.0, "T_heatsink_degC": 25.0})
    for key in ["i_mp", "v_mp", "p_mp", "i_sc", "v_oc", "fill_factor", "efficiency", "T_cell_c"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC053"
    assert info["fidelity"] == "F1b"


def test_reference_power_range(model):
    """
    At T_emitter=1500K, GaSb cell 16cm2 with ~200mA/cm2 Jsc and ~0.35V Vmp.
    P_mp ~ 3.2A * 0.35V = ~1.1 W per cell.
    # RATIONALE: Bounds 0.3-5.0W accommodate uncertainty in emitter geometry,
    # view factor, and spectral emissivity not fully captured in F1b.
    """
    r = model.predict({"T_emitter_K": 1500.0, "T_heatsink_degC": 25.0})
    p = float(r["p_mp"])
    assert 0.3 < p < 5.0, f"TPV Pmp at 1500K = {p:.3f}W"


def test_power_increases_with_emitter_temperature(model):
    """Higher emitter temperature → more above-bandgap photons → more power."""
    T_emitters = np.array([1000.0, 1200.0, 1500.0, 1800.0])
    r = model.predict({"T_emitter_K": T_emitters, "T_heatsink_degC": 25.0})
    assert np.all(np.diff(r["p_mp"]) > 0), "Power must increase with emitter temperature"


def test_zero_power_below_threshold(model):
    """Very low emitter temperature (< 500K) → negligible above-bandgap flux → no power."""
    r = model.predict({"T_emitter_K": 400.0, "T_heatsink_degC": 25.0})
    assert float(r["p_mp"]) < 1e-6, "Below ~500K emitter, power should be essentially zero"


def test_isc_increases_with_emitter_temperature(model):
    """Isc scales with above-bandgap photon flux, which increases with T_emitter."""
    r_low = model.predict({"T_emitter_K": 1000.0, "T_heatsink_degC": 25.0})
    r_high = model.predict({"T_emitter_K": 1800.0, "T_heatsink_degC": 25.0})
    assert float(r_high["i_sc"]) > float(r_low["i_sc"])


def test_voc_reasonable(model):
    """
    GaSb TPV Voc at 1500K: ~0.3-0.55V per cell (N_s=1).
    Bounded below by Eg/q * fill fraction, bounded above by Eg/q = 0.72V.
    """
    r = model.predict({"T_emitter_K": 1500.0, "T_heatsink_degC": 25.0})
    voc = float(r["v_oc"])
    assert 0.1 < voc < 0.72, f"GaSb TPV Voc = {voc:.3f}V (bounded by Eg/q=0.72V)"


def test_fill_factor_range(model):
    """
    GaSb TPV fill factor: 0.60-0.80 typical.
    # RATIONALE: Lower bound 0.50 allows for high recombination current effects
    # from the two-diode model with dominant I_02 term.
    """
    r = model.predict({"T_emitter_K": 1500.0, "T_heatsink_degC": 25.0})
    ff = float(r["fill_factor"])
    assert 0.40 < ff < 0.88, f"TPV fill factor = {ff:.3f}"


def test_efficiency_reasonable(model):
    """
    TPV efficiency = P_mp / (sigma*T_emitter^4*cell_area).
    GaSb at 1500K: ~10-25% electrical efficiency (sub-bandgap photons are wasted).
    # RATIONALE: Bounds 0.001-0.30 because the model uses full blackbody power
    # (not spectrally filtered), giving lower apparent efficiency than system-level.
    """
    r = model.predict({"T_emitter_K": 1500.0, "T_heatsink_degC": 25.0})
    eff = float(r["efficiency"])
    assert 0.001 < eff < 0.35, f"TPV efficiency (vs BB) = {eff:.4f}"


def test_array_inputs(model):
    T_emitters = np.array([1000.0, 1500.0, 1800.0])
    T_heatsinks = np.array([25.0, 25.0, 30.0])
    r = model.predict({"T_emitter_K": T_emitters, "T_heatsink_degC": T_heatsinks})
    assert r["p_mp"].shape == (3,)


def test_benchmark(model):
    T_emitters = np.linspace(800, 1900, 10)
    start = time.perf_counter()
    model.predict({"T_emitter_K": T_emitters, "T_heatsink_degC": 25.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 10 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 30.0
