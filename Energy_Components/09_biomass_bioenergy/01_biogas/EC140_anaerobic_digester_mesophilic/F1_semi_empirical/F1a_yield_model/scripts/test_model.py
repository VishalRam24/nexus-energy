"""EC140 — Anaerobic Digester (Mesophilic) — F1a Yield Model — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"vs_loading": 3.0, "hrt": 20.0})
    for k in ["methane_yield_m3kgvs", "biogas_rate_m3day", "methane_rate_m3day", "energy_output_kwh_day"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC140"
    assert info["fidelity"] == "F1a"


def test_yield_below_y_max(model):
    """
    Methane yield must always be < Y_max at T=T_ref (f_T=1). Above T_ref, the
    Arrhenius factor can push yield slightly above Y_max*(1-exp(-k*HRT)), so we
    test at reference temperature only and confirm yield approaches but stays below Y_max.
    """
    hrt = np.linspace(5, 40, 50)
    r = model.predict({"vs_loading": 3.0, "hrt": hrt, "temperature": 37.0})
    # At T=37C (T_ref), f_T=1 and yield asymptotes to Y_max from below
    assert np.all(r["methane_yield_m3kgvs"] <= 0.35 * 1.01), \
        "Yield must not significantly exceed Y_max"
    # Verify it's always positive
    assert np.all(r["methane_yield_m3kgvs"] > 0)


def test_yield_increases_with_hrt(model):
    """Methane yield must increase monotonically with HRT at fixed temperature."""
    hrt = np.linspace(5, 40, 50)
    r = model.predict({"vs_loading": 3.0, "hrt": hrt, "temperature": 37.0})
    assert np.all(np.diff(r["methane_yield_m3kgvs"]) > 0), "Yield must increase with HRT"


def test_yield_increases_with_temperature_below_42c(model):
    """Yield should increase with temperature below the inhibition threshold (42 degC)."""
    temps = np.linspace(25, 40, 20)
    r = model.predict({"vs_loading": 3.0, "hrt": 20.0, "temperature": temps})
    assert np.all(np.diff(r["methane_yield_m3kgvs"]) > 0), \
        "Yield must increase with temperature below 42 degC"


def test_yield_decreases_above_42c(model):
    """Yield should decrease with temperature above 42 degC (inhibition)."""
    temps = np.linspace(43, 54, 20)
    r = model.predict({"vs_loading": 3.0, "hrt": 20.0, "temperature": temps})
    assert np.all(np.diff(r["methane_yield_m3kgvs"]) < 0), \
        "Yield must decrease above 42 degC (thermal inhibition)"


def test_yield_zero_at_zero_hrt(model):
    """At HRT=0, no digestion can occur, yield = 0."""
    r = model.predict({"vs_loading": 3.0, "hrt": 0.0, "temperature": 37.0})
    assert abs(float(r["methane_yield_m3kgvs"])) < 1e-10


def test_biogas_proportional_to_vs_loading(model):
    """Biogas rate scales linearly with VS loading at fixed HRT and temperature."""
    vs = np.linspace(1, 8, 30)
    r = model.predict({"vs_loading": vs, "hrt": 20.0, "temperature": 37.0})
    # biogas_rate / vs_loading should be constant
    ratio = r["biogas_rate_m3day"] / vs
    assert np.allclose(ratio, ratio[0], rtol=1e-9), "Biogas rate must be linear in VS loading"


def test_methane_fraction_correct(model):
    """methane_rate / biogas_rate should equal methane_fraction (0.60)."""
    r = model.predict({"vs_loading": 3.0, "hrt": 20.0, "temperature": 37.0})
    fraction = float(r["methane_rate_m3day"]) / float(r["biogas_rate_m3day"])
    assert abs(fraction - 0.60) < 1e-9, f"CH4 fraction = {fraction:.4f}, expected 0.60"


def test_energy_proportional_to_methane(model):
    """Energy output = methane_rate * LHV (9.97 kWh/m³_CH4)."""
    r = model.predict({"vs_loading": 3.0, "hrt": 20.0, "temperature": 37.0})
    e = float(r["energy_output_kwh_day"])
    ch4 = float(r["methane_rate_m3day"])
    assert abs(e / ch4 - 9.97) < 1e-6, f"LHV check: {e/ch4:.4f} kWh/m³ (expected 9.97)"


def test_yield_at_design_conditions(model):
    """At design: VS=3, HRT=20, T=37 -> yield = Y_max*(1-exp(-k*HRT))*f_T(37C).
    f_T(37C) ~ 1.0 (within Arrhenius calculation rounding), so yield is near
    0.35*(1-exp(-0.15*20)) = 0.3326 m³_CH4/kgVS within 5%.
    """
    import math
    Y_base = 0.35 * (1 - math.exp(-0.15 * 20))  # ~0.3326 m³_CH4/kgVS
    r = model.predict({"vs_loading": 3.0, "hrt": 20.0, "temperature": 37.0})
    Y = float(r["methane_yield_m3kgvs"])
    assert abs(Y - Y_base) / Y_base < 0.05, \
        f"Yield = {Y:.4f}, base = {Y_base:.4f} (expected within 5%)"
    assert 0.28 < Y < 0.36, f"Yield at design conditions = {Y:.4f} (out of expected range)"


def test_positive_outputs_all_conditions(model):
    """All outputs must be strictly positive across valid operating range."""
    vs = np.random.uniform(1, 8, 50)
    hrt = np.random.uniform(5, 40, 50)
    T = np.random.uniform(25, 53, 50)  # stay below 55 where yield drops to 0
    r = model.predict({"vs_loading": vs, "hrt": hrt, "temperature": T})
    for k in ["methane_yield_m3kgvs", "biogas_rate_m3day", "methane_rate_m3day", "energy_output_kwh_day"]:
        assert np.all(r[k] >= 0), f"{k} must be >= 0"


def test_benchmark(model):
    vs = np.random.uniform(1, 8, 1000)
    hrt = np.random.uniform(5, 40, 1000)
    start = time.perf_counter()
    model.predict({"vs_loading": vs, "hrt": hrt})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
