"""EC048 — Perovskite Solar Cell — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

@pytest.fixture
def model():
    return ComponentModel()

def test_predict_keys(model):
    r = model.predict({"irradiance": 1000.0, "cell_temperature": 25.0})
    for k in ["v_mp", "i_mp", "p_mp", "v_oc", "i_sc", "efficiency"]:
        assert k in r, f"Missing key: {k}"

def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC048"
    assert info["fidelity"] == "F1a"

def test_stc_power_range(model):
    """At STC, Pmp = Vmp*Imp = 1.0V * 0.5A = 0.5W for 25cm2 cell at 20% efficiency."""
    r = model.predict({"irradiance": 1000.0, "cell_temperature": 25.0})
    p = float(r["p_mp"])
    assert 0.3 < p < 0.8, f"STC Pmp={p:.3f}W, expected ~0.5W for 25cm2 perovskite (Vmp=1V, Imp=0.5A)"

def test_stc_voc_high(model):
    """Perovskite Voc should be > 1.0V at STC (wide bandgap benefit)."""
    r = model.predict({"irradiance": 1000.0, "cell_temperature": 25.0})
    voc = float(r["v_oc"])
    assert voc > 1.0, f"Voc={voc:.3f}V, expected >1.0V for perovskite"

def test_stc_efficiency_range(model):
    """Efficiency at STC = Pmp/(G*area) = 0.5/(1000*0.0025) = 20% for lab perovskite."""
    r = model.predict({"irradiance": 1000.0, "cell_temperature": 25.0})
    eta = float(r["efficiency"])
    assert 0.15 < eta < 0.30, f"Efficiency at STC = {eta*100:.1f}%, expected 15-30%"

def test_zero_irradiance(model):
    """At zero irradiance, all outputs should be zero."""
    r = model.predict({"irradiance": 0.0, "cell_temperature": 25.0})
    assert float(r["p_mp"]) == pytest.approx(0.0, abs=1e-6)
    assert float(r["i_sc"]) == pytest.approx(0.0, abs=1e-6)
    assert float(r["efficiency"]) == pytest.approx(0.0, abs=1e-6)

def test_power_proportional_to_irradiance(model):
    """Power should scale roughly linearly with irradiance."""
    G_vals = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    r = model.predict({"irradiance": G_vals, "cell_temperature": 25.0})
    p = r["p_mp"]
    # Power should be monotonically increasing
    assert np.all(np.diff(p) > 0), "Power must increase with irradiance"

def test_power_decreases_with_temperature(model):
    """Power should decrease as temperature increases (negative temp coeff)."""
    T_vals = np.array([0.0, 15.0, 25.0, 40.0, 60.0])
    r = model.predict({"irradiance": 1000.0, "cell_temperature": T_vals})
    p = r["p_mp"]
    assert np.all(np.diff(p) < 0), "Power must decrease as temperature rises"

def test_voc_higher_than_vmp(model):
    """Voc must always be greater than Vmp."""
    G_vals = np.array([200.0, 500.0, 800.0, 1000.0])
    r = model.predict({"irradiance": G_vals, "cell_temperature": 25.0})
    assert np.all(r["v_oc"] > r["v_mp"]), "Voc must be > Vmp always"

def test_isc_higher_than_imp(model):
    """Isc must always be >= Imp."""
    G_vals = np.array([200.0, 500.0, 800.0, 1000.0])
    r = model.predict({"irradiance": G_vals, "cell_temperature": 25.0})
    assert np.all(r["i_sc"] >= r["i_mp"]), "Isc must be >= Imp always"

def test_higher_irradiance_than_silicon_check(model):
    """Voc at STC should be noticeably higher than Si single-cell (~0.6V)."""
    r = model.predict({"irradiance": 1000.0, "cell_temperature": 25.0})
    assert float(r["v_oc"]) > 0.9, "Perovskite single-cell Voc must exceed Si single-cell Voc"

def test_benchmark(model):
    """1000 predictions should complete in <1 second."""
    G = np.random.uniform(50, 1200, 1000)
    T = np.random.uniform(-10, 80, 1000)
    start = time.perf_counter()
    model.predict({"irradiance": G, "cell_temperature": T})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s"
