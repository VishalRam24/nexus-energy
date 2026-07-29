"""EC010 — Solid Oxide Electrolyser (SOEC) — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"current_density": 1.0, "temperature": 800.0})
    for k in ["cell_voltage", "stack_voltage", "hydrogen_rate_mols", "power_kw", "efficiency", "asr"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC010"
    assert info["fidelity"] == "F1a"


def test_voltage_above_e_rev(model):
    """Cell voltage at high j must exceed E_rev."""
    j_arr = np.linspace(0.1, 2.0, 50)
    r = model.predict({"current_density": j_arr, "temperature": 800.0})
    # E_rev at 800C (1073 K) = 1.253 - 0.00024*(1073-298) = 1.253 - 0.186 = ~1.067 V
    E_rev_800 = 1.253 - 0.00024 * (1073.15 - 298.15)
    assert np.all(r["cell_voltage"] >= E_rev_800 - 1e-9), "V_cell must be >= E_rev"


def test_voltage_increases_with_current_density(model):
    """Cell voltage must monotonically increase with current density."""
    j_arr = np.linspace(0.05, 2.0, 100)
    r = model.predict({"current_density": j_arr, "temperature": 800.0})
    diffs = np.diff(r["cell_voltage"])
    assert np.all(diffs > 0), "V_cell must increase with j"


def test_asr_decreases_with_temperature(model):
    """ASR must decrease with increasing temperature (Arrhenius)."""
    temps = np.array([600.0, 700.0, 800.0, 900.0])
    r_low = model.predict({"current_density": 0.5, "temperature": temps[0]})
    r_high = model.predict({"current_density": 0.5, "temperature": temps[-1]})
    assert float(r_high["asr"]) < float(r_low["asr"]), \
        "ASR must decrease with temperature"


def test_asr_arrhenius_monotone(model):
    """ASR is strictly decreasing over full temperature range."""
    temps = np.linspace(600, 900, 50)
    r = model.predict({"current_density": 0.5, "temperature": temps})
    assert np.all(np.diff(r["asr"]) < 0), "ASR should decrease monotonically with T"


def test_h2_rate_proportional_to_current(model):
    """H2 rate proportional to current (100% Faraday efficiency for SOEC)."""
    r1 = model.predict({"current_density": 0.5, "temperature": 800.0})
    r2 = model.predict({"current_density": 1.0, "temperature": 800.0})
    ratio = float(r2["hydrogen_rate_mols"]) / float(r1["hydrogen_rate_mols"])
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_efficiency_reasonable(model):
    """Efficiency should be between 0 and ~1.3 (can exceed 1 if below thermo-neutral)."""
    j_arr = np.linspace(0.1, 2.0, 50)
    r = model.predict({"current_density": j_arr, "temperature": 800.0})
    assert np.all(r["efficiency"] >= 0.0)
    assert np.all(r["efficiency"] <= 1.5)


def test_zero_current(model):
    """At j=0, no H2 and no power."""
    r = model.predict({"current_density": 0.0, "temperature": 800.0})
    assert float(r["hydrogen_rate_mols"]) == pytest.approx(0.0, abs=1e-12)
    assert float(r["power_kw"]) == pytest.approx(0.0, abs=1e-9)


def test_stack_voltage_consistency(model):
    """Stack voltage = N_cells * cell voltage."""
    r = model.predict({"current_density": 1.0, "temperature": 800.0})
    ratio = float(r["stack_voltage"]) / float(r["cell_voltage"])
    assert ratio == pytest.approx(30.0, rel=1e-6)


def test_benchmark(model):
    j_arr = np.random.uniform(0.05, 2.0, 1000)
    T_arr = np.random.uniform(600, 900, 1000)
    start = time.perf_counter()
    model.predict({"current_density": j_arr, "temperature": T_arr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
