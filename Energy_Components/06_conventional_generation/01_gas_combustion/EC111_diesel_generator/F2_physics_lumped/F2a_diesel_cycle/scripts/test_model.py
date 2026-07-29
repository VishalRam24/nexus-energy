"""EC111 -- Diesel Generator -- F2a Diesel Cycle -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import DieselGeneratorF2a


@pytest.fixture
def model():
    return ComponentModel()


@pytest.fixture
def raw(model):
    return model._model


# ---- Thermodynamic cycle tests ----

def test_diesel_efficiency_formula(raw):
    """Verify diesel efficiency matches the analytical formula."""
    r_c = 18.0
    r_co = 2.0
    g = 1.35
    expected = 1.0 - (1.0 / r_c ** (g - 1.0)) * ((r_co ** g - 1.0) / (g * (r_co - 1.0)))
    computed = raw.diesel_efficiency(r_c, r_co)
    assert abs(computed - expected) < 1e-10, f"eta={computed}, expected={expected}"


def test_diesel_efficiency_range(raw):
    """Diesel cycle efficiency must be between 40-65% for typical parameters."""
    eta = raw.diesel_efficiency()
    assert 0.40 < eta < 0.65, f"eta_diesel={eta:.4f} out of expected range"


def test_efficiency_increases_with_compression_ratio(raw):
    """Higher compression ratio should increase efficiency."""
    eta_14 = raw.diesel_efficiency(r_c=14.0, r_co=2.0)
    eta_18 = raw.diesel_efficiency(r_c=18.0, r_co=2.0)
    eta_22 = raw.diesel_efficiency(r_c=22.0, r_co=2.0)
    assert eta_14 < eta_18 < eta_22, \
        f"Efficiency not monotonic: {eta_14:.4f}, {eta_18:.4f}, {eta_22:.4f}"


def test_efficiency_decreases_with_cutoff_ratio(raw):
    """Higher cutoff ratio (more heat addition) should decrease efficiency."""
    eta_low = raw.diesel_efficiency(r_c=18.0, r_co=1.5)
    eta_high = raw.diesel_efficiency(r_c=18.0, r_co=3.0)
    assert eta_low > eta_high, \
        f"eta(r_co=1.5)={eta_low:.4f} should be > eta(r_co=3.0)={eta_high:.4f}"


def test_cycle_state_points_pressure(raw):
    """P2 > P1 (compression), P3 = P2 (constant pressure), P4 < P3 (expansion)."""
    sp = raw.cycle_state_points()
    assert sp["P2"] > sp["P1"], "Compression must increase pressure"
    assert abs(sp["P3"] - sp["P2"]) / sp["P2"] < 1e-10, "Process 2-3 must be const pressure"
    assert sp["P4"] < sp["P3"], "Expansion must decrease pressure"


def test_cycle_state_points_temperature(raw):
    """T2 > T1 (compression), T3 > T2 (heat add), T4 < T3 (expansion), T4 > T1."""
    sp = raw.cycle_state_points()
    assert sp["T2"] > sp["T1"], "Compression must increase temperature"
    assert sp["T3"] > sp["T2"], "Heat addition must increase temperature"
    assert sp["T4"] < sp["T3"], "Expansion must decrease temperature"
    assert sp["T4"] > sp["T1"], "Exhaust temp must exceed intake temp"


def test_first_law_consistency(raw):
    """q_add - q_rej = w_net (first law of thermodynamics)."""
    q_add = raw.heat_added()
    q_rej = raw.heat_rejected()
    w_net = raw.net_work()
    assert abs(q_add - q_rej - w_net) < 1e-6, \
        f"First law violation: q_add={q_add:.2f}, q_rej={q_rej:.2f}, w_net={w_net:.2f}"


def test_efficiency_equals_work_over_heat(raw):
    """eta = w_net / q_add must match diesel_efficiency formula."""
    q_add = raw.heat_added()
    w_net = raw.net_work()
    eta_calc = w_net / q_add
    eta_formula = raw.diesel_efficiency()
    assert abs(eta_calc - eta_formula) < 1e-6, \
        f"eta from work/heat={eta_calc:.6f} != formula={eta_formula:.6f}"


# ---- Generator efficiency tests ----

def test_gen_efficiency_at_rated(raw):
    """Generator efficiency at full load equals rated efficiency."""
    eta = raw.generator_efficiency(1.0)
    assert abs(eta - raw.eta_gen_rated) < 1e-10


def test_gen_efficiency_decreases_at_part_load(raw):
    """Generator efficiency should decrease at part load."""
    eta_full = raw.generator_efficiency(1.0)
    eta_half = raw.generator_efficiency(0.5)
    eta_quarter = raw.generator_efficiency(0.25)
    assert eta_full > eta_half > eta_quarter


def test_gen_efficiency_positive(raw):
    """Generator efficiency must be positive for all reasonable loads."""
    for load in [0.1, 0.25, 0.5, 0.75, 1.0]:
        eta = raw.generator_efficiency(load)
        assert eta > 0, f"eta_gen({load})={eta} must be positive"


# ---- BSFC tests ----

def test_bsfc_at_rated(raw):
    """BSFC at rated load should be ~210 g/kWh."""
    bsfc = raw.bsfc(1.0) * 3.6e9  # convert to g/kWh
    assert 200 < bsfc < 230, f"BSFC at rated = {bsfc:.1f} g/kWh, expected ~210"


def test_bsfc_increases_at_part_load(raw):
    """BSFC should increase at part load (less efficient)."""
    bsfc_full = raw.bsfc(1.0)
    bsfc_half = raw.bsfc(0.5)
    bsfc_quarter = raw.bsfc(0.25)
    assert bsfc_full < bsfc_half < bsfc_quarter


# ---- Steady-state tests ----

def test_steady_state_full_load(model):
    """At full load, P_elec should be rated power."""
    ss = model.predict_steady_state({"load_fraction": 1.0})
    assert abs(ss["P_elec_W"] - 500000.0) < 1.0


def test_steady_state_efficiency(model):
    """Overall efficiency should be 30-45% (typical diesel genset)."""
    ss = model.predict_steady_state({"load_fraction": 0.75})
    assert 0.25 < ss["eta_overall"] < 0.50, f"eta_overall={ss['eta_overall']:.4f}"


def test_steady_state_fuel_rate(model):
    """Fuel rate at rated should be reasonable (90-140 L/h for 500 kW)."""
    ss = model.predict_steady_state({"load_fraction": 1.0})
    assert 80 < ss["fuel_rate_L_h"] < 160, f"Fuel rate={ss['fuel_rate_L_h']:.1f} L/h"


# ---- Dynamic simulation tests ----

def test_simulation_constant_load(model):
    """Under constant load, speed should settle near nominal."""
    r = model.predict({
        "P_load": 250000.0, "dt": 0.05, "duration_s": 20.0,
    })
    # Speed should be within 2% of nominal at end
    assert abs(r["omega_rpm"][-1] - 1500.0) / 1500.0 < 0.02, \
        f"Final speed={r['omega_rpm'][-1]:.1f}, expected ~1500 rpm"


def test_simulation_load_step(model):
    """Load step should cause frequency dip then recovery."""
    def load_step(t):
        return 250000.0 if t < 5.0 else 500000.0

    r = model.predict({
        "P_load": load_step, "dt": 0.05, "duration_s": 20.0,
    })
    # Find minimum speed after step
    idx_step = np.searchsorted(r["t"], 5.0)
    min_rpm_after = np.min(r["omega_rpm"][idx_step:])
    # Should dip but recover
    assert min_rpm_after < 1500.0, "Speed should dip on load increase"
    assert abs(r["omega_rpm"][-1] - 1500.0) / 1500.0 < 0.03, \
        f"Speed should recover: final={r['omega_rpm'][-1]:.1f}"


def test_simulation_output_keys(model):
    """Check all expected output keys are present."""
    r = model.predict({"P_load": 250000.0, "dt": 0.1, "duration_s": 5.0})
    for key in ["t", "omega_rpm", "frequency_Hz", "P_elec_W", "P_engine_W",
                "fuel_rate_kg_s", "eta_overall", "eta_gen", "load_frac"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC111"
    assert info["fidelity"] == "F2a"


# ---- Edge cases ----

def test_zero_load(model):
    """Zero load should still run (idle condition)."""
    ss = model.predict_steady_state({"load_fraction": 0.01})
    assert ss["P_elec_W"] > 0
    assert ss["eta_overall"] > 0


def test_cutoff_ratio_one_limit(raw):
    """As r_co -> 1, diesel efficiency -> Otto efficiency."""
    r_co = 1.0001
    r_c = 18.0
    g = 1.35
    eta_diesel = raw.diesel_efficiency(r_c, r_co)
    eta_otto = 1.0 - 1.0 / r_c ** (g - 1.0)
    # Should be very close for r_co -> 1
    assert abs(eta_diesel - eta_otto) < 0.01, \
        f"eta_diesel(r_co~1)={eta_diesel:.4f}, eta_otto={eta_otto:.4f}"


# ---- Benchmark ----

def test_benchmark(model):
    """20s simulation should complete in < 10s wall time."""
    start = time.perf_counter()
    model.predict({"P_load": 250000.0, "dt": 0.05, "duration_s": 20.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 20s sim in {elapsed * 1000:.1f} ms")
    assert elapsed < 10.0
