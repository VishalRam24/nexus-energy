"""EC117 -- BWR -- F1b Load Following -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Output key checks ---

def test_predict_keys(model):
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 1.0,
    })
    for k in ["power_output_mw", "xenon_concentration_rel",
              "available_reactivity_pcm", "void_reactivity_pcm",
              "ramp_rate_limit_pct_min", "can_restart"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC117"
    assert info["fidelity"] == "F1b"


# --- Power output ---

def test_full_power_output(model):
    """At full power: P_electric = 3300 * 0.33 = 1089 MW."""
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 1.0,
    })
    expected = 3300.0 * 0.33
    assert abs(r["power_output_mw"] - expected) < 1.0, \
        f"P = {r['power_output_mw']:.1f} MW, expected {expected:.1f}"


def test_power_scales_with_fraction(model):
    """Power at 80% should be 80% of full power (no efficiency derating in F1b)."""
    r_full = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 1.0,
    })
    r_80 = model.predict({
        "power_fraction": 0.8,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 0.8,
    })
    assert abs(r_80["power_output_mw"] - r_full["power_output_mw"] * 0.8) < 1.0


# --- Xenon dynamics ---

def test_equilibrium_xenon_at_full_power(model):
    """At long-term full power, Xe_rel should be ~1.0."""
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 72.0,
        "previous_power_fraction": 1.0,
    })
    assert abs(r["xenon_concentration_rel"] - 1.0) < 0.05


def test_xenon_peak_after_power_reduction(model):
    """After 100% -> 75% step, Xe should peak above new equilibrium (~8-12h)."""
    m = model._model
    Xe_10h = m.xenon_transient(1.0, 0.75, 10.0)
    Xe_eq_75 = float(m.equilibrium_xenon(0.75))
    assert Xe_10h > Xe_eq_75, \
        f"Xe at 10h ({Xe_10h:.3f}) should exceed eq at 75% ({Xe_eq_75:.3f})"


def test_xenon_returns_to_equilibrium(model):
    """After 72h at new power level, Xe should approach new equilibrium."""
    m = model._model
    Xe_72h = m.xenon_transient(1.0, 0.75, 72.0)
    Xe_eq  = float(m.equilibrium_xenon(0.75))
    assert abs(Xe_72h - Xe_eq) < 0.05, \
        f"Xe at 72h ({Xe_72h:.3f}) should be near eq ({Xe_eq:.3f})"


def test_xenon_nonnegative(model):
    """Xenon concentration must never be negative."""
    m = model._model
    for t in np.linspace(0, 72, 100):
        Xe = m.xenon_transient(1.0, 0.6, t)
        assert Xe >= 0.0, f"Xe negative at t={t:.1f}h: {Xe}"


# --- Ramp rate ---

def test_ramp_rate_is_1_pct_per_min(model):
    """BWR ramp rate limit must be 1 %/min (not 5 like PWR)."""
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 0.0,
        "previous_power_fraction": 1.0,
    })
    assert r["ramp_rate_limit_pct_min"] == pytest.approx(1.0), \
        f"Ramp rate = {r['ramp_rate_limit_pct_min']} %/min, expected 1.0"


def test_ramp_rate_slower_than_pwr():
    """BWR ramp rate (1%/min) is slower than PWR (5%/min)."""
    model = ComponentModel()
    assert model._model.ramp_limit < 5.0, \
        "BWR ramp rate should be < 5%/min (PWR limit)"


def test_ramp_rate_limits_fast_change(model):
    """A 20% power step in 1 minute should be limited to 1%."""
    m = model._model
    P_achievable, limited = m.ramp_rate_limit(0.8, 1.0, 1.0)
    assert limited is True
    assert abs(P_achievable - 0.81) < 0.001, \
        f"Should reach 0.81 in 1 min (1%/min * 1 min), got {P_achievable:.4f}"


def test_ramp_rate_with_sufficient_time(model):
    """Full ramp from 60% to 100% (40% change) needs 40 minutes at 1%/min."""
    m = model._model
    P_achievable, limited = m.ramp_rate_limit(0.6, 1.0, 40.0)
    assert limited is False
    assert abs(P_achievable - 1.0) < 0.001


def test_ramp_rate_insufficient_time(model):
    """40% change in 20 minutes is not achievable at 1%/min."""
    m = model._model
    P_achievable, limited = m.ramp_rate_limit(0.6, 1.0, 20.0)
    assert limited is True
    assert abs(P_achievable - 0.8) < 0.001  # 0.6 + 1%/min * 20 min = 0.80


# --- Void reactivity feedback ---

def test_void_reactivity_negative_at_part_load(model):
    """
    At part load, void fraction decreases (less boiling).
    Negative void coefficient means positive reactivity insertion (void_rho > 0
    when void decreases, i.e., rho_void = coeff * delta_void < 0 means
    coeff < 0 and delta_void < 0 -> rho_void > 0).
    Test: void_reactivity_pcm at PLR=0.8 should be positive (positive feedback signal
    that must be compensated by control rods).
    """
    m = model._model
    # At PLR=0.8 (lower than reference 1.0): delta_void < 0, coeff < 0 -> rho > 0
    rho_void = m.void_reactivity_pcm(0.8, 1.0)
    assert rho_void > 0, \
        f"Void reactivity at PLR=0.8 should be positive (void decreases at part load), got {rho_void}"


def test_void_reactivity_zero_at_reference(model):
    """Void reactivity = 0 when current PLR equals reference PLR."""
    m = model._model
    rho_void = m.void_reactivity_pcm(1.0, 1.0)
    assert abs(rho_void) < 1e-9


# --- Restart capability ---

def test_can_restart_at_equilibrium(model):
    """At equilibrium full power, reactor can restart."""
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 1.0,
    })
    assert r["can_restart"] is True
    assert r["available_reactivity_pcm"] > 0


# --- PLR constraints ---

def test_plr_min_enforced(model):
    """Power fraction below PLR_min (0.6) should be clamped to PLR_min."""
    r = model.predict({
        "power_fraction": 0.3,     # below BWR PLR_min
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 0.3,
    })
    expected_min = 3300.0 * 0.6 * 0.33
    assert abs(r["power_output_mw"] - expected_min) < 1.0, \
        f"P = {r['power_output_mw']:.1f} MW, expected clamped {expected_min:.1f}"


# --- Benchmark ---

def test_benchmark(model):
    """1000 predictions must complete in < 1 second."""
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({
            "power_fraction": 0.8,
            "time_at_power_hours": 5.0,
            "previous_power_fraction": 1.0,
        })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
