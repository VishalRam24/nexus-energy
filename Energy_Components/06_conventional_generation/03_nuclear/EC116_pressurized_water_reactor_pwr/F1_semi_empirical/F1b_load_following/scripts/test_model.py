"""EC116 -- PWR -- F1b Load Following -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 1.0,
    })
    for k in ["power_output_mw", "xenon_concentration_rel",
              "available_reactivity_pcm", "ramp_rate_limit_pct_min", "can_restart"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC116"
    assert info["fidelity"] == "F1b"


def test_full_power_output(model):
    """At full power, P_electric = 3400 * 0.33 = 1122 MW."""
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 1.0,
    })
    expected = 3400.0 * 0.33
    assert abs(r["power_output_mw"] - expected) < 1.0


def test_power_scales_with_fraction(model):
    """Power output at 50% should be half of full power."""
    r_full = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 1.0,
    })
    r_half = model.predict({
        "power_fraction": 0.5,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 0.5,
    })
    assert abs(r_half["power_output_mw"] - r_full["power_output_mw"] / 2) < 1.0


def test_equilibrium_xenon_at_full_power(model):
    """At equilibrium full power, Xe_rel should be ~1.0."""
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 72.0,
        "previous_power_fraction": 1.0,
    })
    assert abs(r["xenon_concentration_rel"] - 1.0) < 0.05


def test_xenon_peak_after_power_reduction(model):
    """After reducing from 100% to 50%, Xe should peak above equilibrium."""
    m = model._model
    # Check Xe at 10 hours after step-down
    Xe_10h = m.xenon_transient(1.0, 0.5, 10.0)
    Xe_eq_50 = float(m.equilibrium_xenon(0.5))
    # Xe at 10h should be above the new equilibrium (xenon overshoot)
    assert Xe_10h > Xe_eq_50, \
        f"Xe at 10h ({Xe_10h:.3f}) should be above eq at 50% ({Xe_eq_50:.3f})"


def test_xenon_returns_to_equilibrium(model):
    """After long time at new power, Xe should approach new equilibrium."""
    m = model._model
    Xe_72h = m.xenon_transient(1.0, 0.5, 72.0)
    Xe_eq_50 = float(m.equilibrium_xenon(0.5))
    assert abs(Xe_72h - Xe_eq_50) < 0.05, \
        f"Xe at 72h ({Xe_72h:.3f}) should be near eq ({Xe_eq_50:.3f})"


def test_can_restart_at_equilibrium(model):
    """At equilibrium, reactor should be able to restart (available reactivity > 0)."""
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 1.0,
    })
    assert r["can_restart"] is True
    assert r["available_reactivity_pcm"] > 0


def test_ramp_rate_limit_value(model):
    """Ramp rate limit should be 5 %/min."""
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 0.0,
        "previous_power_fraction": 1.0,
    })
    assert r["ramp_rate_limit_pct_min"] == pytest.approx(5.0)


def test_ramp_rate_constraint(model):
    """A 50% power change in 1 minute should be limited to 5%."""
    m = model._model
    P_achievable, limited = m.ramp_rate_limit(0.5, 1.0, 1.0)
    assert limited is True
    assert abs(P_achievable - 0.55) < 0.001, \
        f"Should reach 0.55 in 1 min, got {P_achievable}"


def test_ramp_rate_sufficient_time(model):
    """With enough time (>10 min), full ramp from 50% to 100% should be achievable."""
    m = model._model
    P_achievable, limited = m.ramp_rate_limit(0.5, 1.0, 15.0)
    assert limited is False
    assert abs(P_achievable - 1.0) < 0.001


def test_plr_min_enforced(model):
    """Power fraction below PLR_min should be clamped."""
    r = model.predict({
        "power_fraction": 0.1,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 0.1,
    })
    expected_min = 3400.0 * 0.3 * 0.33
    assert abs(r["power_output_mw"] - expected_min) < 1.0


def test_xenon_nonnegative(model):
    """Xenon concentration must never be negative."""
    m = model._model
    for t in np.linspace(0, 72, 100):
        Xe = m.xenon_transient(1.0, 0.3, t)
        assert Xe >= 0.0, f"Xe negative at t={t:.1f}h: {Xe}"


def test_benchmark(model):
    """1000 predictions must complete in < 1 second."""
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({
            "power_fraction": 0.7,
            "time_at_power_hours": 5.0,
            "previous_power_fraction": 1.0,
        })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
