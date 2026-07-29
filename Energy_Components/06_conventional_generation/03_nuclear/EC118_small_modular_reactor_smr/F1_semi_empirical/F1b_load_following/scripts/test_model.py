"""EC118 -- SMR -- F1b Load-Following + Thermal Inertia -- Test Suite"""
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
    for k in ["power_output_mw", "thermal_power_mw", "coolant_outlet_temp_c",
              "xenon_concentration_rel", "available_reactivity_pcm",
              "ramp_rate_limit_pct_min", "can_restart", "thermal_lag_power_fraction"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC118"
    assert info["fidelity"] == "F1b"


# --- Power output ---

def test_full_power_output_steady_state(model):
    """At full power steady state, P_electric = 540 * 0.33 = 178.2 MW."""
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 1.0,
    })
    expected = 540.0 * 0.33
    assert abs(r["power_output_mw"] - expected) < 1.0, \
        f"P = {r['power_output_mw']:.2f}, expected {expected:.2f}"


def test_thermal_power_at_rated(model):
    """Thermal power at PLR=1.0 should be 540 MW."""
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 1.0,
    })
    assert abs(r["thermal_power_mw"] - 540.0) < 1.0


def test_plr_min_is_02(model):
    """SMR PLR_min should be 0.2 (deep load-following design)."""
    assert model._model.PLR_min == pytest.approx(0.2)


def test_deep_load_following_range(model):
    """SMR should operate at 20% load without clamping."""
    r = model.predict({
        "power_fraction": 0.2,
        "time_at_power_hours": 24.0,
        "previous_power_fraction": 0.2,
    })
    expected = 540.0 * 0.2 * 0.33
    assert abs(r["thermal_power_mw"] - 540.0 * 0.2) < 0.5


# --- Thermal inertia ---

def test_thermal_lag_at_start_of_ramp(model):
    """At t=0 (start of ramp from 100% to 30%), thermal lag = initial power fraction."""
    r = model.predict({
        "power_fraction": 0.3,
        "time_at_power_hours": 0.0,
        "previous_power_fraction": 1.0,
        "time_since_ramp_start_minutes": 0.0,  # at start: T_outlet still at 100% value
    })
    # Thermal lag fraction should be near 1.0 (coolant still at full-power temperature)
    assert float(r["thermal_lag_power_fraction"]) > 0.9, \
        f"At ramp start, thermal lag should be ~1.0, got {r['thermal_lag_power_fraction']:.4f}"


def test_thermal_lag_at_steady_state(model):
    """After a long time at new power, thermal lag should equal new PLR."""
    tau = model._model.tau_min
    r = model.predict({
        "power_fraction": 0.5,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 0.5,
        "time_since_ramp_start_minutes": 10.0 * tau,  # >> tau -> steady state
    })
    assert abs(float(r["thermal_lag_power_fraction"]) - 0.5) < 0.01, \
        f"Steady-state thermal lag should be 0.5 (= PLR), got {r['thermal_lag_power_fraction']:.4f}"


def test_thermal_lag_decays_exponentially(model):
    """
    At intermediate ramp time, thermal lag should be between initial and target values.
    At t=tau, e-folding: lag should be at ~63% of the way from initial to target.
    """
    m = model._model
    tau = m.tau_min
    # Ramp from 1.0 to 0.4
    T_initial = m.coolant_outlet_temp_steady(1.0)
    T_target  = m.coolant_outlet_temp_steady(0.4)
    T_at_tau  = m.coolant_outlet_temp_transient(1.0, 0.4, tau)
    # Expected: T_target + (T_initial - T_target) * exp(-1) = T_target + 0.368 * (T_initial - T_target)
    T_expected = T_target + (T_initial - T_target) * np.exp(-1.0)
    assert abs(T_at_tau - T_expected) < 0.5, \
        f"Temperature at t=tau: {T_at_tau:.2f}C, expected {T_expected:.2f}C"


def test_power_lower_than_target_during_ramp_down(model):
    """
    During a power ramp-down (100% -> 30%), at t=5min,
    actual power (thermal lag) should be HIGHER than target (coolant still hot).
    """
    r = model.predict({
        "power_fraction": 0.3,
        "time_at_power_hours": 0.0,
        "previous_power_fraction": 1.0,
        "time_since_ramp_start_minutes": 5.0,
    })
    assert float(r["thermal_lag_power_fraction"]) > 0.3, \
        "During ramp-down, thermal lag should keep effective power above target initially"


def test_coolant_temp_at_rated(model):
    """At PLR=1.0 steady state, coolant outlet T should be ~321 degC."""
    m = model._model
    T = m.coolant_outlet_temp_steady(1.0)
    assert abs(T - 321.0) < 1.0, f"T_outlet = {T:.2f}C, expected ~321"


def test_coolant_temp_at_min_plr(model):
    """At PLR=0.2, coolant outlet T should be near T_inlet + 0.2*(T_rated - T_inlet)."""
    m = model._model
    T_expected = m.T_inlet + 0.2 * (m.T_outlet_rated - m.T_inlet)
    T = m.coolant_outlet_temp_steady(0.2)
    assert abs(T - T_expected) < 0.5


# --- Xenon dynamics ---

def test_equilibrium_xenon_at_full_power(model):
    """At equilibrium full power, Xe_rel should be ~1.0."""
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 72.0,
        "previous_power_fraction": 1.0,
    })
    assert abs(r["xenon_concentration_rel"] - 1.0) < 0.05


def test_xenon_peak_after_deep_reduction(model):
    """After 100% -> 30% (deep load-following), Xe should peak significantly above new eq."""
    m = model._model
    Xe_10h = m.xenon_transient(1.0, 0.3, 10.0)
    Xe_eq_30 = float(m.equilibrium_xenon(0.3))
    assert Xe_10h > Xe_eq_30, \
        f"Xe at 10h ({Xe_10h:.3f}) should exceed eq at 30% ({Xe_eq_30:.3f})"


def test_xenon_nonnegative(model):
    """Xenon concentration must never be negative."""
    m = model._model
    for t in np.linspace(0, 72, 100):
        Xe = m.xenon_transient(1.0, 0.3, t)
        assert Xe >= 0.0, f"Xe negative at t={t:.1f}h: {Xe}"


def test_xenon_returns_to_equilibrium(model):
    """After 72h at 30%, Xe should be near new equilibrium."""
    m = model._model
    Xe_72h = m.xenon_transient(1.0, 0.3, 72.0)
    Xe_eq  = float(m.equilibrium_xenon(0.3))
    assert abs(Xe_72h - Xe_eq) < 0.05


# --- Ramp rate ---

def test_ramp_rate_is_5_pct_per_min(model):
    """SMR ramp rate limit should be 5 %/min."""
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 0.0,
        "previous_power_fraction": 1.0,
    })
    assert r["ramp_rate_limit_pct_min"] == pytest.approx(5.0)


def test_ramp_rate_constraint(model):
    """A 20% change in 1 minute should be limited to 5%."""
    m = model._model
    P_achievable, limited = m.ramp_rate_limit(0.5, 0.7, 1.0)
    assert limited is True
    assert abs(P_achievable - 0.55) < 0.001, \
        f"Should reach 0.55 in 1 min (5%/min), got {P_achievable:.4f}"


def test_full_range_ramp_feasibility(model):
    """Full swing from 20% to 100% (80% change) needs 16 minutes at 5%/min."""
    m = model._model
    # 16 min * 5%/min = 80% -> achievable
    P_achievable, limited = m.ramp_rate_limit(0.2, 1.0, 16.0)
    assert limited is False
    assert abs(P_achievable - 1.0) < 0.001


def test_ramp_needs_16min_not_14min(model):
    """14 minutes insufficient for 80% swing (only 70% achievable)."""
    m = model._model
    P_achievable, limited = m.ramp_rate_limit(0.2, 1.0, 14.0)
    assert limited is True
    assert abs(P_achievable - 0.90) < 0.001


# --- Restart capability ---

def test_can_restart_at_equilibrium(model):
    """At equilibrium, reactor should be able to restart."""
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 1.0,
    })
    assert r["can_restart"] is True
    assert r["available_reactivity_pcm"] > 0


def test_higher_reactivity_margin_than_pwr():
    """SMR total reactivity margin (5500 pcm) > large PWR (5000 pcm)."""
    model = ComponentModel()
    assert model._model.total_margin > 5000.0, \
        "SMR should have higher reactivity margin than large PWR for deep load-following"


# --- Benchmark ---

def test_benchmark(model):
    """1000 predictions must complete in < 1 second."""
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({
            "power_fraction": 0.4,
            "time_at_power_hours": 5.0,
            "previous_power_fraction": 1.0,
            "time_since_ramp_start_minutes": 10.0,
        })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.1f} ms")
    assert elapsed < 1.0
