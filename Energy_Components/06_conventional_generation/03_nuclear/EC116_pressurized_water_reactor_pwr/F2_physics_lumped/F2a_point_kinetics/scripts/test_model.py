"""EC116 -- PWR Nuclear Reactor -- F2a Point Kinetics -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import PWRPointKineticsF2a


@pytest.fixture
def model():
    return ComponentModel()


@pytest.fixture
def raw(model):
    return model._model


# ---- Initial conditions tests ----

def test_initial_conditions_equilibrium(raw):
    """Initial conditions should satisfy dC_i/dt = 0 (equilibrium)."""
    x0 = raw.initial_conditions()
    n0 = x0[0]
    C0 = x0[1:7]
    # At equilibrium: C_i = beta_i * n / (lambda_i * Lambda)
    C_expected = raw.beta * n0 / (raw.lam * raw.Lambda)
    np.testing.assert_allclose(C0, C_expected, rtol=1e-10)


def test_initial_conditions_thermal_equilibrium(raw):
    """Thermal equilibrium: heat generated = heat removed."""
    x0 = raw.initial_conditions()
    T_f = x0[7]
    T_m = x0[8]
    P0 = raw.P_th
    # Check fuel: P = hA*(T_f - T_m)
    Q_fg = raw.hA_fg * (T_f - T_m)
    assert abs(Q_fg - P0) / P0 < 1e-6, f"Fuel heat balance: Q_fg={Q_fg:.0f}, P={P0:.0f}"
    # Check moderator: hA*(T_f - T_m) = mdot_cp*(T_m - T_in)
    Q_out = raw.mdot_cp * (T_m - raw.T_in)
    assert abs(Q_out - P0) / P0 < 1e-6, f"Moderator heat balance: Q_out={Q_out:.0f}, P={P0:.0f}"


def test_beta_total(raw):
    """Total delayed neutron fraction should be 0.006502."""
    assert abs(raw.beta_total - 0.006502) < 1e-6


def test_six_groups(raw):
    """Model must have exactly 6 delayed neutron groups."""
    assert len(raw.beta) == 6
    assert len(raw.lam) == 6


# ---- Steady-state stability tests ----

def test_zero_reactivity_steady(model):
    """With zero external reactivity, system should remain at steady state."""
    r = model.predict({"rho_ext": 0.0, "dt": 0.1, "duration_s": 50.0})
    # Neutron population should stay at 1.0
    assert np.all(np.abs(r["n"] - 1.0) < 0.001), \
        f"n drifted: min={r['n'].min():.6f}, max={r['n'].max():.6f}"
    # Temperatures should be stable
    T_f_range = r["T_f"].max() - r["T_f"].min()
    T_m_range = r["T_m"].max() - r["T_m"].min()
    assert T_f_range < 0.1, f"T_f range={T_f_range:.4f} K (should be ~0)"
    assert T_m_range < 0.1, f"T_m range={T_m_range:.4f} K (should be ~0)"


def test_power_at_steady_state(model):
    """At n=1.0, thermal power should be P_rated."""
    r = model.predict({"rho_ext": 0.0, "dt": 0.1, "duration_s": 10.0})
    P_th = r["P_thermal_W"][-1]
    assert abs(P_th - 3000e6) / 3000e6 < 0.001


# ---- Reactivity insertion tests ----

def test_positive_step_increases_power(model):
    """Positive reactivity step should increase neutron population."""
    r = model.predict_step({"rho_step": 0.001, "dt": 0.05, "duration_s": 50.0})
    # After insertion, n should increase from 1.0
    idx_after = np.searchsorted(r["t"], 2.0)
    assert r["n"][idx_after] > 1.0, f"n after +step = {r['n'][idx_after]:.4f}, should be > 1"


def test_negative_step_decreases_power(model):
    """Negative reactivity should decrease neutron population."""
    r = model.predict_step({"rho_step": -0.001, "dt": 0.05, "duration_s": 50.0})
    idx_after = np.searchsorted(r["t"], 2.0)
    assert r["n"][idx_after] < 1.0, f"n after -step = {r['n'][idx_after]:.4f}, should be < 1"


def test_negative_feedback_limits_excursion(model):
    """Temperature feedback should limit power excursion for subcritical insertion."""
    r = model.predict_step({"rho_step": 0.003, "dt": 0.05, "duration_s": 100.0})
    # With negative feedback, n should not go unbounded
    # (0.003 < beta_total = 0.006502, so it is a delayed supercritical case)
    assert r["n"].max() < 10.0, \
        f"n_max = {r['n'].max():.2f}, should be bounded by feedback"


def test_fuel_temperature_increases_with_power(model):
    """Positive reactivity should increase fuel temperature."""
    r = model.predict_step({"rho_step": 0.002, "dt": 0.1, "duration_s": 100.0})
    T_f_init = r["T_f"][0]
    T_f_final = r["T_f"][-1]
    assert T_f_final > T_f_init, \
        f"T_f should increase: init={T_f_init:.1f}, final={T_f_final:.1f}"


def test_moderator_temperature_increases_with_power(model):
    """Positive reactivity should increase moderator temperature."""
    r = model.predict_step({"rho_step": 0.002, "dt": 0.1, "duration_s": 100.0})
    T_m_init = r["T_m"][0]
    T_m_final = r["T_m"][-1]
    assert T_m_final > T_m_init


def test_doppler_feedback_sign(raw):
    """Doppler coefficient must be negative (self-limiting)."""
    assert raw.alpha_f < 0, f"alpha_f={raw.alpha_f} should be negative"


def test_moderator_feedback_sign(raw):
    """Moderator temperature coefficient must be negative."""
    assert raw.alpha_m < 0, f"alpha_m={raw.alpha_m} should be negative"


# ---- Ramp reactivity test ----

def test_ramp_insertion(model):
    """Ramp insertion should produce gradual power increase."""
    r = model.predict_ramp({
        "rho_rate": 5e-5, "rho_max": 0.002,
        "dt": 0.1, "duration_s": 100.0
    })
    # Power should increase gradually
    idx_mid = len(r["t"]) // 2
    assert r["n"][idx_mid] > 1.0, "Power should increase during ramp"


# ---- Prompt jump approximation test ----

def test_prompt_jump(model):
    """For small step rho << beta, prompt jump should be ~ n0 * beta / (beta - rho)."""
    rho_step = 0.001  # 100 pcm << beta = 650.2 pcm
    r = model.predict_step({"rho_step": rho_step, "dt": 0.001, "duration_s": 5.0,
                             "t_insert": 0.5})
    # After prompt jump (~milliseconds), before delayed neutrons kick in significantly
    # The prompt jump gives n ~ n0 * beta / (beta - rho) for the first few milliseconds
    idx = np.searchsorted(r["t"], 0.505)  # 5 ms after insertion
    beta = model._model.beta_total
    prompt_jump_expected = 1.0 * beta / (beta - rho_step)
    # Allow 5% tolerance (thermal feedback starts quickly)
    assert abs(r["n"][idx] - prompt_jump_expected) / prompt_jump_expected < 0.05, \
        f"Prompt jump: n={r['n'][idx]:.4f}, expected~{prompt_jump_expected:.4f}"


# ---- Solver tests ----

def test_radau_solver(model):
    """Radau solver should handle stiff system without failure."""
    r = model.predict({
        "rho_ext": 0.0, "dt": 0.1, "duration_s": 10.0, "method": "Radau"
    })
    assert len(r["t"]) > 0, "Radau solver produced no output"


def test_bdf_solver(model):
    """BDF solver should also work for this stiff system."""
    r = model.predict({
        "rho_ext": 0.0, "dt": 0.1, "duration_s": 10.0, "method": "BDF"
    })
    assert len(r["t"]) > 0, "BDF solver produced no output"


def test_solvers_agree(model):
    """Radau and BDF should give same results for same problem."""
    inputs_base = {"rho_ext": 0.001, "dt": 0.1, "duration_s": 20.0}
    # Use callable for rho_ext to test both paths
    def rho_func(t):
        return 0.001 if t >= 1.0 else 0.0
    r_radau = model.predict({**inputs_base, "rho_ext": rho_func, "method": "Radau"})
    r_bdf = model.predict({**inputs_base, "rho_ext": rho_func, "method": "BDF"})
    # Check final neutron populations agree within 0.1%
    assert abs(r_radau["n"][-1] - r_bdf["n"][-1]) / r_radau["n"][-1] < 0.001, \
        f"Radau n={r_radau['n'][-1]:.6f}, BDF n={r_bdf['n'][-1]:.6f}"


# ---- Output and interface tests ----

def test_output_keys(model):
    """Check all expected output keys."""
    r = model.predict({"rho_ext": 0.0, "dt": 0.1, "duration_s": 5.0})
    for key in ["t", "n", "C", "T_f", "T_m", "P_thermal_W", "P_elec_W", "rho"]:
        assert key in r, f"Missing key: {key}"


def test_precursor_array_shape(model):
    """Precursor concentrations should have shape (6, N)."""
    r = model.predict({"rho_ext": 0.0, "dt": 0.1, "duration_s": 5.0})
    assert r["C"].shape[0] == 6, f"Expected 6 precursor groups, got {r['C'].shape[0]}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC116"
    assert info["fidelity"] == "F2a"


# ---- Load-following scenario ----

def test_load_following(model):
    """Simulate a load-following scenario: reduce power by negative reactivity."""
    def rho_ext(t):
        if t < 10.0:
            return 0.0
        elif t < 20.0:
            return -0.002  # Insert negative reactivity to reduce power
        else:
            return 0.0  # Remove rod -> return to full power

    r = model.predict({"rho_ext": rho_ext, "dt": 0.1, "duration_s": 100.0})
    # During negative reactivity, power should decrease
    idx_15 = np.searchsorted(r["t"], 15.0)
    assert r["n"][idx_15] < 1.0, "Power should decrease with negative reactivity"
    # After removal, power should eventually recover toward original
    assert r["n"][-1] > r["n"][idx_15], "Power should recover after rod removal"


# ---- Benchmark ----

def test_benchmark(model):
    """100s transient simulation should complete in < 30s wall time."""
    start = time.perf_counter()
    model.predict_step({"rho_step": 0.001, "dt": 0.1, "duration_s": 100.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 100s transient in {elapsed * 1000:.1f} ms")
    assert elapsed < 30.0
