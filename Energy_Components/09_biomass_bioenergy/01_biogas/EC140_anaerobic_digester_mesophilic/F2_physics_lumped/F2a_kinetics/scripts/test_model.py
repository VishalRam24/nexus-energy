"""EC140 -- Anaerobic Digester -- F2a Monod Kinetics -- Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import AnaerobicDigesterF2a


@pytest.fixture
def model():
    return ComponentModel()


@pytest.fixture
def raw(model):
    return model._model


# ---- Monod kinetics tests ----

def test_monod_zero_substrate(raw):
    """Monod rate must be zero when S = 0."""
    assert raw.monod(0.0) == 0.0


def test_monod_saturation(raw):
    """At very high S, Monod rate approaches mu_max."""
    mu = raw.monod(1e6)
    assert abs(mu - raw.mu_max) / raw.mu_max < 1e-4


def test_monod_half_saturation(raw):
    """At S = K_s, Monod rate should be mu_max / 2."""
    mu = raw.monod(raw.K_s)
    assert abs(mu - raw.mu_max / 2.0) < 1e-10


def test_monod_monotonic(raw):
    """Monod rate must be monotonically increasing with S."""
    S_vals = np.linspace(0, 100, 100)
    mu_vals = [raw.monod(s) for s in S_vals]
    assert all(mu_vals[i] <= mu_vals[i + 1] for i in range(len(mu_vals) - 1))


# ---- Temperature factor tests ----

def test_temperature_factor_at_reference(raw):
    """Temperature factor should be 1.0 at reference temperature."""
    f_T = raw.temperature_factor(raw.T_ref)
    assert abs(f_T - 1.0) < 1e-10


def test_temperature_factor_decreases_below_ref(raw):
    """Lower temperature should give f_T < 1 (slower kinetics)."""
    f_T = raw.temperature_factor(293.15)  # 20 C
    assert f_T < 1.0, f"f_T at 20C = {f_T}, should be < 1"


def test_temperature_factor_increases_above_ref(raw):
    """Higher temperature should give f_T > 1 (faster kinetics)."""
    f_T = raw.temperature_factor(318.15)  # 45 C
    assert f_T > 1.0, f"f_T at 45C = {f_T}, should be > 1"


# ---- pH inhibition tests ----

def test_ph_optimal(raw):
    """pH inhibition factor should be ~1.0 at optimal pH."""
    f = raw.ph_inhibition(raw.pH_opt)
    assert abs(f - 1.0) < 1e-10


def test_ph_inhibition_low(raw):
    """pH far below optimal should inhibit growth."""
    f = raw.ph_inhibition(5.0)
    assert f < 0.5, f"f_pH at pH=5.0 = {f}, should be < 0.5"


def test_ph_inhibition_high(raw):
    """pH far above optimal should inhibit growth."""
    f = raw.ph_inhibition(9.5)
    assert f < 0.5, f"f_pH at pH=9.5 = {f}, should be < 0.5"


def test_ph_inhibition_symmetric_trend(raw):
    """pH inhibition should be roughly symmetric around optimum."""
    f_low = raw.ph_inhibition(6.0)
    f_high = raw.ph_inhibition(8.0)
    # Both should be reduced
    assert f_low < 1.0
    assert f_high < 1.0


# ---- Steady-state tests ----

def test_steady_state_cod_removal(model):
    """COD removal should be > 50% under default conditions."""
    ss = model.predict_steady_state()
    assert ss["COD_removal_pct"] > 50, \
        f"COD removal = {ss['COD_removal_pct']:.1f}%, expected > 50%"


def test_steady_state_positive_biomass(model):
    """Biomass concentration should be positive at steady state."""
    ss = model.predict_steady_state()
    assert ss["X_ss"] > 0, f"X_ss = {ss['X_ss']}, should be > 0"


def test_steady_state_methane_production(model):
    """Methane production should be positive."""
    ss = model.predict_steady_state()
    assert ss["V_ch4_d"] > 0, f"V_ch4 = {ss['V_ch4_d']:.0f} L/d, should be > 0"


def test_steady_state_substrate_less_than_influent(model):
    """Effluent substrate must be less than influent."""
    ss = model.predict_steady_state()
    assert ss["S_ss"] < 40.0, f"S_ss = {ss['S_ss']}, should be < S_in=40"


def test_steady_state_no_washout_default(model):
    """Default conditions should not cause washout."""
    ss = model.predict_steady_state()
    assert ss["washout"] is False


def test_washout_at_short_hrt(model):
    """Very short HRT should cause washout."""
    ss = model.predict_steady_state({"HRT": 1.0})  # 1 day
    assert ss["washout"] is True, "HRT=1d should cause washout"


def test_higher_sin_more_methane(model):
    """Higher influent COD should produce more methane."""
    ss_low = model.predict_steady_state({"S_in": 20.0})
    ss_high = model.predict_steady_state({"S_in": 60.0})
    assert ss_high["V_ch4_d"] > ss_low["V_ch4_d"]


def test_longer_hrt_better_removal(model):
    """Longer HRT should give better COD removal (up to a point)."""
    ss_short = model.predict_steady_state({"HRT": 10.0})
    ss_long = model.predict_steady_state({"HRT": 30.0})
    assert ss_long["COD_removal_pct"] > ss_short["COD_removal_pct"]


# ---- Dynamic simulation tests ----

def test_simulation_startup(model):
    """Startup from low biomass should show biomass growth."""
    r = model.predict({"x0": [40.0, 0.5], "dt": 0.5, "duration_d": 60.0})
    assert r["X"][-1] > r["X"][0], "Biomass should grow during startup"


def test_simulation_approaches_steady_state(model):
    """Long simulation should approach analytical steady state."""
    ss = model.predict_steady_state()
    # Start closer to steady state so convergence is within reach
    r = model.predict({
        "dt": 0.5, "duration_d": 200.0,
        "x0": [ss["S_ss"] * 1.5, ss["X_ss"] * 0.5],
    })
    # Allow 10% tolerance
    if not ss["washout"]:
        assert abs(r["S"][-1] - ss["S_ss"]) / max(ss["S_ss"], 0.1) < 0.10, \
            f"S final={r['S'][-1]:.2f}, SS={ss['S_ss']:.2f}"
        assert abs(r["X"][-1] - ss["X_ss"]) / ss["X_ss"] < 0.10, \
            f"X final={r['X'][-1]:.2f}, SS={ss['X_ss']:.2f}"


def test_simulation_methane_increases(model):
    """Cumulative methane should be monotonically increasing."""
    r = model.predict({"dt": 0.5, "duration_d": 60.0})
    V_cum = r["V_ch4_cumulative_L"]
    diffs = np.diff(V_cum)
    assert np.all(diffs >= -1e-6), "Cumulative CH4 should be non-decreasing"


def test_simulation_substrate_nonnegative(model):
    """Substrate concentration must never go negative."""
    r = model.predict({"dt": 0.1, "duration_d": 60.0})
    assert np.all(r["S"] >= 0), f"Negative substrate: min={r['S'].min()}"


def test_simulation_biomass_nonnegative(model):
    """Biomass concentration must never go negative."""
    r = model.predict({"dt": 0.1, "duration_d": 60.0})
    assert np.all(r["X"] >= 0), f"Negative biomass: min={r['X'].min()}"


def test_simulation_output_keys(model):
    """Check all expected output keys."""
    r = model.predict({"dt": 1.0, "duration_d": 10.0})
    for key in ["t", "S", "X", "V_ch4_rate_L_d", "V_ch4_cumulative_L",
                "COD_removal_pct", "mu_eff"]:
        assert key in r, f"Missing key: {key}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC140"
    assert info["fidelity"] == "F2a"


# ---- Shock load test ----

def test_shock_load_recovery(model):
    """After a sudden increase in S_in, system should eventually stabilize."""
    def S_in_shock(t):
        return 80.0 if 20.0 < t < 25.0 else 40.0

    r = model.predict({
        "S_in": S_in_shock, "dt": 0.5, "duration_d": 100.0,
        "x0": [5.0, 10.0],  # start near steady state
    })
    # System should not crash (X > 0) and should recover
    assert r["X"][-1] > 0, "Biomass should survive shock load"
    assert r["S"][-1] < 40.0, "Substrate should be consumed after recovery"


# ---- Temperature effect test ----

def test_cold_temperature_slower(model):
    """Cold temperature should give slower kinetics and less methane."""
    ss_warm = model.predict_steady_state({"T": 308.15})  # 35 C
    ss_cold = model.predict_steady_state({"T": 298.15})  # 25 C
    assert ss_warm["V_ch4_d"] > ss_cold["V_ch4_d"], \
        "Warm should produce more CH4 than cold"


# ---- Benchmark ----

def test_benchmark(model):
    """60-day simulation should complete in < 5s."""
    start = time.perf_counter()
    model.predict({"dt": 0.1, "duration_d": 60.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 60-day sim in {elapsed * 1000:.1f} ms")
    assert elapsed < 5.0
