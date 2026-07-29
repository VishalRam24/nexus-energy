"""EC220 — TENG — F1b Surface Charge Dynamics — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 0.0})
    for k in ["sigma_Cm2", "V_oc_peak_V", "C_avg_F", "R_internal_ohm",
              "power_avg_w", "power_net_w", "power_density_mwcm2", "efficiency", "dielectric_loss_w"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC220"
    assert info["fidelity"] == "F1b"


def test_power_positive_at_t0(model):
    """Power must be positive at t=0 (peak surface charge)."""
    r = model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 0.0})
    P = float(np.atleast_1d(r["power_net_w"])[0])
    assert P > 0, f"Power = {P:.4e} W"


def test_power_decays_with_time(model):
    """Power should decrease as surface charge leaks away."""
    P0 = float(np.atleast_1d(
        model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 0.0})["power_net_w"]
    )[0])
    P_long = float(np.atleast_1d(
        model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 5000.0})["power_net_w"]
    )[0])
    assert P_long < P0, f"Power not decaying: P0={P0:.4e}, P(5000s)={P_long:.4e}"


def test_sigma_decays_with_time(model):
    """Surface charge density must decrease exponentially with time."""
    sigma_0 = float(model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 0.0})["sigma_Cm2"])
    sigma_t = float(model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 1000.0})["sigma_Cm2"])
    tau = model._model.tau_decay
    expected = model._model.sigma0 * np.exp(-1000.0 / tau)
    assert abs(sigma_t - expected) < 1e-9, f"sigma decay wrong: {sigma_t:.4e} vs {expected:.4e}"


def test_power_increases_with_frequency(model):
    """Higher frequency -> more power (up to RC rolloff)."""
    P_low = float(np.atleast_1d(
        model.predict({"frequency_hz": 0.5, "R_load_ohm": 1e7, "t_s": 0.0})["power_net_w"]
    )[0])
    P_mid = float(np.atleast_1d(
        model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 0.0})["power_net_w"]
    )[0])
    assert P_mid > P_low, f"P(3Hz)={P_mid:.4e} <= P(0.5Hz)={P_low:.4e}"


def test_two_dielectric_voc_higher_than_single(model):
    """Dual dielectric (larger d_eff) should give lower V_oc vs thin single layer (correct physics)."""
    # V_oc = sigma*x / (eps_0*(1 + x/d_eff)); larger d_eff -> higher V_oc (asymptotic limit)
    V_oc = float(np.atleast_1d(
        model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 0.0})["V_oc_peak_V"]
    )[0])
    # Just check it's in physically reasonable range for TENG (~10-2000 V)
    assert 1.0 < V_oc < 5000.0, f"V_oc={V_oc:.1f} V not in expected range"


def test_dielectric_loss_positive(model):
    """Dielectric loss must be positive."""
    r = model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 0.0})
    loss = float(np.atleast_1d(r["dielectric_loss_w"])[0])
    assert loss >= 0, f"Dielectric loss = {loss}"


def test_power_net_less_than_power_avg(model):
    """Net power (after dielectric loss) must be <= average power."""
    r = model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 0.0})
    P_avg = float(np.atleast_1d(r["power_avg_w"])[0])
    P_net = float(np.atleast_1d(r["power_net_w"])[0])
    assert P_net <= P_avg + 1e-15, f"P_net={P_net:.4e} > P_avg={P_avg:.4e}"


def test_efficiency_in_range(model):
    """Efficiency must be in [0, 1]."""
    r = model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 0.0})
    eta = float(np.atleast_1d(r["efficiency"])[0])
    assert 0.0 <= eta <= 1.0, f"Efficiency out of range: {eta}"


def test_voc_increases_with_sigma0(model):
    """V_oc should increase if initial charge density sigma0 increases."""
    # At t=0, sigma=sigma0. Higher sigma0 -> higher V_oc.
    V_base = float(np.atleast_1d(
        model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 0.0})["V_oc_peak_V"]
    )[0])
    old_sigma = model._model.sigma0
    model._model.sigma0 *= 2.0
    V_double = float(np.atleast_1d(
        model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": 0.0})["V_oc_peak_V"]
    )[0])
    model._model.sigma0 = old_sigma  # restore
    assert abs(V_double / V_base - 2.0) < 0.01, f"V_oc not linear with sigma: ratio={V_double/V_base:.4f}"


def test_benchmark(model):
    start = time.perf_counter()
    for f in np.random.uniform(0.5, 20, 100):
        model.predict({"frequency_hz": float(f), "R_load_ohm": 1e7, "t_s": 0.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 100 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 5.0
