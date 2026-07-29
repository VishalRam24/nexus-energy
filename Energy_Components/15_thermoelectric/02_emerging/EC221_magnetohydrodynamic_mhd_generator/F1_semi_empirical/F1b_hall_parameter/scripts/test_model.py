"""EC221 — MHD Generator — F1b Hall Parameter — Test Suite

Physics covered:
  1. Output keys present
  2. get_info returns correct ec_id / fidelity
  3. Stagnation Q_in >> kinetic-only Q_in (cp*T >> 0.5*u^2 for hot plasma)
  4. P_elec < Q_in at all points (first law)
  5. eta_electric >= 0 and <= 1 everywhere
  6. Hall factor reduces power vs beta=0 case
  7. Power scales as P ~ sigma_eff * u^2 * B^2
  8. P ~ u^2 scaling (K, B, beta fixed)
  9. P ~ B^2 scaling (K, u, beta fixed)
 10. eta_hall = 1/(1+beta^2) formula
 11. Max power still at K=0.5 (Hall does not shift K_opt analytically)
 12. sigma_eff decreases with increasing beta
 13. sigma_actual increases with plasma temperature (T-dependent sigma)
 14. J_hall = beta * J_faraday
 15. Benchmark: 1000 predictions < 1 s
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5, "beta": 3.0})
    expected = [
        "EMF_V", "J_Am2", "J_hall_Am2", "sigma_actual_Sm", "sigma_eff_Sm",
        "power_density_Wm3", "power_elec_W", "heat_input_stag_W",
        "eta_mhd", "eta_hall", "eta_electric", "K_optimal",
    ]
    for k in expected:
        assert k in r, f"Missing output key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC221"
    assert info["fidelity"] == "F1b"


def test_stagnation_Q_in_larger_than_kinetic(model):
    """Q_stag = rho*u*(cp*T + 0.5*u^2)*A >> 0.5*rho*u^3*A for hot plasma.
    At T=2500K, cp=1200: cp*T = 3e6 J/kg vs 0.5*u^2 = 3.2e5 J/kg (~10x smaller).
    Stagnation Q_in should be at least 5x larger than kinetic-only estimate.
    """
    r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5,
                       "beta": 3.0, "T_plasma_K": 2500.0})
    Q_stag = float(r["heat_input_stag_W"])
    # kinetic-only baseline
    params = model.params["unit"]
    rho = params["rho_plasma"]["value"]
    w = params["channel_width"]["value"]
    h = params["channel_height"]["value"]
    Q_kinetic = 0.5 * rho * 800.0 ** 3 * w * h
    assert Q_stag > 5.0 * Q_kinetic, \
        f"Q_stag={Q_stag:.3e} should be >5x kinetic Q={Q_kinetic:.3e}"


def test_first_law_P_less_than_Q(model):
    """Electrical power must never exceed heat input (first law)."""
    u_arr = np.linspace(200.0, 1500.0, 50)
    r = model.predict({"sigma": 10.0, "u": u_arr, "B": 5.0, "K": 0.5, "beta": 3.0})
    assert np.all(np.asarray(r["power_elec_W"]) < np.asarray(r["heat_input_stag_W"])), \
        "P_elec must always be < Q_in (first law)"


def test_eta_electric_bounds(model):
    """eta_electric must be in [0, 1] at all operating points."""
    u_arr = np.linspace(200.0, 1500.0, 30)
    B_arr = np.linspace(1.0, 8.0, 30)
    for K in [0.1, 0.5, 0.9]:
        r = model.predict({"sigma": 10.0, "u": u_arr, "B": 5.0, "K": K, "beta": 3.0})
        eta = np.asarray(r["eta_electric"])
        assert np.all(eta >= 0.0), f"eta_electric < 0 at K={K}"
        assert np.all(eta <= 1.0), f"eta_electric > 1 at K={K}"


def test_hall_reduces_power(model):
    """Higher Hall parameter reduces effective conductivity and thus output power."""
    r_no_hall = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5, "beta": 0.0})
    r_with_hall = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5, "beta": 3.0})
    P_no = float(r_no_hall["power_elec_W"])
    P_w = float(r_with_hall["power_elec_W"])
    assert P_w < P_no, f"Hall should reduce power: P_beta=3={P_w:.2f} vs P_beta=0={P_no:.2f}"


def test_eta_hall_formula(model):
    """eta_hall = 1 / (1 + beta^2) — check numerically for several beta values."""
    for beta in [0.0, 1.0, 2.0, 3.0, 5.0]:
        r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5, "beta": beta})
        eta_hall_computed = float(np.atleast_1d(r["eta_hall"])[0])
        eta_hall_expected = 1.0 / (1.0 + beta ** 2)
        assert abs(eta_hall_computed - eta_hall_expected) < 1e-9, \
            f"eta_hall mismatch at beta={beta}: {eta_hall_computed:.6f} vs {eta_hall_expected:.6f}"


def test_sigma_eff_decreases_with_beta(model):
    """sigma_eff = sigma / (1 + beta^2) must decrease as beta increases."""
    betas = [0.0, 1.0, 2.0, 4.0, 8.0]
    sigmas_eff = []
    for b in betas:
        r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5, "beta": b})
        sigmas_eff.append(float(np.atleast_1d(r["sigma_eff_Sm"])[0]))
    assert all(sigmas_eff[i] > sigmas_eff[i + 1] for i in range(len(sigmas_eff) - 1)), \
        f"sigma_eff must decrease with beta: {sigmas_eff}"


def test_power_scales_u_squared(model):
    """P ~ u^2 (Hall-corrected power density still proportional to u^2)."""
    u1, u2 = 400.0, 800.0  # ratio = 2
    r1 = model.predict({"sigma": 10.0, "u": u1, "B": 5.0, "K": 0.5, "beta": 3.0})
    r2 = model.predict({"sigma": 10.0, "u": u2, "B": 5.0, "K": 0.5, "beta": 3.0})
    ratio = float(r2["power_elec_W"]) / float(r1["power_elec_W"])
    assert abs(ratio - 4.0) < 0.01, f"P~u^2 violated: ratio={ratio:.4f}, expected 4.0"


def test_power_scales_B_squared(model):
    """P ~ B^2."""
    B1, B2 = 2.5, 5.0  # ratio = 2
    r1 = model.predict({"sigma": 10.0, "u": 800.0, "B": B1, "K": 0.5, "beta": 3.0})
    r2 = model.predict({"sigma": 10.0, "u": 800.0, "B": B2, "K": 0.5, "beta": 3.0})
    ratio = float(r2["power_elec_W"]) / float(r1["power_elec_W"])
    assert abs(ratio - 4.0) < 0.01, f"P~B^2 violated: ratio={ratio:.4f}, expected 4.0"


def test_max_power_at_K_half(model):
    """Maximum power still at K=0.5 (Hall effect does not shift K_opt analytically)."""
    K_range = np.linspace(0.01, 0.99, 200)
    r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": K_range, "beta": 3.0})
    idx_max = np.argmax(np.asarray(r["power_elec_W"]))
    K_at_max = K_range[idx_max]
    assert abs(K_at_max - 0.5) < 0.02, f"Max power at K={K_at_max:.3f}, expected ~0.5"


def test_sigma_T_increases_with_temperature(model):
    """sigma_actual must increase with plasma temperature (positive T exponent)."""
    r_cold = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5,
                            "beta": 3.0, "T_plasma_K": 1800.0})
    r_hot = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5,
                           "beta": 3.0, "T_plasma_K": 3200.0})
    s_cold = float(np.atleast_1d(r_cold["sigma_actual_Sm"])[0])
    s_hot = float(np.atleast_1d(r_hot["sigma_actual_Sm"])[0])
    assert s_hot > s_cold, \
        f"sigma should increase with T: sigma(3200K)={s_hot:.3f} vs sigma(1800K)={s_cold:.3f}"


def test_J_hall_equals_beta_times_J_faraday(model):
    """J_hall = beta * J_faraday — check for several beta values."""
    for beta in [0.5, 1.0, 2.0, 4.0]:
        r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5, "beta": beta})
        J = float(np.atleast_1d(r["J_Am2"])[0])
        J_hall = float(np.atleast_1d(r["J_hall_Am2"])[0])
        assert abs(J_hall - beta * J) < 1e-6, \
            f"J_hall/J != beta at beta={beta}: J_hall={J_hall:.4f}, beta*J={beta*J:.4f}"


def test_Q_in_scales_with_velocity(model):
    """Q_stag ~ rho * u * (cp*T + 0.5*u^2) * A — should increase monotonically with u."""
    u_arr = np.linspace(300.0, 1500.0, 50)
    r = model.predict({"sigma": 10.0, "u": u_arr, "B": 5.0, "K": 0.5, "beta": 3.0})
    Q = np.asarray(r["heat_input_stag_W"])
    assert np.all(np.diff(Q) > 0), "Q_stag must increase monotonically with u"


def test_power_zero_at_K_boundaries(model):
    """P=0 at K=0 (short circuit) and K=1 (open circuit)."""
    for K_edge in [0.0, 1.0]:
        r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": K_edge, "beta": 3.0})
        P = float(np.atleast_1d(r["power_elec_W"])[0])
        assert P == pytest.approx(0.0, abs=1.0), f"P should be ~0 at K={K_edge}, got {P:.3f}"


def test_benchmark(model):
    sigma = np.random.uniform(1.0, 50.0, 1000)
    u = np.random.uniform(200.0, 1500.0, 1000)
    B = np.random.uniform(1.0, 8.0, 1000)
    K = np.random.uniform(0.1, 0.9, 1000)
    beta = np.random.uniform(0.0, 8.0, 1000)
    start = time.perf_counter()
    model.predict({"sigma": sigma, "u": u, "B": B, "K": K, "beta": beta})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 1.0
