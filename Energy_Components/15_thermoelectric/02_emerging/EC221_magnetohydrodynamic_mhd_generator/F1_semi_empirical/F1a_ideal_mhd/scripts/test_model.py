"""EC221 — MHD Generator — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5})
    for k in ["EMF_V", "J_Am2", "power_density_Wm3", "power_w", "heat_input_w", "eta_mhd", "eta_plant"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC221"
    assert info["fidelity"] == "F1a"


def test_max_power_at_K_half(model):
    """P = sigma*u^2*B^2*K*(1-K)*V — maximum at K=0.5."""
    K_range = np.linspace(0.01, 0.99, 200)
    r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": K_range})
    idx_max = np.argmax(r["power_w"])
    K_at_max = K_range[idx_max]
    assert abs(K_at_max - 0.5) < 0.02, \
        f"Max power at K={K_at_max:.3f}, expected K≈0.5"


def test_power_zero_at_K_zero(model):
    """K=0 (short circuit): no useful power."""
    r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.0})
    assert float(r["power_w"]) == pytest.approx(0.0, abs=1.0)


def test_power_zero_at_K_one(model):
    """K=1 (open circuit): no current, no power."""
    r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 1.0})
    assert float(r["power_w"]) == pytest.approx(0.0, abs=1.0)


def test_power_scales_with_u_squared(model):
    """P ~ u^2 at fixed K, sigma, B."""
    u1, u2 = 500.0, 1000.0  # ratio = 2
    r1 = model.predict({"sigma": 10.0, "u": u1, "B": 5.0, "K": 0.5})
    r2 = model.predict({"sigma": 10.0, "u": u2, "B": 5.0, "K": 0.5})
    ratio = float(r2["power_w"]) / float(r1["power_w"])
    assert abs(ratio - 4.0) < 0.01, \
        f"P~u^2 violated: ratio={ratio:.3f}, expected 4.0"


def test_power_scales_with_B_squared(model):
    """P ~ B^2 at fixed K, sigma, u."""
    B1, B2 = 3.0, 6.0  # ratio = 2
    r1 = model.predict({"sigma": 10.0, "u": 800.0, "B": B1, "K": 0.5})
    r2 = model.predict({"sigma": 10.0, "u": 800.0, "B": B2, "K": 0.5})
    ratio = float(r2["power_w"]) / float(r1["power_w"])
    assert abs(ratio - 4.0) < 0.01, \
        f"P~B^2 violated: ratio={ratio:.3f}, expected 4.0"


def test_eta_mhd_max_is_0_25(model):
    """Maximum MHD efficiency at K=0.5 is exactly 0.25 (K*(1-K) = 0.25)."""
    r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5})
    assert float(r["eta_mhd"]) == pytest.approx(0.25, abs=1e-9)


def test_energy_conservation(model):
    """P_out must be less than heat input."""
    r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5})
    assert float(r["power_w"]) < float(r["heat_input_w"]), \
        "Power output cannot exceed heat input"


def test_emf_scales_with_u_and_B(model):
    """EMF = u*B*h — must scale linearly with u and B."""
    r1 = model.predict({"sigma": 10.0, "u": 400.0, "B": 5.0, "K": 0.5})
    r2 = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5})
    ratio_u = float(r2["EMF_V"]) / float(r1["EMF_V"])
    assert abs(ratio_u - 2.0) < 0.01, f"EMF~u violated: ratio={ratio_u:.3f}"

    r3 = model.predict({"sigma": 10.0, "u": 800.0, "B": 10.0, "K": 0.5})
    ratio_B = float(r3["EMF_V"]) / float(r2["EMF_V"])
    assert abs(ratio_B - 2.0) < 0.01, f"EMF~B violated: ratio={ratio_B:.3f}"


def test_benchmark(model):
    sigma = np.random.uniform(1.0, 50.0, 1000)
    u = np.random.uniform(200.0, 1500.0, 1000)
    B = np.random.uniform(1.0, 8.0, 1000)
    K = np.random.uniform(0.1, 0.9, 1000)
    start = time.perf_counter()
    model.predict({"sigma": sigma, "u": u, "B": B, "K": K})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
