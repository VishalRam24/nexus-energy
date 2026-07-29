"""EC176 — PMSM — F1a — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"torque": 160.0, "speed_rpm": 3000.0})
    for k in ["efficiency", "output_power_kw", "input_power_kw", "total_losses_kw"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC176"
    assert "efficiency" in info["outputs"]


def test_efficiency_never_exceeds_one(model):
    """Efficiency must always be <= 1."""
    T = np.random.uniform(1, 200, 500)
    omega = np.random.uniform(100, 12000, 500)
    r = model.predict({"torque": T, "speed_rpm": omega})
    assert np.all(r["efficiency"] <= 1.0 + 1e-9)


def test_efficiency_at_zero_torque(model):
    """At T=0, output power is zero, efficiency should be 0."""
    r = model.predict({"torque": 0.0, "speed_rpm": 3000.0})
    assert float(r["efficiency"]) == 0.0
    assert float(r["output_power_kw"]) == 0.0


def test_efficiency_at_zero_speed(model):
    """At omega=0, output power is zero, efficiency should be 0."""
    r = model.predict({"torque": 160.0, "speed_rpm": 0.0})
    assert float(r["efficiency"]) == 0.0


def test_rated_point_efficiency(model):
    """At T_rated=160Nm, omega=3000rpm, eta must be ~0.96."""
    r = model.predict({"torque": 160.0, "speed_rpm": 3000.0})
    eta = float(r["efficiency"])
    assert abs(eta - 0.96) < 0.005, f"Rated-point eta={eta:.4f}, expected ~0.96"


def test_rated_output_power(model):
    """At T=160Nm, 3000rpm: P_out = 160 * 3000 * pi/30 ≈ 50.27 kW."""
    r = model.predict({"torque": 160.0, "speed_rpm": 3000.0})
    p_out = float(r["output_power_kw"])
    assert abs(p_out - 50.27) < 0.5, f"P_out={p_out:.2f}kW, expected ~50.27kW"


def test_losses_positive(model):
    """Total losses must be > 0 whenever the motor is operating."""
    r = model.predict({"torque": 160.0, "speed_rpm": 3000.0})
    assert float(r["total_losses_kw"]) > 0


def test_power_balance(model):
    """P_in = P_out + P_loss."""
    T = np.linspace(10, 200, 50)
    omega = np.linspace(500, 6000, 50)
    r = model.predict({"torque": T, "speed_rpm": omega})
    residual = np.abs(r["input_power_kw"] - r["output_power_kw"] - r["total_losses_kw"])
    assert np.all(residual < 1e-6), f"Max power balance error: {np.max(residual):.2e} kW"


def test_efficiency_peaks_near_rated(model):
    """Peak efficiency over operating range should be near eta_peak=0.96."""
    T = np.linspace(10, 200, 30)
    omega = np.linspace(500, 6000, 30)
    TT, OO = np.meshgrid(T, omega)
    r = model.predict({"torque": TT.ravel(), "speed_rpm": OO.ravel()})
    eta_max = float(np.max(r["efficiency"]))
    assert 0.95 <= eta_max <= 0.98, f"Peak eta={eta_max:.4f} outside expected [0.95, 0.98]"


def test_high_speed_iron_loss_dominates(model):
    """At high speed, iron losses should be significant."""
    T = 10.0  # Low torque → low copper loss
    r_low = model.predict({"torque": T, "speed_rpm": 500.0})
    r_high = model.predict({"torque": T, "speed_rpm": 10000.0})
    # Efficiency at high speed with low torque should be lower
    assert float(r_high["efficiency"]) < float(r_low["efficiency"]), \
        "High-speed iron losses should degrade efficiency at low torque"


def test_benchmark(model):
    T = np.random.uniform(0, 200, 1000)
    omega = np.random.uniform(0, 12000, 1000)
    start = time.perf_counter()
    model.predict({"torque": T, "speed_rpm": omega})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
