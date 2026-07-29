"""EC176 — PMSM — F1b Thermal — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"torque": 16.0, "speed_rpm": 3000.0})
    for k in ["efficiency", "output_power_kw", "input_power_kw", "total_losses_kw",
              "torque_Nm", "back_emf_V", "derating_factor", "demag_risk"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC176"
    assert info["fidelity"] == "F1b"


# --- Physics Sanity ---

def test_efficiency_bounded(model):
    T = np.random.uniform(1, 25, 500)
    omega = np.random.uniform(100, 12000, 500)
    r = model.predict({"torque": T, "speed_rpm": omega, "magnet_temperature": 80.0})
    assert np.all(r["efficiency"] >= 0.0)
    assert np.all(r["efficiency"] <= 1.0 + 1e-9)


def test_efficiency_at_zero_torque(model):
    r = model.predict({"torque": 0.0, "speed_rpm": 3000.0})
    assert float(r["efficiency"]) == 0.0


def test_efficiency_at_zero_speed(model):
    r = model.predict({"torque": 16.0, "speed_rpm": 0.0})
    assert float(r["efficiency"]) == 0.0


def test_power_balance(model):
    T = np.linspace(1, 25, 50)
    omega = np.linspace(500, 6000, 50)
    r = model.predict({"torque": T, "speed_rpm": omega, "magnet_temperature": 80.0})
    residual = np.abs(r["input_power_kw"] - r["output_power_kw"] - r["total_losses_kw"])
    assert np.all(residual < 1e-6), f"Max power balance error: {np.max(residual):.2e} kW"


def test_losses_positive(model):
    r = model.predict({"torque": 16.0, "speed_rpm": 3000.0})
    assert float(r["total_losses_kw"]) > 0


# --- Thermal Monotonicity ---

def test_efficiency_decreases_with_magnet_temperature(model):
    """Higher magnet temp -> less flux -> more current -> lower efficiency."""
    temps = [25, 60, 80, 120, 150]
    etas = []
    for T_mag in temps:
        r = model.predict({"torque": 16.0, "speed_rpm": 3000.0,
                           "magnet_temperature": T_mag})
        etas.append(float(r["efficiency"]))
    for i in range(len(etas) - 1):
        assert etas[i] > etas[i + 1], (
            f"eta should decrease: eta({temps[i]}C)={etas[i]:.4f} "
            f">= eta({temps[i+1]}C)={etas[i+1]:.4f}"
        )


def test_back_emf_decreases_with_temperature(model):
    """Back-EMF proportional to Phi_m, which decreases with temperature."""
    emf_25 = float(model.predict({"torque": 10.0, "speed_rpm": 3000.0,
                                   "magnet_temperature": 25.0})["back_emf_V"])
    emf_150 = float(model.predict({"torque": 10.0, "speed_rpm": 3000.0,
                                    "magnet_temperature": 150.0})["back_emf_V"])
    assert emf_25 > emf_150, "Back-EMF should decrease with magnet temperature"


def test_flux_decreases_with_temperature(model):
    """PM flux decreases linearly with temperature for NdFeB."""
    from model import PMSMF1b
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = PMSMF1b(params)
    assert m.flux(25.0) > m.flux(150.0)


# --- Demagnetization ---

def test_no_demag_risk_below_threshold(model):
    r = model.predict({"torque": 16.0, "speed_rpm": 3000.0,
                       "magnet_temperature": 100.0})
    assert not bool(r["demag_risk"])


def test_demag_risk_above_threshold(model):
    r = model.predict({"torque": 16.0, "speed_rpm": 3000.0,
                       "magnet_temperature": 160.0})
    assert bool(r["demag_risk"])


def test_derating_below_demag_threshold(model):
    r = model.predict({"torque": 16.0, "speed_rpm": 3000.0,
                       "magnet_temperature": 100.0, "ambient_temperature": 25.0})
    assert float(r["derating_factor"]) == 1.0


def test_derating_above_demag_threshold(model):
    r = model.predict({"torque": 16.0, "speed_rpm": 3000.0,
                       "magnet_temperature": 160.0, "ambient_temperature": 25.0})
    assert float(r["derating_factor"]) < 1.0


# --- Back-EMF physics ---

def test_back_emf_proportional_to_speed(model):
    """At constant temperature, back-EMF is linear in speed."""
    r1 = model.predict({"torque": 10.0, "speed_rpm": 1000.0, "magnet_temperature": 80.0})
    r2 = model.predict({"torque": 10.0, "speed_rpm": 2000.0, "magnet_temperature": 80.0})
    ratio = float(r2["back_emf_V"]) / float(r1["back_emf_V"])
    assert abs(ratio - 2.0) < 0.01, f"EMF ratio should be 2.0, got {ratio:.3f}"


# --- Edge Cases ---

def test_extreme_magnet_temperature(model):
    r = model.predict({"torque": 10.0, "speed_rpm": 3000.0, "magnet_temperature": 180.0})
    assert 0 <= float(r["efficiency"]) <= 1.0


def test_vectorized(model):
    T = np.linspace(1, 25, 50)
    omega = np.linspace(500, 6000, 50)
    T_mag = np.linspace(25, 150, 50)
    r = model.predict({"torque": T, "speed_rpm": omega, "magnet_temperature": T_mag})
    assert len(r["efficiency"]) == 50


def test_benchmark(model):
    T = np.random.uniform(1, 25, 1000)
    omega = np.random.uniform(100, 12000, 1000)
    start = time.perf_counter()
    model.predict({"torque": T, "speed_rpm": omega, "magnet_temperature": 80.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
