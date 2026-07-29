"""EC177 — BLDC Motor — F1b Thermal — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"torque_nm": 3.18, "speed_rpm": 3000.0})
    for k in ["efficiency", "input_power_w", "output_power_w", "losses_w",
              "p_copper_w", "p_iron_w", "p_mech_w", "p_stray_w",
              "phase_current_A", "derating_factor", "demagnetization_risk"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC177"
    assert info["fidelity"] == "F1b"


# --- Physics Sanity ---

def test_efficiency_bounded(model):
    T = np.linspace(0.1, 4.8, 50)
    spd = np.linspace(100, 6000, 50)
    r = model.predict({"torque_nm": T, "speed_rpm": spd})
    assert np.all(r["efficiency"] >= 0.0)
    assert np.all(r["efficiency"] <= 1.0)


def test_power_balance(model):
    T = np.linspace(0.5, 4.8, 40)
    spd = np.full(40, 3000.0)
    r = model.predict({"torque_nm": T, "speed_rpm": spd})
    diff = np.abs(r["input_power_w"] - r["output_power_w"] - r["losses_w"])
    assert np.all(diff < 1e-9), "Power balance violated"


def test_losses_positive(model):
    r = model.predict({"torque_nm": 3.0, "speed_rpm": 3000.0})
    assert float(r["losses_w"]) > 0.0
    assert float(r["p_copper_w"]) >= 0.0
    assert float(r["p_iron_w"]) >= 0.0


def test_loss_components_sum(model):
    """Loss components must sum to total losses."""
    r = model.predict({"torque_nm": 3.18, "speed_rpm": 3000.0})
    total = (float(r["p_copper_w"]) + float(r["p_iron_w"]) +
             float(r["p_mech_w"]) + float(r["p_stray_w"]))
    assert abs(total - float(r["losses_w"])) < 1e-9


# --- Thermal Monotonicity ---

def test_efficiency_decreases_with_temperature(model):
    """Higher magnet/winding temperature -> lower efficiency."""
    temps = [25, 50, 80, 110, 140]
    etas = []
    for T in temps:
        r = model.predict({"torque_nm": 3.18, "speed_rpm": 3000.0,
                           "magnet_temperature": T, "winding_temperature": T})
        etas.append(float(r["efficiency"]))
    for i in range(len(etas) - 1):
        assert etas[i] >= etas[i + 1], (
            f"eta should not increase with T: eta({temps[i]}C)={etas[i]:.4f}, "
            f"eta({temps[i+1]}C)={etas[i+1]:.4f}"
        )


def test_copper_loss_increases_with_winding_temperature(model):
    """Higher winding temperature -> higher R_s -> more copper loss."""
    r25 = model.predict({"torque_nm": 3.18, "speed_rpm": 3000.0,
                         "winding_temperature": 25.0, "magnet_temperature": 80.0})
    r150 = model.predict({"torque_nm": 3.18, "speed_rpm": 3000.0,
                          "winding_temperature": 150.0, "magnet_temperature": 80.0})
    assert float(r150["p_copper_w"]) > float(r25["p_copper_w"])


def test_current_increases_with_demagnetization(model):
    """Lower k_t at high T_magnet -> higher current for same torque."""
    I_cold = float(model.predict({"torque_nm": 3.18, "speed_rpm": 3000.0,
                                   "magnet_temperature": 25.0})["phase_current_A"])
    I_hot = float(model.predict({"torque_nm": 3.18, "speed_rpm": 3000.0,
                                  "magnet_temperature": 140.0})["phase_current_A"])
    assert I_hot > I_cold, "Higher T_magnet should increase current demand"


# --- Derating ---

def test_derating_unity_below_threshold(model):
    r = model.predict({"torque_nm": 3.18, "speed_rpm": 3000.0, "ambient_temperature": 25.0})
    assert float(r["derating_factor"]) == 1.0


def test_derating_below_unity_above_threshold(model):
    r = model.predict({"torque_nm": 3.18, "speed_rpm": 3000.0, "ambient_temperature": 55.0})
    assert float(r["derating_factor"]) < 1.0


def test_derating_monotonic(model):
    from model import BLDCMotorF1b
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = BLDCMotorF1b(params)
    T_a = np.array([20.0, 40.0, 45.0, 50.0, 60.0])
    d = m.derating_factor(T_a)
    assert np.all(np.diff(d) <= 0)


# --- Demagnetization ---

def test_demagnetization_flag(model):
    r_safe = model.predict({"torque_nm": 1.0, "speed_rpm": 1000.0, "magnet_temperature": 100.0})
    r_demag = model.predict({"torque_nm": 1.0, "speed_rpm": 1000.0, "magnet_temperature": 155.0})
    assert not bool(r_safe["demagnetization_risk"])
    assert bool(r_demag["demagnetization_risk"])


# --- Iron loss speed scaling ---

def test_iron_loss_increases_with_speed(model):
    r_low = model.predict({"torque_nm": 1.0, "speed_rpm": 500.0})
    r_high = model.predict({"torque_nm": 1.0, "speed_rpm": 5000.0})
    assert float(r_high["p_iron_w"]) > float(r_low["p_iron_w"])


# --- Zero speed ---

def test_zero_speed(model):
    """At zero speed: iron and mechanical losses must be zero."""
    r = model.predict({"torque_nm": 1.0, "speed_rpm": 0.0})
    assert float(r["p_iron_w"]) == 0.0
    assert float(r["p_mech_w"]) == 0.0
    assert float(r["output_power_w"]) == 0.0


# --- Vectorized ---

def test_vectorized(model):
    T = np.linspace(0.1, 4.8, 100)
    spd = np.linspace(100, 6000, 100)
    r = model.predict({"torque_nm": T, "speed_rpm": spd})
    assert len(r["efficiency"]) == 100


# --- Benchmark ---

def test_benchmark(model):
    T = np.random.uniform(0.1, 4.8, 1000)
    spd = np.random.uniform(100, 6000, 1000)
    start = time.perf_counter()
    model.predict({"torque_nm": T, "speed_rpm": spd})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
