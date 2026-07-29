"""EC178 — SRM — F1b Thermal — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"torque_nm": 9.55, "speed_rpm": 3000.0})
    for k in ["efficiency", "input_power_w", "output_power_w", "losses_w",
              "p_copper_w", "p_iron_w", "p_mech_w", "p_stray_w",
              "phase_current_A", "derating_factor"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC178"
    assert info["fidelity"] == "F1b"


def test_efficiency_bounded(model):
    T = np.linspace(0.1, 14.0, 50)
    spd = np.linspace(100, 9000, 50)
    r = model.predict({"torque_nm": T, "speed_rpm": spd})
    assert np.all(r["efficiency"] >= 0.0)
    assert np.all(r["efficiency"] <= 1.0)


def test_power_balance(model):
    T = np.linspace(1.0, 14.0, 40)
    spd = np.full(40, 3000.0)
    r = model.predict({"torque_nm": T, "speed_rpm": spd})
    diff = np.abs(r["input_power_w"] - r["output_power_w"] - r["losses_w"])
    assert np.all(diff < 1e-9), "Power balance violated"


def test_loss_components_sum(model):
    r = model.predict({"torque_nm": 9.55, "speed_rpm": 3000.0})
    total = (float(r["p_copper_w"]) + float(r["p_iron_w"]) +
             float(r["p_mech_w"]) + float(r["p_stray_w"]))
    assert abs(total - float(r["losses_w"])) < 1e-9


def test_efficiency_decreases_with_temperature(model):
    temps = [25, 60, 90, 120, 155]
    etas = []
    for T_w in temps:
        r = model.predict({"torque_nm": 9.55, "speed_rpm": 3000.0,
                           "winding_temperature": T_w})
        etas.append(float(r["efficiency"]))
    for i in range(len(etas) - 1):
        assert etas[i] >= etas[i + 1], (
            f"eta should not increase with T_w: "
            f"eta({temps[i]}C)={etas[i]:.4f} vs eta({temps[i+1]}C)={etas[i+1]:.4f}"
        )


def test_copper_loss_increases_with_temperature(model):
    r25 = model.predict({"torque_nm": 9.55, "speed_rpm": 3000.0, "winding_temperature": 25.0})
    r155 = model.predict({"torque_nm": 9.55, "speed_rpm": 3000.0, "winding_temperature": 155.0})
    assert float(r155["p_copper_w"]) > float(r25["p_copper_w"])


def test_iron_loss_increases_with_speed(model):
    r_low = model.predict({"torque_nm": 5.0, "speed_rpm": 500.0})
    r_high = model.predict({"torque_nm": 5.0, "speed_rpm": 5000.0})
    assert float(r_high["p_iron_w"]) > float(r_low["p_iron_w"])


def test_zero_speed_no_iron_mech(model):
    r = model.predict({"torque_nm": 5.0, "speed_rpm": 0.0})
    assert float(r["p_iron_w"]) == 0.0
    assert float(r["p_mech_w"]) == 0.0
    assert float(r["output_power_w"]) == 0.0


def test_losses_positive(model):
    r = model.predict({"torque_nm": 9.0, "speed_rpm": 3000.0})
    assert float(r["losses_w"]) > 0.0


def test_derating_unity_below_threshold(model):
    r = model.predict({"torque_nm": 9.55, "speed_rpm": 3000.0, "ambient_temperature": 30.0})
    assert float(r["derating_factor"]) == 1.0


def test_derating_below_unity_above_threshold(model):
    r = model.predict({"torque_nm": 9.55, "speed_rpm": 3000.0, "ambient_temperature": 55.0})
    assert float(r["derating_factor"]) < 1.0


def test_vectorized(model):
    T = np.linspace(0.5, 14.0, 100)
    spd = np.linspace(100, 8000, 100)
    r = model.predict({"torque_nm": T, "speed_rpm": spd})
    assert len(r["efficiency"]) == 100


def test_benchmark(model):
    T = np.random.uniform(0.5, 14.0, 1000)
    spd = np.random.uniform(100, 9000, 1000)
    start = time.perf_counter()
    model.predict({"torque_nm": T, "speed_rpm": spd})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
