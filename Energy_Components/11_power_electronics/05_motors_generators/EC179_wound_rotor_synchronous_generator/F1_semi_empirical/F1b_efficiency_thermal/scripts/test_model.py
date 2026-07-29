"""EC179 — WRSG — F1b Thermal — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"load_fraction": 1.0})
    for k in ["efficiency", "input_power_w", "output_power_w", "losses_w",
              "p_stator_cu_w", "p_rotor_cu_w", "p_iron_w", "p_mech_w", "p_stray_w",
              "stator_current_A", "field_current_A", "derating_factor"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC179"
    assert info["fidelity"] == "F1b"


def test_efficiency_bounded(model):
    plr = np.linspace(0.05, 1.2, 50)
    r = model.predict({"load_fraction": plr})
    assert np.all(r["efficiency"] >= 0.0)
    assert np.all(r["efficiency"] <= 1.0)


def test_power_balance(model):
    plr = np.linspace(0.1, 1.2, 40)
    r = model.predict({"load_fraction": plr})
    diff = np.abs(r["input_power_w"] - r["output_power_w"] - r["losses_w"])
    assert np.all(diff < 1e-6), "Power balance violated"


def test_loss_components_sum(model):
    r = model.predict({"load_fraction": 1.0})
    total = (float(r["p_stator_cu_w"]) + float(r["p_rotor_cu_w"]) +
             float(r["p_iron_w"]) + float(r["p_mech_w"]) + float(r["p_stray_w"]))
    assert abs(total - float(r["losses_w"])) < 1e-9


def test_losses_positive(model):
    r = model.predict({"load_fraction": 1.0})
    assert float(r["losses_w"]) > 0.0
    assert float(r["p_iron_w"]) > 0.0


def test_efficiency_decreases_with_temperature(model):
    temps = [25, 60, 90, 120, 155]
    etas = []
    for T in temps:
        r = model.predict({"load_fraction": 1.0, "stator_temperature": T, "rotor_temperature": T})
        etas.append(float(r["efficiency"]))
    for i in range(len(etas) - 1):
        assert etas[i] >= etas[i + 1]


def test_stator_copper_increases_with_temperature(model):
    r25 = model.predict({"load_fraction": 1.0, "stator_temperature": 25.0})
    r155 = model.predict({"load_fraction": 1.0, "stator_temperature": 155.0})
    assert float(r155["p_stator_cu_w"]) > float(r25["p_stator_cu_w"])


def test_rotor_copper_increases_with_temperature(model):
    r25 = model.predict({"load_fraction": 1.0, "rotor_temperature": 25.0})
    r155 = model.predict({"load_fraction": 1.0, "rotor_temperature": 155.0})
    assert float(r155["p_rotor_cu_w"]) > float(r25["p_rotor_cu_w"])


def test_iron_loss_constant_vs_load(model):
    """Iron loss must be the same at 50% and 100% load (synchronous speed is constant)."""
    r50 = model.predict({"load_fraction": 0.5})
    r100 = model.predict({"load_fraction": 1.0})
    assert abs(float(r50["p_iron_w"]) - float(r100["p_iron_w"])) < 1e-9


def test_stator_current_scales_with_load(model):
    """Stator current must be higher at full load than half load."""
    r50 = model.predict({"load_fraction": 0.5})
    r100 = model.predict({"load_fraction": 1.0})
    assert float(r100["stator_current_A"]) > float(r50["stator_current_A"])


def test_derating_unity_below_threshold(model):
    r = model.predict({"load_fraction": 1.0, "ambient_temperature": 35.0})
    assert float(r["derating_factor"]) == 1.0


def test_derating_below_unity_above_threshold(model):
    r = model.predict({"load_fraction": 1.0, "ambient_temperature": 55.0})
    assert float(r["derating_factor"]) < 1.0


def test_vectorized(model):
    plr = np.linspace(0.1, 1.2, 100)
    r = model.predict({"load_fraction": plr})
    assert len(r["efficiency"]) == 100


def test_benchmark(model):
    plr = np.random.uniform(0.05, 1.2, 1000)
    start = time.perf_counter()
    model.predict({"load_fraction": plr})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
